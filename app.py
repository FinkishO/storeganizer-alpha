"""
Storeganizer Planning Tool (Streamlit)

Refactored to use the modular architecture:
- core.eligibility for SKU filtering
- core.allocation for planning metrics and layout generation
- core.data_ingest for flexible CSV/Excel parsing
- rag.rag_service for Lena chat
- visualization.planogram_2d for planogram rendering

This app focuses exclusively on Storeganizer (no multi-solution branching).
"""

from __future__ import annotations

import io
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

from core import allocation, data_ingest, eligibility, exports
from core.article_library import ArticleLibrary
from core.roi_calculator import calculate_roi, ROIInputs, ROIResults
from rag import rag_service
from visualization import planogram_2d
from visualization.viewer_3d import embed_3d_viewer, get_configuration_suggestions
from config import storeganizer as config

# Auto-initialize RAG database if missing (for Streamlit Cloud deployment)
if not Path("rag_store.db").exists():
    try:
        from rag.ingest_ref import ingest
        ingest()
    except Exception:
        pass  # Silently fail if RAG init fails

# Initialize article library (builds from master file if not exists)
@st.cache_resource
def get_article_library():
    """Load article library (cached across app sessions)."""
    library = ArticleLibrary()

    # Build library from master file if empty
    master_file = Path(__file__).parent / "ref" / "cp_input1.xlsx"
    if master_file.exists():
        stats = library.get_stats()
        if stats['total_articles'] == 0:
            library.import_from_excel(str(master_file), sheet_name='rpt')

    return library


APP_TITLE = "Storeganizer Planning Tool"
APP_TAGLINE = "Plan Storeganizer pocket storage layouts from raw inventory to bay-level planogram."
LENA_AVATAR_PATH = Path(__file__).parent / "source" / "lena2_img.png"

# Wizard step labels (Storeganizer-only flow)
WIZARD_STEPS = {
    1: "Upload",
    2: "Select Analysis",
    3: "Configure",
    4: "Results",
    5: "ROI",
}


# ===========================
# Session State Management
# ===========================

def init_session_state():
    """Initialize session state with Storeganizer defaults."""
    defaults = {
        "wizard_step": 1,
        "plan_name": "Storeganizer Plan",
        "selected_service": None,
        "chat_history": [],
        "inventory_raw": None,
        "inventory_filtered": None,
        "rejected_items": None,
        "rejected_count": 0,
        "inventory_filename": None,
        "column_status": None,
        "rejection_reasons": {},
        "blocks": [],
        "columns_summary": None,
        "planning_df": None,
        "data_quality": None,
        "smart_rejections": [],
        "recommended_config": None,
        "processing_started": False,
        "processing_done": False,
        "processing_error": None,
        # Configurable parameters (default from config)
        "selected_config_size": config.DEFAULT_CONFIG_SIZE,
        "pocket_width": config.DEFAULT_POCKET_WIDTH,
        "pocket_depth": config.DEFAULT_POCKET_DEPTH,
        "pocket_height": config.DEFAULT_POCKET_HEIGHT,
        "pocket_weight_limit": config.DEFAULT_POCKET_WEIGHT_LIMIT,
        "columns_per_bay": config.DEFAULT_COLUMNS_PER_BAY,
        "rows_per_column": config.DEFAULT_ROWS_PER_COLUMN,
        "max_weight_per_column": config.DEFAULT_COLUMN_WEIGHT_LIMIT,
        "num_bays": 5,
        "show_custom_config": False,
        "velocity_band_filter": "All",
        # Stockweeks filter settings (OFF by default - user enables in Step 3)
        "min_stockweeks": config.MIN_STOCKWEEKS,
        "max_stockweeks": config.MAX_STOCKWEEKS,
        "use_stockweeks_filter": False,  # OFF by default - no velocity filtering unless user enables
        "allow_extra_width": config.ALLOW_SQUEEZE_PACKAGING,
        "remove_fragile": config.DEFAULT_REMOVE_FRAGILE,
        "bay_count": 5,
        "elig_max_w": config.DEFAULT_POCKET_WIDTH,
        "elig_max_d": config.DEFAULT_POCKET_DEPTH,
        "elig_max_h": config.DEFAULT_POCKET_HEIGHT,
        "elig_custom_override": False,
        # ROI Calculator inputs (defaults from ROIInputs)
        "roi_rack_width": 2.7,
        "roi_rack_depth": 1.0,
        "roi_locations_before": 40,
        "roi_locations_after": 90,
        "roi_num_racks_before": 30,
        "roi_num_racks_after": 15,
        "roi_cost_per_sqm": 90.0,
        "roi_utility_per_sqm": 10.0,
        "roi_num_staff": 2,
        "roi_wage_per_hour": 28.0,
        "roi_efficiency_increase": 15.0,  # Store as percentage (15% = 15.0)
        "roi_investment": 35000.0,
        "roi_results": None,
        # Output template settings (column selection for eligibility preview)
        "output_template_name": "Default",
        "output_visible_columns": None,  # None = show all columns
        "output_templates": {
            "Default": None,  # Show all
            "Compact": ["sku_code", "description", "pocket_size", "stockweeks"],
            "Dimensions": ["sku_code", "description", "width_mm", "depth_mm", "height_mm", "weight_kg", "pocket_size"],
            "Planning": ["sku_code", "description", "pocket_size", "assq_units", "stockweeks", "weekly_demand", "velocity_band"],
        },
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_inventory_state():
    """Clear inventory-dependent session data."""
    st.session_state["inventory_raw"] = None
    st.session_state["inventory_filtered"] = None
    st.session_state["rejected_items"] = None
    st.session_state["rejected_count"] = 0
    st.session_state["inventory_filename"] = None
    st.session_state["column_status"] = None
    st.session_state["rejection_reasons"] = {}
    st.session_state["blocks"] = []
    st.session_state["columns_summary"] = None
    st.session_state["planning_df"] = None
    st.session_state["data_quality"] = None
    st.session_state["smart_rejections"] = []
    st.session_state["recommended_config"] = None
    st.session_state["processing_started"] = False
    st.session_state["processing_done"] = False
    st.session_state["processing_error"] = None


def reset_processing_outputs():
    """Reset processing outputs while keeping the uploaded file."""
    st.session_state["inventory_filtered"] = None
    st.session_state["rejected_items"] = None
    st.session_state["rejected_count"] = 0
    st.session_state["rejection_reasons"] = {}
    st.session_state["blocks"] = []
    st.session_state["columns_summary"] = None
    st.session_state["planning_df"] = None
    st.session_state["data_quality"] = None
    st.session_state["smart_rejections"] = []
    st.session_state["recommended_config"] = None
    st.session_state["processing_started"] = False
    st.session_state["processing_done"] = False
    st.session_state["processing_error"] = None

def discard_changes_and_home():
    """Reset session to defaults and return to Welcome."""
    st.session_state.clear()
    init_session_state()
    st.session_state["wizard_step"] = 1


# ===========================
# Utility helpers
# ===========================

def download_excel(label: str, df: pd.DataFrame, filename: str, help: str | None = None, key: str = None):
    """Render an Excel download button for a dataframe."""
    if df is None or df.empty:
        st.button(label, disabled=True, help="Nothing to download yet", key=key)
        return

    xlsx_bytes = exports.export_to_excel(df)
    st.download_button(
        label=label,
        data=xlsx_bytes,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help=help,
        key=key,
    )


def blocks_to_dataframe(blocks: List[allocation.CellBlock]) -> pd.DataFrame:
    """
    Convert CellBlock list to DataFrame for export.

    Format matches old project output:
    - Article Number, Article Name, Bay, Column, Row, Cell Label, Weight kg
    - Column and Row are 1-based (not 0-based)
    - Cell Label: B{bay:02d}-C{column:02d}-R{row:02d}
    - For multi-row blocks (row_span > 1), expand into multiple rows
    """
    if not blocks:
        return pd.DataFrame()

    records = []
    for block in blocks:
        # If block spans multiple rows, create one record per row
        for row_offset in range(block.row_span):
            row_1based = block.row_start + row_offset + 1  # Convert to 1-based
            column_1based = block.column_index + 1  # Convert to 1-based

            records.append({
                "Article Number": block.sku_code,
                "Article Name": block.description,
                "Bay": block.bay,
                "Column": column_1based,
                "Row": row_1based,
                "Cell Label": f"B{block.bay:02d}-C{column_1based:02d}-R{row_1based:02d}",
                "Velocity Band": block.velocity_band,
                "Units in Block": block.units_in_block,
                "Column Weight kg": round(block.column_weight_kg, 2),
                "Overweight Flag": block.overweight_flag,
            })

    return pd.DataFrame(records)


def render_stepper():
    """Show wizard progress with clearer hierarchy."""
    current = st.session_state["wizard_step"]
    total_steps = len(WIZARD_STEPS)
    progress_pct = int((current - 1) / (total_steps - 1) * 100) if total_steps > 1 else 0

    top_cols = st.columns([3, 1])
    with top_cols[0]:
        st.markdown(
            f"<div style='font-size:20px;font-weight:700;'>Step {current} of {total_steps}</div>",
            unsafe_allow_html=True,
        )
    with top_cols[1]:
        st.progress(progress_pct / 100)

    cols = st.columns(total_steps)
    for idx, (step, label) in enumerate(WIZARD_STEPS.items()):
        state = "current" if step == current else "done" if step < current else "pending"
        color = {"current": "#0f766e", "done": "#4b5563", "pending": "#9ca3af"}[state]
        weight = "700" if state == "current" else "500"
        cols[idx].markdown(
            f"<div style='text-align:center;color:{color};font-weight:{weight};'>"
            f"{step}. {label}</div>",
            unsafe_allow_html=True,
        )

def render_top_nav():
    """Quick-access navigation so Next isn't buried after long content."""
    step = st.session_state.get("wizard_step", 1)
    total = len(WIZARD_STEPS)
    # Don't show top nav on Step 1 (has its own navigation)
    if step == 1 or step >= total:
        return
    can_go_next = can_advance(step)
    cols = st.columns([5, 1])
    with cols[1]:
        st.button(
            "Next",
            key=f"top_next_{step}",
            type="primary",
            disabled=not can_go_next,
            on_click=lambda: go_to_step(step + 1),
        )


def go_to_step(step: int):
    """Update wizard step within bounds."""
    step = max(1, min(step, len(WIZARD_STEPS)))
    st.session_state["wizard_step"] = step


def can_advance(step: int) -> bool:
    """Gate Next navigation so steps aren't skipped."""
    if step == 1:
        return st.session_state.get("inventory_raw") is not None
    if step == 2:
        return bool(st.session_state.get("selected_service"))
    if step == 3:
        return bool(st.session_state.get("processing_done"))
    return True


def format_metric(value, suffix=""):
    try:
        return f"{int(value):,}{suffix}"
    except Exception:
        return f"{value}{suffix}"


ASSQ_MULTIPACK_COLUMNS = [
    "multibox_qty",
    "multi_box_qty",
    "multipack_qty",
    "case_qty",
    "inner_pack_qty",
    "pack_qty",
    "package_qty",
]


def _first_multibox_qty(row: dict) -> int:
    for col in ASSQ_MULTIPACK_COLUMNS:
        val = row.get(col)
        try:
            qty = int(val)
            if qty > 0:
                return qty
        except Exception:
            continue
    return 0


def add_assq_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ASSQ (units per column) based on pocket dimensions and SKU sizing.

    Logic:
    - Calculate how many single units fit by dimension (floor on each axis).
    - If a multibox quantity exists and fits within that count, use it.
    - If multibox quantity exists but doesn't fit, fall back to single-unit count and flag for review.
    - Always ensure at least 1.
    """
    if df is None or len(df) == 0:
        return df

    df_work = df.copy()
    pocket_w = float(st.session_state.get("pocket_width", config.DEFAULT_POCKET_WIDTH))
    pocket_d = float(st.session_state.get("pocket_depth", config.DEFAULT_POCKET_DEPTH))
    pocket_h = float(st.session_state.get("pocket_height", config.DEFAULT_POCKET_HEIGHT))

    assq_values: list[int] = []
    assq_source: list[str] = []
    assq_review: list[bool] = []

    for row in df_work.to_dict(orient="records"):
        w = float(row.get("width_mm", 0) or 0)
        d = float(row.get("depth_mm", 0) or 0)
        h = float(row.get("height_mm", 0) or 0)

        # Dimension-based stacking (simple axis fit)
        if w > 0 and d > 0 and h > 0:
            fit_w = int(pocket_w // w)
            fit_d = int(pocket_d // d)
            fit_h = int(pocket_h // h)
            units_by_dim = max(0, fit_w * fit_d * fit_h)
        else:
            units_by_dim = 0

        multi_qty = _first_multibox_qty(row)
        needs_review = False

        if multi_qty > 0 and units_by_dim >= multi_qty:
            assq = int(multi_qty)
            source = "multibox"
        elif multi_qty > 0 and units_by_dim < multi_qty:
            assq = max(1, units_by_dim)
            source = "multibox_fallback"
            needs_review = True
        else:
            assq = max(1, units_by_dim)
            source = "single_stack"
            needs_review = units_by_dim <= 1

        assq_values.append(int(assq))
        assq_source.append(source)
        assq_review.append(bool(needs_review))

    df_work["assq_units"] = assq_values
    df_work["assq_source"] = assq_source
    df_work["assq_needs_review"] = assq_review
    return df_work


def get_active_pocket_limits():
    """
    Return the current pocket limits based on selected config or overrides.
    Ensures Step 4 uses the same dimensions chosen in Step 2.
    """
    sel = st.session_state.get("selected_config_size", config.DEFAULT_CONFIG_SIZE)
    cfg = config.STANDARD_CONFIGS.get(sel)
    if cfg:
        return (
            float(cfg["pocket_width"]),
            float(cfg["pocket_depth"]),
            float(cfg["pocket_height"]),
            float(cfg["pocket_weight_limit"]),
        )
    return (
        float(st.session_state.get("pocket_width", config.DEFAULT_POCKET_WIDTH)),
        float(st.session_state.get("pocket_depth", config.DEFAULT_POCKET_DEPTH)),
        float(st.session_state.get("pocket_height", config.DEFAULT_POCKET_HEIGHT)),
        float(st.session_state.get("pocket_weight_limit", config.DEFAULT_POCKET_WEIGHT_LIMIT)),
    )


def get_chat_context() -> Dict:
    """Build context dict for Lena based on current planning state."""
    filtered = st.session_state.get("inventory_filtered")
    columns_summary = st.session_state.get("columns_summary")
    overweight = 0
    if columns_summary is not None and not columns_summary.empty:
        overweight = int(columns_summary.get("overweight_flag", pd.Series(dtype=bool)).sum())
    return {
        "preset": st.session_state.get("plan_name", "Storeganizer plan"),
        "bay_count": st.session_state.get("bay_count"),
        "overweight_count": overweight,
        "reject_count": st.session_state.get("rejected_count", 0),
        "sku_count": len(filtered) if isinstance(filtered, pd.DataFrame) else None,
        "columns_per_bay": st.session_state.get("columns_per_bay"),
        "rows_per_column": st.session_state.get("rows_per_column"),
        "max_weight_per_column": st.session_state.get("max_weight_per_column"),
    }


def analyze_data_quality(df: pd.DataFrame) -> dict:
    """Assess data completeness and dimensional health for dashboard messaging."""
    if df is None or len(df) == 0:
        return {"rows": 0, "missing": {}, "alerts": ["No data loaded yet."], "dimension_stats": {}}

    required_cols = ["sku_code", "description", "width_mm", "depth_mm", "height_mm", "weight_kg", "weekly_demand"]
    missing = {}
    for col in required_cols:
        if col not in df.columns:
            missing[col] = len(df)
        else:
            missing[col] = int(df[col].isna().sum())

    dimension_cols = ["width_mm", "depth_mm", "height_mm", "weight_kg"]
    stats = {}
    for col in dimension_cols:
        series = pd.to_numeric(df.get(col), errors="coerce")
        stats[col] = {
            "avg": float(series.mean()) if len(series) else 0.0,
            "max": float(series.max()) if len(series) else 0.0,
            "p95": float(series.quantile(0.95)) if len(series) else 0.0,
            "missing": int(series.isna().sum()) if len(series) else len(df),
            "zero_or_negative": int((series <= 0).sum()) if len(series) else len(df),
        }

    total_cells = len(required_cols) * len(df)
    total_missing = sum(missing.values())
    completeness_score = max(0, 100 - int((total_missing / total_cells) * 100)) if total_cells else 0

    alerts = []
    if completeness_score < 85:
        alerts.append("Fill missing dimensions and weights to improve eligibility confidence.")
    if stats["weight_kg"]["p95"] > config.DEFAULT_POCKET_WEIGHT_LIMIT:
        alerts.append("High-weight tail detected — double-check heavy SKUs or split packs.")
    if stats["width_mm"]["p95"] > config.DEFAULT_POCKET_WIDTH:
        alerts.append("Width-heavy assortment — Large pockets may unlock more fit.")

    return {
        "rows": len(df),
        "missing": missing,
        "dimension_stats": stats,
        "completeness_score": completeness_score,
        "alerts": alerts,
    }


def get_smart_rejection_analysis(rejected_df: pd.DataFrame, reasons: dict, config_profile: dict | None) -> List[dict]:
    """Provide contextual rejection breakdown with recovery ideas."""
    if rejected_df is None or len(rejected_df) == 0:
        return []

    cfg = config_profile or config.STANDARD_CONFIGS.get(config.DEFAULT_CONFIG_SIZE, {})
    width_limit = cfg.get("pocket_width", config.DEFAULT_POCKET_WIDTH)
    depth_limit = cfg.get("pocket_depth", config.DEFAULT_POCKET_DEPTH)
    height_limit = cfg.get("pocket_height", config.DEFAULT_POCKET_HEIGHT)
    weight_limit = cfg.get("pocket_weight_limit", config.DEFAULT_POCKET_WEIGHT_LIMIT)

    def numeric(series_name: str):
        return pd.to_numeric(rejected_df.get(series_name), errors="coerce")

    width_series = numeric("width_mm")
    depth_series = numeric("depth_mm")
    height_series = numeric("height_mm")
    weight_series = numeric("weight_kg")
    demand_series = numeric("weekly_demand")

    def recovery_config(value: float, dimension: str) -> dict | None:
        for _, candidate in config.STANDARD_CONFIGS.items():
            limit = candidate.get(f"pocket_{dimension}", 0)
            if value <= limit:
                return candidate
        return None

    insights: List[dict] = []

    width_over = width_series > width_limit
    if width_over.any():
        avg_width = float(width_series[width_over].mean())
        alt_cfg = recovery_config(avg_width, "width")
        suggestion = "Consider a custom layout."
        if alt_cfg:
            alt_limit = alt_cfg.get("pocket_width", width_limit)
            fit_alt_count = int((width_series[width_over] <= alt_limit).sum())
            suggestion = f"{fit_alt_count} would fit {alt_cfg.get('name', '')} ({alt_limit}mm width)."
        insights.append({
            "title": "Too wide for current pocket",
            "detail": f"{int(width_over.sum())} SKUs average {avg_width:.0f}mm vs {width_limit}mm limit.",
            "suggestion": suggestion,
        })

    depth_over = depth_series > depth_limit
    if depth_over.any():
        avg_depth = float(depth_series[depth_over].mean())
        alt_cfg = recovery_config(avg_depth, "depth")
        suggestion = "Consider rotating packaging or a deeper pocket."
        if alt_cfg:
            alt_limit = alt_cfg.get("pocket_depth", depth_limit)
            fit_alt_count = int((depth_series[depth_over] <= alt_limit).sum())
            suggestion = f"{fit_alt_count} would fit {alt_cfg.get('name', '')} ({alt_limit}mm depth)."
        insights.append({
            "title": "Depth overages",
            "detail": f"{int(depth_over.sum())} SKUs average {avg_depth:.0f}mm depth vs {depth_limit}mm limit.",
            "suggestion": suggestion,
        })

    height_over = height_series > height_limit
    if height_over.any():
        avg_height = float(height_series[height_over].mean())
        alt_cfg = recovery_config(avg_height, "height")
        suggestion = "Consider a taller pocket or reducing carton height."
        if alt_cfg:
            alt_limit = alt_cfg.get("pocket_height", height_limit)
            fit_alt_count = int((height_series[height_over] <= alt_limit).sum())
            suggestion = f"{fit_alt_count} would fit {alt_cfg.get('name', '')} ({alt_limit}mm height)."
        insights.append({
            "title": "Height overages",
            "detail": f"{int(height_over.sum())} SKUs average {avg_height:.0f}mm height vs {height_limit}mm limit.",
            "suggestion": suggestion,
        })

    weight_over = weight_series > weight_limit
    if weight_over.any():
        avg_weight = float(weight_series[weight_over].mean())
        insights.append({
            "title": "Over weight limit",
            "detail": f"{int(weight_over.sum())} SKUs average {avg_weight:.1f}kg vs {weight_limit:.1f}kg limit.",
            "suggestion": "Split multi-packs or store heavy items in pallet flow/long-span.",
        })

    missing_dimensions = ((width_series <= 0) | (depth_series <= 0) | (height_series <= 0))
    if missing_dimensions.any():
        insights.append({
            "title": "Missing or zero dimensions",
            "detail": f"{int(missing_dimensions.sum())} SKUs were rejected with missing width/depth/height data.",
            "suggestion": "Fill in dimensions for these articles to unlock eligibility.",
        })

    # Stockweeks-based velocity detection (only if stockweeks data available)
    if "stockweeks" in rejected_df.columns:
        stockweeks_series = pd.to_numeric(rejected_df.get("stockweeks"), errors="coerce")
        min_sw = st.session_state.get("min_stockweeks", config.MIN_STOCKWEEKS)
        max_sw = st.session_state.get("max_stockweeks", config.MAX_STOCKWEEKS)

        # Fast movers (below min stockweeks)
        too_fast = stockweeks_series < min_sw
        if too_fast.any():
            fast_count = int(too_fast.sum())
            avg_sw = float(stockweeks_series[too_fast].mean())
            insights.append({
                "title": "Fast movers excluded",
                "detail": f"{fast_count} SKUs have stockweeks below {min_sw} (avg: {avg_sw:.1f} weeks).",
                "suggestion": "These sell too fast for Storeganizer. Keep in carton flow or lower min stockweeks in config.",
            })

        # Slow movers (above max stockweeks)
        too_slow = stockweeks_series > max_sw
        if too_slow.any():
            slow_count = int(too_slow.sum())
            avg_sw = float(stockweeks_series[too_slow].mean())
            insights.append({
                "title": "Slow movers excluded",
                "detail": f"{slow_count} SKUs have stockweeks above {max_sw} (avg: {avg_sw:.1f} weeks).",
                "suggestion": "These are too slow for Storeganizer pockets. Consider archive storage or raise max stockweeks.",
            })

    if not insights and reasons:
        insights.append({
            "title": "Eligibility guardrails applied",
            "detail": f"Items filtered: {reasons}",
            "suggestion": "Adjust limits in a future release or refine source data.",
        })

    return insights


def choose_recommended_config(eligible_df: pd.DataFrame) -> dict:
    """
    Select the recommended Storeganizer configuration based on cascading allocation results.

    With cascading allocation, articles are assigned to smallest fitting pocket (XS/S/M/L).
    Recommendation prioritizes the most common pocket size in the mix.
    """
    if eligible_df is None or len(eligible_df) == 0:
        cfg = config.STANDARD_CONFIGS.get(config.DEFAULT_CONFIG_SIZE, {})
        return {"key": config.DEFAULT_CONFIG_SIZE, "config": cfg, "reason": "Defaulting to Medium until eligible SKUs are available."}

    # If pocket_size column exists (cascading allocation), use it to recommend
    if "pocket_size" in eligible_df.columns:
        pocket_counts = eligible_df["pocket_size"].value_counts()
        if not pocket_counts.empty:
            # Most common pocket size
            most_common = pocket_counts.idxmax()
            count = pocket_counts.max()
            pct = int((count / len(eligible_df)) * 100)

            # Map display name to config key
            size_key_map = {
                "XS": "xs",
                "Small": "small",
                "Medium": "medium",
                "Large": "large",
            }
            config_key = size_key_map.get(most_common, "medium")
            cfg = config.STANDARD_CONFIGS.get(config_key, config.STANDARD_CONFIGS["medium"])

            # Build distribution summary
            dist_str = ", ".join([f"{size}: {ct}" for size, ct in pocket_counts.items()])

            return {
                "key": config_key,
                "config": cfg,
                "reason": f"Cascading allocation: {pct}% use {most_common} pockets. Distribution: {dist_str}.",
            }

    # Fallback: original logic (if pocket_size not present)
    scores = []
    for key, cfg in config.STANDARD_CONFIGS.items():
        width_ok = pd.to_numeric(eligible_df.get("width_mm"), errors="coerce") <= cfg.get("pocket_width", 0)
        depth_ok = pd.to_numeric(eligible_df.get("depth_mm"), errors="coerce") <= cfg.get("pocket_depth", 0)
        height_ok = pd.to_numeric(eligible_df.get("height_mm"), errors="coerce") <= cfg.get("pocket_height", 0)
        weight_ok = pd.to_numeric(eligible_df.get("weight_kg"), errors="coerce") <= cfg.get("pocket_weight_limit", 0)
        fit_mask = width_ok & depth_ok & height_ok & weight_ok
        coverage = float(fit_mask.mean()) if len(fit_mask) else 0.0
        scores.append((coverage, cfg.get("cells_per_bay", 0), key, cfg))

    best = max(scores, key=lambda x: (x[0], x[1]))
    coverage_pct = int(best[0] * 100)
    return {
        "key": best[2],
        "config": best[3],
        "reason": f"Fits {coverage_pct}% of eligible SKUs with {best[3].get('cells_per_bay', 0)} pockets per bay.",
    }


def run_auto_processing_pipeline() -> bool:
    """Execute the auto-processing flow with sensible defaults."""
    raw_df = st.session_state.get("inventory_raw")
    if raw_df is None or len(raw_df) == 0:
        st.session_state["processing_error"] = "Upload an inventory file to continue."
        return False

    try:
        enriched_df = add_assq_columns(raw_df)

        if "sku_code" in enriched_df.columns:
            library = get_article_library()
            enriched_df, _ = library.enrich_dataframe(enriched_df, "sku_code")

        planning_ready_df = allocation.compute_planning_metrics(
            enriched_df,
            units_per_column=config.DEFAULT_UNITS_PER_COLUMN,
            max_weight_per_column_kg=config.DEFAULT_COLUMN_WEIGHT_LIMIT,
            per_sku_units_col="assq_units",
        )

        # Apply cascading pocket allocation (XS → S → M → L)
        # Each article is assigned to the smallest pocket that fits
        eligible_df, rejected_df, rejected_count, rejection_reasons = eligibility.apply_cascading_pocket_allocation(
            planning_ready_df,
            max_weight_kg=st.session_state.get("pocket_weight_limit", config.DEFAULT_POCKET_WEIGHT_LIMIT),
            velocity_band=st.session_state.get("velocity_band_filter", "All"),
            min_stockweeks=st.session_state.get("min_stockweeks", config.MIN_STOCKWEEKS),
            max_stockweeks=st.session_state.get("max_stockweeks", config.MAX_STOCKWEEKS),
            use_stockweeks_filter=st.session_state.get("use_stockweeks_filter", config.USE_STOCKWEEKS_FILTER),
            allow_squeeze=st.session_state.get("allow_extra_width", config.ALLOW_SQUEEZE_PACKAGING),
            remove_fragile=st.session_state.get("remove_fragile", config.DEFAULT_REMOVE_FRAGILE),
        )

        columns_required = int(eligible_df["columns_required"].sum()) if "columns_required" in eligible_df else len(eligible_df)
        bay_count = allocation.calculate_bay_requirements(
            sku_count=len(eligible_df),
            columns_per_bay=config.DEFAULT_COLUMNS_PER_BAY,
            rows_per_column=config.DEFAULT_ROWS_PER_COLUMN,
            columns_required_total=columns_required,
        ) if len(eligible_df) > 0 else 0

        planning_df, blocks, columns_summary = pd.DataFrame(), [], pd.DataFrame()
        if len(eligible_df) > 0:
            planning_df, blocks, columns_summary = allocation.build_layout(
                eligible_df,
                bays=bay_count or 1,
                columns_per_bay=config.DEFAULT_COLUMNS_PER_BAY,
                rows_per_column=config.DEFAULT_ROWS_PER_COLUMN,
                units_per_column=config.DEFAULT_UNITS_PER_COLUMN,
                max_weight_per_column_kg=config.DEFAULT_COLUMN_WEIGHT_LIMIT,
                per_sku_units_col="assq_units",
            )

        st.session_state["inventory_filtered"] = eligible_df
        st.session_state["rejected_items"] = rejected_df
        st.session_state["rejected_count"] = rejected_count
        st.session_state["rejection_reasons"] = rejection_reasons
        st.session_state["planning_df"] = planning_df
        st.session_state["blocks"] = blocks
        st.session_state["columns_summary"] = columns_summary
        st.session_state["bay_count"] = bay_count
        st.session_state["num_bays"] = bay_count

        st.session_state["data_quality"] = analyze_data_quality(enriched_df)
        recommended_cfg = choose_recommended_config(eligible_df)
        st.session_state["recommended_config"] = recommended_cfg
        st.session_state["smart_rejections"] = get_smart_rejection_analysis(rejected_df, rejection_reasons, recommended_cfg.get("config"))

        st.session_state["processing_error"] = None
        return True
    except Exception as exc:  # pragma: no cover - defensive
        st.session_state["processing_error"] = str(exc)
        return False


# ===========================
# Sidebar: Lena Chat
# ===========================

def render_lena_chat():
    """Sidebar with Lena's RAG chat."""
    step = st.session_state.get("wizard_step", 1)
    if step == 1:
        return  # Hide on welcome
    context = get_chat_context()
    with st.sidebar:
        col_l, col_img, col_r = st.columns([1, 2, 1])
        with col_img:
            st.image(str(LENA_AVATAR_PATH), width=180)
        with st.expander("Lena — Storeganizer Analyst", expanded=False):
            st.caption(config.LENA_PERSONA[:140] + "…")

        if st.button("Reset chat", key="reset_chat", use_container_width=True):
            st.session_state["chat_history"] = []

        # Step-aware suggested prompts
        prompts_by_step = {
            2: ["Which columns are required in the .xlsx file?", "Can I use the sample dataset?"],
            3: ["What happens during auto-processing?", "How are ASSQ units calculated?"],
            4: ["How do I interpret the rejection breakdown?", "How many bays do I need?"],
            5: ["How is ROI calculated?", "What factors affect payback period?"],
        }
        step_prompts = prompts_by_step.get(step, [])
        if step_prompts:
            st.markdown("**Suggested prompts**")
            for idx, prompt in enumerate(step_prompts):
                if st.button(
                    prompt,
                    key=f"suggest_{step}_{idx}",
                    use_container_width=True,
                ):
                    st.session_state["lena_input"] = prompt

        user_message = st.text_input("Ask Lena", key="lena_input", placeholder="e.g., What are the pocket limits?")
        if st.button("Send to Lena", key="send_lena", use_container_width=True):
            if user_message:
                answer, docs = rag_service.answer(user_message, context)
                st.session_state["chat_history"].append(("you", user_message))
                st.session_state["chat_history"].append(("lena", answer))

        if st.session_state["chat_history"]:
            st.markdown("---")
            for role, msg in st.session_state["chat_history"][::-1]:
                prefix = "🧑" if role == "you" else "🤖"
                st.markdown(f"**{prefix} {role.title()}:** {msg}")

        st.markdown("---")
        st.markdown("**Knowledge Areas**")
        for item in config.LENA_KNOWLEDGE_AREAS:
            st.markdown(f"- {item}")


# ===========================
# Step 2: Select Analysis
# ===========================

def render_step_service_selection():
    st.subheader("Step 2 — What do you want to do?")

    # Check if data is uploaded
    if st.session_state.get("inventory_raw") is None:
        st.warning("Upload data first.")
        return

    services = [
        {
            "key": "identify_scope",
            "name": "Identify Scope",
            "tagline": "Eligibility + ROI estimate",
            "available": True,
        },
        {
            "key": "top_50",
            "name": "Top 50 Analysis",
            "tagline": "Low-hanging fruit",
            "available": False,
        },
        {
            "key": "full_planning",
            "name": "Full Planogram",
            "tagline": "Complete bay layout",
            "available": False,
        },
    ]

    cols = st.columns(3)
    for idx, svc in enumerate(services):
        with cols[idx]:
            selected = st.session_state.get("selected_service") == svc["key"]
            st.markdown(f"**{svc['name']}**")
            st.caption(svc['tagline'])

            if svc["available"]:
                st.button(
                    "Select" if not selected else "✓ Selected",
                    key=f"svc_{svc['key']}",
                    type="primary" if selected else "secondary",
                    use_container_width=True,
                    on_click=lambda key=svc["key"]: (
                        st.session_state.update({"selected_service": key}),
                        go_to_step(3),
                    ),
                )
            else:
                st.button("Coming soon", key=f"svc_{svc['key']}", disabled=True, use_container_width=True)


# ===========================
# Step 1: Upload
# ===========================

def render_step_upload():
    st.subheader("Step 1 — Upload")
    uploaded_file = st.file_uploader("Upload .xlsx inventory file", type=["xlsx"], label_visibility="collapsed")

    if uploaded_file:
        try:
            reset_processing_outputs()
            df = data_ingest.load_inventory_file(uploaded_file)
            st.session_state["inventory_raw"] = df
            st.session_state["inventory_filename"] = uploaded_file.name
            st.session_state["column_status"] = data_ingest.get_column_status(df)
            st.success(f"Loaded {len(df):,} rows from {uploaded_file.name}")
        except ValueError as exc:
            st.error(f"File error: {exc}")
        except Exception as exc:  # pragma: no cover - defensive log
            st.error(f"Unexpected error reading file: {exc}")

    load_example = st.button("Load Example", key="load_example")
    if load_example:
        example_file_path = Path(__file__).parent / "sample_inventory.csv"
        if example_file_path.exists():
            try:
                reset_processing_outputs()
                df = data_ingest.load_inventory_file(str(example_file_path))
                st.session_state["inventory_raw"] = df
                st.session_state["inventory_filename"] = example_file_path.name
                st.session_state["column_status"] = data_ingest.get_column_status(df)
                st.success(f"✅ Loaded {len(df):,} example SKUs from included sample")
            except Exception as exc:
                st.error(f"Error loading example: {exc}")
        else:
            st.warning("Example file not found in repository.")

    if st.session_state.get("inventory_raw") is not None:
        status = st.session_state.get("column_status") or {}
        missing_required = status.get("required_missing", [])
        with st.expander("Column validation", expanded=False):
            col_a, col_b = st.columns(2)
            col_a.markdown("**Required present**")
            col_a.write(status.get("required_present", []))
            col_b.markdown("**Missing required**")
            col_b.write(missing_required or "None")
            st.markdown("**Optional columns detected**")
            st.write(status.get("optional_present", []))

        # Smart preview: show a representative sample, not just first 10 rows
        raw_df = st.session_state["inventory_raw"]
        # Quick summary line
        quality = analyze_data_quality(raw_df)
        st.success(f"Loaded {len(raw_df):,} SKUs | {quality.get('completeness_score', 0)}% complete")

        with st.expander("Data preview", expanded=False):
            stat_cols = st.columns(4)
            stat_cols[0].metric("Total SKUs", f"{len(raw_df):,}")

            if "weight_kg" in raw_df.columns:
                weights = pd.to_numeric(raw_df["weight_kg"], errors="coerce")
                under_limit = (weights <= config.DEFAULT_POCKET_WEIGHT_LIMIT).sum()
                stat_cols[1].metric("Weight OK", f"{under_limit:,}")

            if all(c in raw_df.columns for c in ["width_mm", "depth_mm", "height_mm"]):
                w = pd.to_numeric(raw_df["width_mm"], errors="coerce")
                d = pd.to_numeric(raw_df["depth_mm"], errors="coerce")
                h = pd.to_numeric(raw_df["height_mm"], errors="coerce")
                fits_default = ((w <= config.DEFAULT_POCKET_WIDTH) &
                               (d <= config.DEFAULT_POCKET_DEPTH) &
                               (h <= config.DEFAULT_POCKET_HEIGHT)).sum()
                stat_cols[2].metric("Fits Pocket", f"{fits_default:,}")

            stat_cols[3].metric("Completeness", f"{quality.get('completeness_score', 0)}%")

            st.dataframe(raw_df.head(10), height=200)

        if missing_required:
            st.warning("Missing required columns will block auto-processing.")

    # Navigation buttons
    nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 1])
    with nav_col1:
        if st.button("Reset", key="reset_upload_btn"):
            reset_inventory_state()
    with nav_col3:
        if st.button("Next →", type="primary", disabled=st.session_state.get("inventory_raw") is None):
            go_to_step(2)


# ===========================
# Step 3: Configuration & Processing
# ===========================

def render_step_auto_processing():
    st.subheader("Step 3 — Configure & Process")
    raw_df = st.session_state.get("inventory_raw")
    if raw_df is None:
        st.warning("Upload an .xlsx file first.")
        return

    # If already processed, show results
    if st.session_state.get("processing_done"):
        inv_filtered = st.session_state.get("inventory_filtered")
        eligible = 0 if inv_filtered is None else len(inv_filtered)
        rejected = st.session_state.get("rejected_count", 0)
        st.success(f"Processing complete. Eligible: {eligible:,} | Rejected: {rejected:,}.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Re-configure & Re-run", type="secondary", use_container_width=True):
                st.session_state["processing_done"] = False
                st.session_state["processing_started"] = False
                st.rerun()
        with col2:
            st.button("View results dashboard →", type="primary", on_click=lambda: go_to_step(4), use_container_width=True)
        return

    # === CONFIGURATION SECTION ===
    with st.expander("Stockweeks Filter (Optional)", expanded=False):
        st.caption("Filter articles by velocity - OFF by default (all articles eligible if they fit)")
        use_stockweeks = st.checkbox(
            "Enable stockweeks filter",
            value=st.session_state.get("use_stockweeks_filter", False),
            key="use_stockweeks_filter_input",
            help="Filter articles based on weeks of stock coverage (ASSQ/EWS)"
        )
        st.session_state["use_stockweeks_filter"] = use_stockweeks

        if use_stockweeks:
            col1, col2 = st.columns(2)
            with col1:
                min_sw = st.slider(
                    "Min stockweeks",
                    min_value=0.0, max_value=10.0, step=0.5,
                    value=st.session_state.get("min_stockweeks", 1.0),
                    key="min_stockweeks_input",
                    help="Minimum weeks of stock (items below this sell too fast)"
                )
                st.session_state["min_stockweeks"] = min_sw
            with col2:
                max_sw = st.slider(
                    "Max stockweeks",
                    min_value=4.0, max_value=52.0, step=1.0,
                    value=st.session_state.get("max_stockweeks", 26.0),
                    key="max_stockweeks_input",
                    help="Maximum weeks of stock (items above this are too slow)"
                )
                st.session_state["max_stockweeks"] = max_sw

    with st.expander("Other Filters", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            remove_fragile = st.checkbox(
                "Exclude fragile items",
                value=st.session_state.get("remove_fragile", False),
                key="remove_fragile_input",
                help="Remove items with 'glass', 'fragile', 'ceramic' in description"
            )
            st.session_state["remove_fragile"] = remove_fragile

            allow_squeeze = st.checkbox(
                "Allow soft packaging squeeze (10%)",
                value=st.session_state.get("allow_extra_width", False),
                key="allow_squeeze_input",
                help="Allow 10% extra width for compressible packaging"
            )
            st.session_state["allow_extra_width"] = allow_squeeze

        with col2:
            velocity = st.selectbox(
                "Velocity band filter",
                options=["All", "A", "B", "C"],
                index=["All", "A", "B", "C"].index(st.session_state.get("velocity_band_filter", "All")),
                key="velocity_band_input",
                help="A=fast movers, B=medium, C=slow movers"
            )
            st.session_state["velocity_band_filter"] = velocity

    # Pocket size selection removed - now using cascading allocation
    # Articles will be assigned to the smallest pocket that fits (XS → S → M → L)

    if st.session_state.get("processing_error"):
        st.error(st.session_state["processing_error"])

    # Constrain button width
    process_col1, process_col2, process_col3 = st.columns([1, 2, 1])
    with process_col2:
        run_processing = st.button("Run Processing", type="primary", use_container_width=True)

    if run_processing:
        st.session_state["processing_error"] = None
        with st.spinner("Running Storeganizer analysis..."):
            success = run_auto_processing_pipeline()
        if success:
            st.session_state["processing_done"] = True
            go_to_step(4)
            st.rerun()
        else:
            st.error(st.session_state.get("processing_error", "Processing failed. Check data and settings."))


# ===========================
# Step 4: Results Dashboard
# ===========================

def render_step_results_dashboard():
    st.subheader("Step 4 — Results Dashboard")
    filtered = st.session_state.get("inventory_filtered")
    rejected_df = st.session_state.get("rejected_items")
    quality = st.session_state.get("data_quality") or {}
    recommended_cfg = st.session_state.get("recommended_config") or {}

    eligible_count = len(filtered) if isinstance(filtered, pd.DataFrame) else 0
    rejected_count = len(rejected_df) if isinstance(rejected_df, pd.DataFrame) else st.session_state.get("rejected_count", 0)
    total_rows = quality.get("rows") or (eligible_count + rejected_count)

    summary_cols = st.columns(4)
    summary_cols[0].metric("Eligible SKUs", format_metric(eligible_count))
    summary_cols[1].metric("Rejected SKUs", format_metric(rejected_count))
    summary_cols[2].metric("Bay requirement", format_metric(st.session_state.get("bay_count", 0)))
    summary_cols[3].metric("Data completeness", f"{quality.get('completeness_score', 0)}%")

    # Pocket size distribution (if cascading allocation was used)
    if isinstance(filtered, pd.DataFrame) and "pocket_size" in filtered.columns:
        with st.expander("Pocket size distribution", expanded=True):
            # Order by size (XS → S → M → L), not alphabetically
            size_order = ["XS", "Small", "Medium", "Large"]
            pocket_counts = filtered["pocket_size"].value_counts()
            pocket_counts = pocket_counts.reindex([s for s in size_order if s in pocket_counts.index])
            if not pocket_counts.empty:
                pocket_cols = st.columns(len(pocket_counts))
                for idx, (size, count) in enumerate(pocket_counts.items()):
                    pocket_cols[idx].metric(f"{size}", format_metric(count))

                st.markdown("**Articles by pocket size:**")
                pocket_summary = filtered.groupby("pocket_size").agg({
                    "sku_code": "count",
                    "width_mm": "mean",
                    "depth_mm": "mean",
                    "height_mm": "mean",
                }).round(1)
                pocket_summary.columns = ["Articles", "Avg Width (mm)", "Avg Depth (mm)", "Avg Height (mm)"]
                st.dataframe(pocket_summary, width="stretch")

    cfg = recommended_cfg.get("config") or config.STANDARD_CONFIGS.get(config.DEFAULT_CONFIG_SIZE, {})
    cfg_name = cfg.get("name", "Medium")
    cfg_dims = f"{cfg.get('pocket_width', config.DEFAULT_POCKET_WIDTH)}×{cfg.get('pocket_depth', config.DEFAULT_POCKET_DEPTH)}×{cfg.get('pocket_height', config.DEFAULT_POCKET_HEIGHT)} mm"
    st.info(f"Recommended configuration: **{cfg_name}** ({cfg_dims}). {recommended_cfg.get('reason', '')}")

    if quality.get("alerts"):
        with st.expander("Data quality summary", expanded=True):
            st.markdown(f"Rows analyzed: {total_rows}")
            st.markdown("**Key alerts**")
            for alert in quality["alerts"]:
                st.markdown(f"- {alert}")
            missing = quality.get("missing", {})
            if missing:
                st.markdown("**Missing values**")
                st.write({k: v for k, v in missing.items() if v > 0})

    insights = st.session_state.get("smart_rejections", [])
    with st.expander("Rejection analysis", expanded=True):
        if insights:
            for insight in insights:
                st.markdown(f"**{insight['title']}** — {insight['detail']}")
                st.caption(f"Recovery: {insight['suggestion']}")
        else:
            st.markdown("No rejections. Great job!")

    if eligible_count == 0:
        st.error("No eligible SKUs yet. Review the rejection analysis and data quality alerts above.")
        reupload_col1, reupload_col2, reupload_col3 = st.columns([1, 1, 1])
        with reupload_col2:
            st.button("Re-upload data", on_click=lambda: go_to_step(1), use_container_width=True)
    else:
        # === ELIGIBILITY TABLE WITH COLUMN SELECTOR ===
        st.markdown("---")
        st.markdown("### Eligibility Results")

        # Get available columns from the filtered dataframe
        all_columns = list(filtered.columns)

        # Column selector UI
        with st.expander("📊 Customize columns (Output Template)", expanded=True):
            template_col, custom_col = st.columns([1, 2])

            with template_col:
                # Template selector
                templates = st.session_state.get("output_templates", {})
                template_names = list(templates.keys())

                selected_template = st.selectbox(
                    "Template",
                    options=template_names,
                    index=template_names.index(st.session_state.get("output_template_name", "Default")),
                    key="template_selector",
                    help="Pre-configured column sets for common use cases"
                )

                # Update session state when template changes
                if selected_template != st.session_state.get("output_template_name"):
                    st.session_state["output_template_name"] = selected_template
                    template_cols = templates.get(selected_template)
                    if template_cols:
                        # Filter to columns that exist in the data
                        st.session_state["output_visible_columns"] = [c for c in template_cols if c in all_columns]
                    else:
                        st.session_state["output_visible_columns"] = None  # Show all

            with custom_col:
                # Get current visible columns (from template or custom selection)
                current_visible = st.session_state.get("output_visible_columns")
                if current_visible is None:
                    current_visible = all_columns  # Show all by default

                # Multiselect for custom column selection
                visible_columns = st.multiselect(
                    "Visible columns",
                    options=all_columns,
                    default=[c for c in current_visible if c in all_columns],
                    key="column_selector",
                    help="Select which columns to display"
                )

                # Update session state when columns change
                if visible_columns != st.session_state.get("output_visible_columns"):
                    st.session_state["output_visible_columns"] = visible_columns
                    # If user customizes columns, switch to "Custom" template name
                    template_cols = templates.get(st.session_state.get("output_template_name"))
                    if template_cols and set(visible_columns) != set([c for c in template_cols if c in all_columns]):
                        st.session_state["output_template_name"] = "Custom"

        # Display the filtered dataframe with selected columns
        if visible_columns:
            display_df = filtered[visible_columns]
        else:
            display_df = filtered

        st.markdown(f"**Showing {len(display_df):,} eligible articles** ({len(visible_columns) if visible_columns else len(all_columns)} columns)")
        st.dataframe(display_df, height=400, use_container_width=True)

    # === Configuration Recommendation & Pricing ===
    st.markdown("---")
    with st.expander("📦 Configuration Recommendation & Pricing", expanded=True):
        rec_cols = st.columns([2, 1])

        with rec_cols[0]:
            st.markdown(f"### Recommended: **{cfg_name}**")
            st.markdown(f"**Pocket dimensions:** {cfg_dims}")
            st.markdown(f"**Cells per bay:** {cfg.get('cells_per_bay', 90)}")
            st.markdown(f"**Pockets per column:** {cfg.get('pockets_per_column', 6)}")
            st.markdown(f"**Rows deep:** {cfg.get('rows_deep', 3)}")

            reason = recommended_cfg.get("reason", "")
            if reason:
                st.info(f"💡 {reason}")

        with rec_cols[1]:
            bay_count = st.session_state.get("bay_count", 0)
            st.metric("Bays required", bay_count if bay_count > 0 else "—")
            cells_total = bay_count * cfg.get("cells_per_bay", 90) if bay_count > 0 else 0
            st.metric("Total pockets", f"{cells_total:,}" if cells_total > 0 else "—")

        # Size comparison table
        st.markdown("#### Size Comparison")
        size_data = []
        for size_key, size_cfg in config.STANDARD_CONFIGS.items():
            # Calculate fit percentage for this size
            eligible = st.session_state.get("inventory_filtered")
            fit_count = 0
            if isinstance(eligible, pd.DataFrame) and len(eligible) > 0:
                max_w = size_cfg.get("pocket_width", 450)
                max_d = size_cfg.get("pocket_depth", 300)
                max_h = size_cfg.get("pocket_height", 300)
                fit_mask = (
                    (eligible.get("width_mm", pd.Series([0])) <= max_w) &
                    (eligible.get("depth_mm", pd.Series([0])) <= max_d) &
                    (eligible.get("height_mm", pd.Series([0])) <= max_h)
                )
                fit_count = fit_mask.sum() if hasattr(fit_mask, 'sum') else 0

            size_data.append({
                "Size": size_cfg.get("name", size_key.capitalize()),
                "Pocket (WxDxH mm)": f"{size_cfg.get('pocket_width')}×{size_cfg.get('pocket_depth')}×{size_cfg.get('pocket_height')}",
                "Cells/Bay": size_cfg.get("cells_per_bay", "—"),
                "SKUs Fit": fit_count if fit_count > 0 else "—",
                "Recommended": "✓" if size_key.lower() == cfg_name.lower() or size_cfg.get("name", "").lower() == cfg_name.lower() else "",
            })
        st.dataframe(pd.DataFrame(size_data), width="stretch", hide_index=True)

        # Pricing CTA
        st.markdown("#### Pricing")
        st.warning(
            "💰 **Pricing is project-specific.** Contact Storeganizer for a custom quote based on:\n"
            "- Number of bays and configuration\n"
            "- Installation requirements\n"
            "- Location and logistics"
        )
        st.markdown(
            "📧 **Request a quote:** [sales@storeganizer.com](mailto:sales@storeganizer.com) "
            "or visit [storeganizer.com](https://storeganizer.com)"
        )

    # Actionable suggestions
    suggestions = []
    if rejected_count > 0 and insights:
        suggestions.append("Address the top rejection driver first (e.g., width or missing dimensions).")
    if quality.get("completeness_score", 0) < 90:
        suggestions.append("Fill missing width/depth/height fields and re-run auto-processing.")
    if cfg_name.lower() != "medium":
        suggestions.append(f"Consider ordering {cfg_name} pockets to better fit the article mix.")
    if bay_count > 0:
        suggestions.append(f"Request pricing for {bay_count} {cfg_name} bays from Storeganizer.")
    if not suggestions:
        suggestions.append("Download the XLSX report and share with ops for ordering.")

    st.markdown("### Next actions")
    for item in suggestions:
        st.markdown(f"- {item}")

    # Downloads
    st.markdown("### Downloads")
    _raw = st.session_state.get("inventory_raw")
    full_report = exports.create_full_article_report(
        original_df=_raw if isinstance(_raw, pd.DataFrame) else pd.DataFrame(),
        eligible_df=filtered if isinstance(filtered, pd.DataFrame) else pd.DataFrame(),
        rejection_reasons=st.session_state.get("rejection_reasons", {}),
        planning_df=st.session_state.get("planning_df"),
    )
    rejection_report = exports.create_rejection_report(rejected_df)

    # Get selected columns for custom export
    selected_cols = st.session_state.get("output_visible_columns")
    if selected_cols and isinstance(filtered, pd.DataFrame):
        # Filter to columns that exist in the dataframe
        export_cols = [c for c in selected_cols if c in filtered.columns]
        custom_export_df = filtered[export_cols] if export_cols else filtered
        template_name = st.session_state.get("output_template_name", "Custom")
    else:
        custom_export_df = filtered if isinstance(filtered, pd.DataFrame) else pd.DataFrame()
        template_name = "Full"

    download_cols = st.columns(3)
    with download_cols[0]:
        download_excel(
            label="📊 Full results",
            df=full_report,
            filename="storeganizer_results.xlsx",
            help="Eligible + rejected SKUs with status and reasoning",
            key="download_results",
        )
    with download_cols[1]:
        download_excel(
            label=f"📋 Eligible ({template_name})",
            df=custom_export_df,
            filename=f"storeganizer_eligible_{template_name.lower()}.xlsx",
            help=f"Eligible articles with {len(selected_cols) if selected_cols else 'all'} columns",
            key="download_eligible_custom",
        )
    with download_cols[2]:
        download_excel(
            label="🚫 Rejections",
            df=rejection_report,
            filename="storeganizer_rejections.xlsx",
            help="Items excluded with reasons and dimensions",
            key="download_rejections",
        )

    # ROI Analysis CTA
    st.markdown("---")
    st.info("💡 **Ready to calculate ROI?** Proceed to Step 5 to estimate financial impact, payback period, and annual savings.")

    roi_cta_col1, roi_cta_col2, roi_cta_col3 = st.columns([1, 2, 1])
    with roi_cta_col2:
        if st.button("Next: ROI Analysis →", type="primary", use_container_width=True):
            go_to_step(5)


# ===========================
# Step 5: ROI Analysis
# ===========================

def render_step_roi_analysis():
    """ROI Calculator - estimate financial impact of Storeganizer investment."""
    st.subheader("Step 5 — ROI Analysis")
    st.caption("Calculate projected savings, payback period, and return on investment.")

    # Try to pre-fill intelligent defaults from planning data
    bay_count = st.session_state.get("bay_count", 0)
    if bay_count > 0:
        # Smart defaults: use actual bay count from planning
        default_racks_after = bay_count
        default_racks_before = bay_count * 2  # Assume 2:1 consolidation
    else:
        default_racks_after = st.session_state.get("roi_num_racks_after", 15)
        default_racks_before = st.session_state.get("roi_num_racks_before", 30)

    st.markdown("### Investment Parameters")
    st.caption("Adjust these values to match your warehouse and cost structure.")

    # Input section with columns
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Space Configuration**")

        # Unit selection
        unit_system = st.radio(
            "Measurement units",
            ["Metric (m)", "Metric (cm)", "Imperial (inches)"],
            horizontal=True,
            help="Choose your preferred measurement unit for rack dimensions"
        )

        # Set defaults and parameters based on unit selection
        if unit_system == "Metric (m)":
            width_default = float(st.session_state.get("roi_rack_width", 2.7))
            depth_default = float(st.session_state.get("roi_rack_depth", 1.0))
            width_label = "Rack width (m)"
            depth_label = "Rack depth (m)"
            step_size = 0.1
            min_val = 0.1
            max_val = 10.0
        elif unit_system == "Metric (cm)":
            width_default = float(st.session_state.get("roi_rack_width", 2.7)) * 100
            depth_default = float(st.session_state.get("roi_rack_depth", 1.0)) * 100
            width_label = "Rack width (cm)"
            depth_label = "Rack depth (cm)"
            step_size = 10.0
            min_val = 10.0
            max_val = 1000.0
        else:  # Imperial (inches)
            width_default = float(st.session_state.get("roi_rack_width", 2.7)) * 39.3701
            depth_default = float(st.session_state.get("roi_rack_depth", 1.0)) * 39.3701
            width_label = "Rack width (inches)"
            depth_label = "Rack depth (inches)"
            step_size = 1.0
            min_val = 4.0
            max_val = 400.0

        rack_width_input = st.number_input(
            width_label,
            min_value=min_val,
            max_value=max_val,
            value=float(width_default),
            step=step_size,
            help=f"Width of each rack in {unit_system.split('(')[1].strip(')')}"
        )
        rack_depth_input = st.number_input(
            depth_label,
            min_value=min_val,
            max_value=max_val,
            value=float(depth_default),
            step=step_size,
            help=f"Depth of each rack in {unit_system.split('(')[1].strip(')')}"
        )

        # Convert to meters for calculation
        if unit_system == "Metric (cm)":
            rack_width = rack_width_input / 100
            rack_depth = rack_depth_input / 100
        elif unit_system == "Imperial (inches)":
            rack_width = rack_width_input * 0.0254
            rack_depth = rack_depth_input * 0.0254
        else:  # Metric (m)
            rack_width = rack_width_input
            rack_depth = rack_depth_input
        locations_before = st.number_input(
            "Locations per rack - before",
            min_value=1,
            max_value=1000,
            value=int(st.session_state.get("roi_locations_before", 40)),
            step=1,
            help="Storage locations per rack in current setup"
        )
        locations_after = st.number_input(
            "Locations per rack - after (Storeganizer)",
            min_value=1,
            max_value=1000,
            value=int(st.session_state.get("roi_locations_after", 90)),
            step=1,
            help="Storage locations per rack with Storeganizer"
        )
        num_racks_before = st.number_input(
            "Number of racks - before",
            min_value=1,
            max_value=10000,
            value=int(default_racks_before),
            step=1,
            help="Total racks needed in current setup"
        )
        num_racks_after = st.number_input(
            "Number of racks - after (Storeganizer)",
            min_value=1,
            max_value=10000,
            value=int(default_racks_after),
            step=1,
            help="Total racks needed with Storeganizer"
        )

    with col_b:
        st.markdown("**Cost Structure**")
        cost_per_sqm = st.number_input(
            "Annual cost per m² (€)",
            min_value=0.0,
            max_value=1000.0,
            value=float(st.session_state.get("roi_cost_per_sqm", 90.0)),
            step=1.0,
            help="Annual real estate cost per square meter"
        )
        utility_per_sqm = st.number_input(
            "Utility cost per m² annual (€)",
            min_value=0.0,
            max_value=1000.0,
            value=float(st.session_state.get("roi_utility_per_sqm", 10.0)),
            step=1.0,
            help="Annual utility cost per square meter"
        )

        st.markdown("**Labor Parameters**")
        num_staff = st.number_input(
            "Number of staff",
            min_value=1,
            max_value=1000,
            value=int(st.session_state.get("roi_num_staff", 2)),
            step=1,
            help="Number of picking staff"
        )
        wage_per_hour = st.number_input(
            "Wage per hour (€)",
            min_value=0.0,
            max_value=500.0,
            value=float(st.session_state.get("roi_wage_per_hour", 28.0)),
            step=0.5,
            help="Hourly wage per staff member"
        )
        efficiency_increase = st.slider(
            "Efficiency increase (%)",
            min_value=0.0,
            max_value=50.0,
            value=float(st.session_state.get("roi_efficiency_increase", 15.0)),
            step=1.0,
            help="Expected labor efficiency gain from Storeganizer (default: 15%)"
        )

        st.markdown("**Investment**")
        investment = st.number_input(
            "Total investment (€)",
            min_value=1.0,
            max_value=10000000.0,
            value=float(st.session_state.get("roi_investment", 35000.0)),
            step=1000.0,
            help="Total upfront investment in Storeganizer system"
        )

    # Store inputs in session state
    st.session_state["roi_rack_width"] = rack_width
    st.session_state["roi_rack_depth"] = rack_depth
    st.session_state["roi_locations_before"] = locations_before
    st.session_state["roi_locations_after"] = locations_after
    st.session_state["roi_num_racks_before"] = num_racks_before
    st.session_state["roi_num_racks_after"] = num_racks_after
    st.session_state["roi_cost_per_sqm"] = cost_per_sqm
    st.session_state["roi_utility_per_sqm"] = utility_per_sqm
    st.session_state["roi_num_staff"] = num_staff
    st.session_state["roi_wage_per_hour"] = wage_per_hour
    st.session_state["roi_efficiency_increase"] = efficiency_increase
    st.session_state["roi_investment"] = investment

    st.markdown("---")

    # Calculate button (constrained)
    calc_col1, calc_col2, calc_col3 = st.columns([1, 2, 1])
    with calc_col2:
        calculate_clicked = st.button("📊 Calculate ROI", type="primary", use_container_width=True)

    if calculate_clicked:
        try:
            # Build inputs
            roi_inputs = ROIInputs(
                rack_width_m=rack_width,
                rack_depth_m=rack_depth,
                locations_per_rack_before=locations_before,
                locations_per_rack_after=locations_after,
                num_racks_before=num_racks_before,
                num_racks_after=num_racks_after,
                cost_per_sqm_annual=cost_per_sqm,
                utility_cost_per_sqm_annual=utility_per_sqm,
                num_staff=num_staff,
                wage_per_hour=wage_per_hour,
                efficiency_increase_pct=efficiency_increase / 100.0,  # Convert percentage to decimal
                total_investment=investment,
            )

            # Calculate
            results = calculate_roi(roi_inputs)
            st.session_state["roi_results"] = results

        except ValueError as e:
            st.error(f"Input validation error: {e}")
        except Exception as e:
            st.error(f"Calculation error: {e}")

    # Display results if available
    results = st.session_state.get("roi_results")
    if results:
        st.markdown("---")
        st.markdown("### ROI Results")

        # Key metrics in columns (top row)
        metric_cols = st.columns(4)
        metric_cols[0].metric(
            "Annual Savings",
            f"€{results.total_annual_saving:,.0f}",
            help="Total annual savings (space + utilities + labor)"
        )
        metric_cols[1].metric(
            "ROI",
            f"{results.roi_pct:.1f}%",
            help="Return on Investment percentage"
        )
        metric_cols[2].metric(
            "Payback Period",
            f"{results.payback_years:.2f} years",
            help="Years to recover investment"
        )
        metric_cols[3].metric(
            "NPV (5-year)",
            f"€{results.npv:,.0f}",
            help="Net Present Value over 5 years"
        )

        # Storage density improvement (second row)
        st.markdown("---")
        density_cols = st.columns(3)
        density_improvement_pct = ((results.locations_per_sqm_after / results.locations_per_sqm_before) - 1) * 100
        density_cols[0].metric(
            "Storage Density Improvement",
            f"{results.locations_per_sqm_after:.1f} loc/m²",
            f"+{density_improvement_pct:.0f}% vs before",
            help="Storage locations per square meter after Storeganizer implementation"
        )
        density_cols[1].metric(
            "Total Locations Before",
            f"{results.total_locations_before:,}",
            help="Total storage locations in current setup"
        )
        density_cols[2].metric(
            "Total Locations After",
            f"{results.total_locations_after:,}",
            delta=f"+{results.total_locations_after - results.total_locations_before:,}" if results.total_locations_after > results.total_locations_before else f"{results.total_locations_after - results.total_locations_before:,}",
            delta_color="normal",
            help="Total storage locations with Storeganizer"
        )

        # Detailed breakdown
        with st.expander("📋 Detailed breakdown", expanded=True):
            st.markdown("**Space Optimization**")
            space_cols = st.columns(2)
            space_cols[0].write(f"**Square meters before:** {results.total_sqm_before:.1f} m²")
            space_cols[0].write(f"**Square meters after:** {results.total_sqm_after:.1f} m²")
            space_cols[0].write(f"**Space saved:** {results.sqm_saved:.1f} m² ({results.sqm_saved/results.total_sqm_before*100:.1f}% reduction)")
            space_cols[1].write(f"**Locations before:** {results.total_locations_before:,}")
            space_cols[1].write(f"**Locations after:** {results.total_locations_after:,}")
            space_cols[1].write(f"**Density improvement:** {results.locations_per_sqm_before:.1f} → {results.locations_per_sqm_after:.1f} locations/m² (+{density_improvement_pct:.0f}%)")

            st.markdown("---")
            st.markdown("**Annual Savings Breakdown**")
            savings_cols = st.columns(3)
            savings_cols[0].metric("Real Estate", f"€{results.annual_space_saving:,.0f}")
            savings_cols[1].metric("Utilities", f"€{results.annual_utility_saving:,.0f}")
            savings_cols[2].metric("Labor", f"€{results.annual_labor_saving:,.0f}")

            st.markdown("---")
            st.markdown("**Labor Impact**")
            labor_cols = st.columns(2)
            labor_cols[0].write(f"Annual staff cost before: €{results.annual_staff_cost_before:,.0f}")
            labor_cols[1].write(f"Annual staff cost after: €{results.annual_staff_cost_after:,.0f}")
            labor_cols[0].write(f"Estimated annual picks: {results.annual_picks:,}")
            labor_cols[1].write(f"Labor savings: €{results.annual_labor_saving:,.0f} ({efficiency_increase}% efficiency gain)")

        # Formatted summary
        with st.expander("📄 Full summary (export-ready)", expanded=False):
            st.code(results.format_summary(), language=None)

        # Warnings
        if results.warnings:
            with st.expander("⚠️  Warnings", expanded=True):
                for warning in results.warnings:
                    st.warning(warning)
    else:
        st.info("Click **Calculate ROI** above to see projected savings and payback period.")

    # Navigation
    st.markdown("---")
    nav_cols = st.columns([1, 1, 1])
    with nav_cols[0]:
        st.button("← Back to Results", on_click=lambda: go_to_step(4), use_container_width=True)
    with nav_cols[2]:
        if results:
            st.success("✅ ROI analysis complete. You can now export or restart the wizard.")


# ===========================
# Main
# ===========================

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="📦")
    init_session_state()

    st.title(APP_TITLE)
    render_stepper()

    # render_lena_chat()  # Disabled for now

    step = st.session_state["wizard_step"]
    with st.container():
        render_top_nav()
        if step == 1:
            render_step_upload()
        elif step == 2:
            render_step_service_selection()
        elif step == 3:
            render_step_auto_processing()
        elif step == 4:
            render_step_results_dashboard()
        elif step == 5:
            render_step_roi_analysis()
        else:
            render_step_upload()

    st.markdown("---")
    nav_cols = st.columns([1, 1, 1])
    next_disabled = step == len(WIZARD_STEPS) or not can_advance(step)
    with nav_cols[0]:
        # Only show Previous if not on Step 1
        if step > 1:
            st.button("Previous", key="nav_prev", on_click=lambda: go_to_step(step - 1), use_container_width=True)
    with nav_cols[1]:
        st.button("Home & Discard", key="nav_home_discard", on_click=discard_changes_and_home, use_container_width=True)
    with nav_cols[2]:
        # Only show Next if not on Step 1 (Step 1 has its own internal navigation)
        if step > 1:
            st.button(
                "Next",
                key="nav_next",
                disabled=next_disabled,
                type="primary",
                on_click=lambda: go_to_step(step + 1),
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
