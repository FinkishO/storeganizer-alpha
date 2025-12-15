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
from math import ceil
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

            # Detect priority/whitelist (separate sheets or columns)
            uploaded_file.seek(0)
            whitelist_info = data_ingest.detect_priority_whitelist(uploaded_file, df)
            st.session_state["whitelist_info"] = whitelist_info

            st.success(f"Loaded {len(df):,} rows from {uploaded_file.name}")
            if whitelist_info["detected"]:
                st.info(f"Detected priority list: {whitelist_info['description']}")
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
        raw_df = st.session_state["inventory_raw"]

        # Relevance-based health analysis
        health = exports.analyze_file_health(raw_df)
        total = health["total"]
        ready = health["ready"]
        needs_data = health["needs_data"]
        irrelevant = health["irrelevant"]
        relevant = health["relevant"]
        data_ranges = health["data_ranges"]
        columns_found = health["columns_found"]
        missing_breakdown = health.get("missing_breakdown", {})
        irrelevant_breakdown = health.get("irrelevant_breakdown", {})
        top_ready = health["top_ready"]
        top_needs_data = health["top_needs_data"]
        top_irrelevant = health["top_irrelevant"]

        # === RELEVANCE SUMMARY ===
        # Key insight: of RELEVANT articles, how many are ready vs need data?
        relevant_pct = int((relevant / total) * 100) if total > 0 else 0
        ready_of_relevant_pct = int((ready / relevant) * 100) if relevant > 0 else 0

        if irrelevant > 0:
            st.info(f"**{total:,} articles** — {irrelevant:,} too large for Storeganizer ({100 - relevant_pct}%)")

        if ready == relevant and relevant > 0:
            st.success(f"**{relevant:,} relevant articles** — All have complete data")
        elif ready > 0:
            st.warning(f"**{relevant:,} relevant articles** — {ready:,} ready, {needs_data:,} need data")
        elif needs_data > 0:
            st.error(f"**{relevant:,} relevant articles** — All missing data")
        else:
            st.error("No relevant articles found")

        # === COLUMN DETECTION ===
        col_status = []
        for col, found in columns_found.items():
            if col == "demand":
                if found:
                    col_status.append(f"✓ {found}")
                else:
                    col_status.append("○ demand")
            elif found:
                col_status.append(f"✓ {col}")
            else:
                col_status.append(f"✗ {col}")
        st.caption(f"Columns: {' · '.join(col_status)}")

        # === DATA RANGES (sanity check) ===
        if data_ranges:
            range_parts = []
            if "width_mm" in data_ranges:
                r = data_ranges["width_mm"]
                range_parts.append(f"W: {r['min']}-{r['max']}mm")
            if "depth_mm" in data_ranges:
                r = data_ranges["depth_mm"]
                range_parts.append(f"D: {r['min']}-{r['max']}mm")
            if "height_mm" in data_ranges:
                r = data_ranges["height_mm"]
                range_parts.append(f"H: {r['min']}-{r['max']}mm")
            if "weight_kg" in data_ranges:
                r = data_ranges["weight_kg"]
                range_parts.append(f"Weight: {r['min']}-{r['max']}kg")
            if range_parts:
                st.caption(f"Ranges: {' · '.join(range_parts)}")

        # === THREE-TAB PREVIEW: Ready / Needs Data / Irrelevant ===
        tab_ready, tab_needs, tab_irrelevant = st.tabs([
            f"Ready ({ready:,})",
            f"Needs Data ({needs_data:,})",
            f"Irrelevant ({irrelevant:,})"
        ])

        with tab_ready:
            if isinstance(top_ready, pd.DataFrame) and len(top_ready) > 0:
                st.dataframe(top_ready, height=250, width="stretch")
                if ready > 15:
                    st.caption(f"Showing 15 of {ready:,} ready articles")
            else:
                st.info("No articles ready yet")

        with tab_needs:
            if isinstance(top_needs_data, pd.DataFrame) and len(top_needs_data) > 0:
                # Show missing breakdown
                if missing_breakdown:
                    missing_lines = [f"{count:,} missing {field}" for field, count in missing_breakdown.items() if "forecast" not in field.lower()]
                    if missing_lines:
                        st.caption(" · ".join(missing_lines))
                st.dataframe(top_needs_data, height=250, width="stretch")
                if needs_data > 15:
                    st.caption(f"Showing 15 of {needs_data:,} articles needing data")
            else:
                st.success("All relevant articles have complete data")

        with tab_irrelevant:
            if isinstance(top_irrelevant, pd.DataFrame) and len(top_irrelevant) > 0:
                # Show irrelevant breakdown
                if irrelevant_breakdown:
                    irr_lines = [f"{count:,} {reason}" for reason, count in irrelevant_breakdown.items()]
                    if irr_lines:
                        st.caption(" · ".join(irr_lines))
                st.dataframe(top_irrelevant, height=250, width="stretch")
                if irrelevant > 15:
                    st.caption(f"Showing 15 of {irrelevant:,} oversized/overweight articles")
            else:
                st.info("All articles could potentially fit in Storeganizer pockets")

        # === DOWNLOAD BUTTON (only for RELEVANT articles needing data) ===
        if needs_data > 0:
            incomplete_excel = exports.create_incomplete_articles_export(raw_df, health)
            if incomplete_excel:
                st.download_button(
                    label=f"Download {needs_data:,} articles needing data",
                    data=incomplete_excel,
                    file_name="articles_needing_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="Only relevant articles that could fit - fill in dimensions/weight",
                    key="download_incomplete",
                )

        # === FULL DATA PREVIEW (collapsed) ===
        with st.expander("Full data preview", expanded=False):
            # Filter out excluded columns
            preview_cols = [c for c in raw_df.columns if not any(
                excl.lower() in c.lower() for excl in config.EXCLUDED_PREVIEW_COLUMNS
            ) and c not in config.EXCLUDED_SINGLE_COLUMNS]
            st.dataframe(raw_df[preview_cols].head(20), height=300, width="stretch")

    # Navigation buttons
    nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 1])
    with nav_col1:
        if st.button("Reset", key="reset_upload_btn"):
            reset_inventory_state()
    with nav_col3:
        if st.button("Next →", type="primary", key="step1_next", disabled=st.session_state.get("inventory_raw") is None):
            go_to_step(2)


# ===========================
# Step 3: Configuration & Processing
# ===========================

def render_step_auto_processing():
    st.subheader("Step 3 — Process")
    raw_df = st.session_state.get("inventory_raw")
    if raw_df is None:
        st.warning("Upload an .xlsx file first.")
        return

    # If already processed, show quick summary and nav
    if st.session_state.get("processing_done"):
        inv_filtered = st.session_state.get("inventory_filtered")
        eligible = 0 if inv_filtered is None else len(inv_filtered)
        rejected = st.session_state.get("rejected_count", 0)
        st.success(f"Processed: {eligible:,} eligible | {rejected:,} rejected")
        st.button("View Results →", type="primary", on_click=lambda: go_to_step(4), use_container_width=True)
        return

    # Simple processing - no config expanders, all refinement happens in Step 4
    st.caption("Cascading allocation: articles assigned to smallest fitting pocket (XS → S → M → L)")

    if st.session_state.get("processing_error"):
        st.error(st.session_state["processing_error"])

    # Centered run button
    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        run_processing = st.button("Run Processing", type="primary", key="run_processing_btn", use_container_width=True)

    if run_processing:
        st.session_state["processing_error"] = None
        with st.spinner("Running Storeganizer analysis..."):
            success = run_auto_processing_pipeline()
        if success:
            st.session_state["processing_done"] = True
            go_to_step(4)
            st.rerun()
        else:
            st.error(st.session_state.get("processing_error", "Processing failed."))


# ===========================
# Step 4: Results Dashboard
# ===========================

# Columns to EXCLUDE from eligibility display (IKEA internal fields)
COLUMNS_TO_EXCLUDE = [
    "pia facts", "pia_facts", "piafacts",
    "vat%", "vat_pct", "vat",
    "price ladder", "price_ladder", "priceladder",
    "style expression", "style_expression", "styleexpression",
    "4a+k", "4a_k", "4ak",
    "gm0%", "gm0_pct", "gm0",
    "gm55+", "gm55_plus", "gm55",
    "business approved", "business_approved", "businessapproved",
    "stocked",
    "wished presented", "wished_presented", "wishedpresented",
    "online order", "online_order", "onlineorder",
    "to be planned", "to_be_planned", "tobeplanned",
    "range allocation", "range_allocation", "rangeallocation",
]


def format_date_column_mmyyyy(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """Format a date column as MM-YYYY if it exists."""
    if col_name not in df.columns:
        return df
    df = df.copy()
    try:
        # Try to parse as datetime and format
        dates = pd.to_datetime(df[col_name], errors="coerce")
        df[col_name] = dates.dt.strftime("%m-%Y").fillna(df[col_name])
    except Exception:
        pass  # Keep original if parsing fails
    return df


def calculate_bay_requirements_by_size(filtered_df: pd.DataFrame) -> dict:
    """
    Calculate bay requirements per pocket size using actual cells_per_bay values.

    Returns dict with per-size breakdown and total.
    """
    if filtered_df is None or len(filtered_df) == 0 or "pocket_size" not in filtered_df.columns:
        return {"total": 0, "breakdown": {}}

    # Map pocket size names to config keys
    size_to_key = {"XS": "xs", "Small": "small", "Medium": "medium", "Large": "large"}

    breakdown = {}
    total_bays = 0

    for size_name, config_key in size_to_key.items():
        count = len(filtered_df[filtered_df["pocket_size"] == size_name])
        if count > 0:
            cfg = config.STANDARD_CONFIGS.get(config_key, {})
            cells_per_bay = cfg.get("cells_per_bay", 90)
            bays_needed = ceil(count / cells_per_bay)
            breakdown[size_name] = {
                "articles": count,
                "cells_per_bay": cells_per_bay,
                "bays_needed": bays_needed,
            }
            total_bays += bays_needed

    return {"total": total_bays, "breakdown": breakdown}


def filter_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove unwanted columns and format dates."""
    if df is None or len(df) == 0:
        return df

    df = df.copy()

    # Remove excluded columns (case-insensitive)
    cols_to_drop = []
    for col in df.columns:
        col_lower = col.lower().replace(" ", "").replace("_", "")
        for exclude in COLUMNS_TO_EXCLUDE:
            if exclude.replace(" ", "").replace("_", "") == col_lower:
                cols_to_drop.append(col)
                break
    df = df.drop(columns=cols_to_drop, errors="ignore")

    # Format SSD and EDS columns as MM-YYYY
    for col in df.columns:
        col_lower = col.lower()
        if "ssd" in col_lower or "eds" in col_lower:
            df = format_date_column_mmyyyy(df, col)

    return df


def rerun_with_refinements():
    """Re-run processing with refinement settings from Step 4."""
    # Store previous counts for delta display
    prev_filtered = st.session_state.get("inventory_filtered")
    st.session_state["prev_eligible_count"] = len(prev_filtered) if prev_filtered is not None else 0
    st.session_state["prev_rejected_count"] = st.session_state.get("rejected_count", 0)

    raw_df = st.session_state.get("inventory_raw")
    if raw_df is None:
        return False

    try:
        enriched_df = add_assq_columns(raw_df)
        if "sku_code" in enriched_df.columns:
            library = get_article_library()
            enriched_df, _ = library.enrich_dataframe(enriched_df, "sku_code")

        # Apply SM filter if any SM options selected
        sm_selection = st.session_state.get("refine_sm_selection", [])
        if sm_selection:
            sm_cols = [c for c in enriched_df.columns if "wished sm" in c.lower() or c.upper() in ["SM", "SALES METHOD", "STOCK MODEL"]]
            if sm_cols:
                sm_col = sm_cols[0]
                sm_vals = pd.to_numeric(enriched_df[sm_col], errors="coerce")
                # Filter to only selected SM values
                enriched_df = enriched_df[sm_vals.isin(sm_selection)].copy()

        # Apply whitelist/requested filter if enabled
        if st.session_state.get("refine_requested_only", False):
            whitelist_info = st.session_state.get("whitelist_info", {})
            if whitelist_info.get("detected") and whitelist_info.get("article_numbers"):
                whitelist_articles = set(str(a).strip() for a in whitelist_info["article_numbers"])
                # Find article column - prioritize sku_code, then article number patterns
                # NOTE: PA is Product Area (short code), NOT article number - skip it
                article_col = None
                if "sku_code" in enriched_df.columns:
                    article_col = "sku_code"
                else:
                    for col in enriched_df.columns:
                        col_lower = col.lower().strip()
                        if col_lower in ["article number", "article_number"] or ("article" in col_lower and "name" not in col_lower):
                            article_col = col
                            break
                if article_col:
                    enriched_df = enriched_df[enriched_df[article_col].astype(str).str.strip().isin(whitelist_articles)].copy()

        planning_ready_df = allocation.compute_planning_metrics(
            enriched_df,
            units_per_column=config.DEFAULT_UNITS_PER_COLUMN,
            max_weight_per_column_kg=config.DEFAULT_COLUMN_WEIGHT_LIMIT,
            per_sku_units_col="assq_units",
        )

        # Get refinement settings
        force_pocket = st.session_state.get("refine_pocket_size", "Auto")
        force_pocket_size = None if force_pocket == "Auto" else force_pocket

        eligible_df, rejected_df, rejected_count, rejection_reasons = eligibility.apply_cascading_pocket_allocation(
            planning_ready_df,
            max_weight_kg=st.session_state.get("pocket_weight_limit", config.DEFAULT_POCKET_WEIGHT_LIMIT),
            velocity_band=st.session_state.get("velocity_band_filter", "All"),
            min_stockweeks=st.session_state.get("refine_min_stockweeks", 1.0),
            max_stockweeks=st.session_state.get("refine_max_stockweeks", 26.0),
            use_stockweeks_filter=st.session_state.get("refine_use_stockweeks", False),
            allow_squeeze=st.session_state.get("allow_extra_width", config.ALLOW_SQUEEZE_PACKAGING),
            remove_fragile=st.session_state.get("refine_no_fragile", False),
            force_pocket_size=force_pocket_size,
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
        st.session_state["show_delta"] = True
        return True
    except Exception as exc:
        st.session_state["processing_error"] = str(exc)
        return False


def render_step_results_dashboard():
    st.subheader("Step 4 — Results")

    # === REFINEMENT BAR (always visible at top) ===
    st.markdown("##### Refine Selection")

    # Check for whitelist detection
    whitelist_info = st.session_state.get("whitelist_info", {})
    whitelist_detected = whitelist_info.get("detected", False)

    refine_cols = st.columns([2, 1, 1, 1, 1, 1])

    with refine_cols[0]:
        pocket_options = ["Auto", "XS", "Small", "Medium", "Large"]
        current_pocket = st.session_state.get("refine_pocket_size", "Auto")
        pocket_idx = pocket_options.index(current_pocket) if current_pocket in pocket_options else 0
        pocket_choice = st.radio(
            "Pocket",
            pocket_options,
            index=pocket_idx,
            horizontal=True,
            key="refine_pocket_radio",
            help="Auto = cascading (XS→S→M→L), or force single size"
        )
        st.session_state["refine_pocket_size"] = pocket_choice

    with refine_cols[1]:
        no_fragile = st.checkbox(
            "No fragile",
            value=st.session_state.get("refine_no_fragile", False),
            key="refine_fragile_cb",
            help="Exclude glass/ceramic/fragile items"
        )
        st.session_state["refine_no_fragile"] = no_fragile

    with refine_cols[2]:
        # SM filter - multiselect for SM 0/1/2
        st.markdown("**SM Filter**", help="Sales Method / Stock Model")
        sm_cols_container = st.columns(3)
        current_sm = st.session_state.get("refine_sm_selection", [])

        with sm_cols_container[0]:
            sm0 = st.checkbox("SM0", value=0 in current_sm, key="sm0_cb")
        with sm_cols_container[1]:
            sm1 = st.checkbox("SM1", value=1 in current_sm, key="sm1_cb")
        with sm_cols_container[2]:
            sm2 = st.checkbox("SM2", value=2 in current_sm, key="sm2_cb")

        sm_selection = []
        if sm0: sm_selection.append(0)
        if sm1: sm_selection.append(1)
        if sm2: sm_selection.append(2)
        st.session_state["refine_sm_selection"] = sm_selection

    with refine_cols[3]:
        # Requested/whitelist filter - only enabled if detected
        requested_only = st.checkbox(
            "Requested only",
            value=st.session_state.get("refine_requested_only", False),
            key="refine_requested_cb",
            disabled=not whitelist_detected,
            help=whitelist_info.get("description", "No priority list detected") if whitelist_detected else "No priority list detected in file"
        )
        st.session_state["refine_requested_only"] = requested_only if whitelist_detected else False

    with refine_cols[4]:
        use_sw = st.checkbox(
            "Stockweeks 1-26",
            value=st.session_state.get("refine_use_stockweeks", False),
            key="refine_stockweeks_cb",
            help="Filter by weeks of stock (1-26)"
        )
        st.session_state["refine_use_stockweeks"] = use_sw
        if use_sw:
            # Set sensible defaults
            if "refine_min_stockweeks" not in st.session_state:
                st.session_state["refine_min_stockweeks"] = 1.0
            if "refine_max_stockweeks" not in st.session_state:
                st.session_state["refine_max_stockweeks"] = 26.0

    with refine_cols[5]:
        if st.button("Regenerate", type="primary", key="regenerate_btn", use_container_width=True):
            with st.spinner("Re-processing..."):
                success = rerun_with_refinements()
            if success:
                st.rerun()
            else:
                st.error(st.session_state.get("processing_error", "Processing failed."))

    # Get data first (needed for delta and display)
    filtered = st.session_state.get("inventory_filtered")
    rejected_df = st.session_state.get("rejected_items")
    quality = st.session_state.get("data_quality") or {}

    eligible_count = len(filtered) if isinstance(filtered, pd.DataFrame) else 0
    rejected_count = len(rejected_df) if isinstance(rejected_df, pd.DataFrame) else st.session_state.get("rejected_count", 0)

    # Show delta after regenerate
    if st.session_state.get("show_delta"):
        prev_elig = st.session_state.get("prev_eligible_count", 0)
        delta = eligible_count - prev_elig
        if delta != 0:
            delta_str = f"+{delta}" if delta > 0 else str(delta)
            delta_color = "green" if delta > 0 else "red"
            st.markdown(f"<span style='color:{delta_color}'>Eligible: {delta_str} articles</span>", unsafe_allow_html=True)
        st.session_state["show_delta"] = False

    st.markdown("---")

    # Calculate bay requirements per pocket size (CORRECT calculation)
    bay_calc = calculate_bay_requirements_by_size(filtered)
    total_bays = bay_calc["total"]

    # Summary metrics
    summary_cols = st.columns(3)
    summary_cols[0].metric("Eligible", format_metric(eligible_count))
    summary_cols[1].metric("Rejected", format_metric(rejected_count))
    summary_cols[2].metric("Total Bays", format_metric(total_bays))

    # Pocket size distribution with bay breakdown
    if isinstance(filtered, pd.DataFrame) and "pocket_size" in filtered.columns:
        st.markdown("### Pocket Distribution & Bay Requirements")
        size_order = ["XS", "Small", "Medium", "Large"]
        breakdown = bay_calc.get("breakdown", {})

        if breakdown:
            # Create summary table
            rows = []
            for size in size_order:
                if size in breakdown:
                    info = breakdown[size]
                    rows.append({
                        "Pocket Size": size,
                        "Articles": info["articles"],
                        "Cells/Bay": info["cells_per_bay"],
                        "Bays Needed": info["bays_needed"],
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # ASSQ insight
        if "assq_units" in filtered.columns:
            avg_assq = filtered["assq_units"].mean()
            total_pockets = filtered["assq_units"].count()  # 1 pocket per article (rough)
            st.caption(f"Avg ASSQ: {avg_assq:.1f} units/pocket | {total_pockets:,} pockets needed (1 per article)")

        # === ROI QUICK ESTIMATE ===
        if breakdown:
            total_price = 0
            total_locations_after = 0
            for size, info in breakdown.items():
                bays = info["bays_needed"]
                total_price += bays * config.BAY_PRICES.get(size, 0)
                total_locations_after += bays * config.CELLS_PER_BAY.get(size, 0)

            # Before: standard racking at 12 locations/rack
            # Need same number of articles = need num_articles / 12 racks
            num_articles = eligible_count
            racks_before = max(1, (num_articles + config.IKEA_LOCATIONS_PER_RACK_BEFORE - 1) // config.IKEA_LOCATIONS_PER_RACK_BEFORE)
            locations_before = racks_before * config.IKEA_LOCATIONS_PER_RACK_BEFORE

            locations_saved = total_locations_after - locations_before
            location_multiplier = total_locations_after / locations_before if locations_before > 0 else 0

            st.markdown("### Investment Estimate")
            roi_cols = st.columns(5)
            roi_cols[0].metric("Investment", f"€{total_price:,.0f}")
            roi_cols[1].metric("Bays Before", f"{racks_before}", help=f"Standard racking @ {config.IKEA_LOCATIONS_PER_RACK_BEFORE} loc/rack")
            roi_cols[2].metric("Bays After", f"{total_bays}", help="Storeganizer bays")
            roi_cols[3].metric("Locations", f"{locations_before} → {total_locations_after:,}")
            roi_cols[4].metric("Density Gain", f"{location_multiplier:.1f}x", delta=f"+{locations_saved:,}" if locations_saved > 0 else f"{locations_saved:,}")

            # Store values for ROI step (Step 5)
            st.session_state["roi_investment"] = total_price
            st.session_state["roi_num_racks_before"] = racks_before
            st.session_state["roi_num_racks_after"] = total_bays
            st.session_state["roi_locations_before"] = locations_before
            st.session_state["roi_locations_after"] = total_locations_after

    # Data quality alerts (collapsed)
    if quality.get("alerts"):
        with st.expander("Data quality", expanded=False):
            for alert in quality["alerts"]:
                st.markdown(f"- {alert}")

    # Rejection analysis (collapsed)
    insights = st.session_state.get("smart_rejections", [])
    if rejected_count > 0:
        with st.expander(f"Rejection analysis ({rejected_count} items)", expanded=False):
            if insights:
                for insight in insights:
                    st.markdown(f"**{insight['title']}** — {insight['detail']}")
                    st.caption(f"Recovery: {insight['suggestion']}")
            else:
                st.markdown("No specific rejection patterns detected.")

    # Eligible articles table
    if eligible_count == 0:
        st.error("No eligible articles. Check rejection analysis above.")
    else:
        st.markdown("---")
        st.markdown("### Eligible Articles")

        # Apply column filtering and date formatting
        display_df = filter_display_columns(filtered)
        st.markdown(f"**{len(display_df):,} articles**")
        st.dataframe(display_df, height=400, use_container_width=True)

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

    download_cols = st.columns(3)
    with download_cols[0]:
        download_excel(
            label="📊 Full report",
            df=full_report,
            filename="storeganizer_results.xlsx",
            help="Eligible + rejected SKUs with status",
            key="download_results",
        )
    with download_cols[1]:
        download_excel(
            label="✅ Eligible",
            df=filtered if isinstance(filtered, pd.DataFrame) else pd.DataFrame(),
            filename="storeganizer_eligible.xlsx",
            help="Eligible articles only",
            key="download_eligible",
        )
    with download_cols[2]:
        download_excel(
            label="🚫 Rejected",
            df=rejection_report,
            filename="storeganizer_rejections.xlsx",
            help="Rejected articles with reasons",
            key="download_rejections",
        )



# ===========================
# Step 5: ROI Analysis
# ===========================

def render_step_roi_analysis():
    """ROI Calculator - estimate financial impact of Storeganizer investment."""
    st.subheader("Step 5 — ROI Analysis")
    st.caption("Calculate projected savings, payback period, and return on investment.")

    # Show what was configured in Step 4
    step4_investment = st.session_state.get("roi_investment", 0)
    step4_bays_before = st.session_state.get("roi_num_racks_before", 0)
    step4_bays_after = st.session_state.get("roi_num_racks_after", 0)

    if step4_investment > 0:
        st.success(f"**From Step 4:** €{step4_investment:,.0f} investment | {step4_bays_before} bays before → {step4_bays_after} Storeganizer bays")

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
            value=int(st.session_state.get("roi_num_racks_before", 30)),
            step=1,
            help="Total racks needed in current setup"
        )
        num_racks_after = st.number_input(
            "Number of racks - after (Storeganizer)",
            min_value=1,
            max_value=10000,
            value=int(st.session_state.get("roi_num_racks_after", 15)),
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
