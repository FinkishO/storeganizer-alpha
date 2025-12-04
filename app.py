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

from core import allocation, data_ingest, eligibility
from rag import rag_service
from visualization import planogram_2d
from visualization.viewer_3d import embed_3d_viewer, get_configuration_suggestions
from config import storeganizer as config


APP_TITLE = "Storeganizer Planning Tool"
APP_TAGLINE = "Plan Storeganizer pocket storage layouts from raw inventory to bay-level planogram."
LENA_AVATAR_PATH = Path(__file__).parent / "source" / "lena2_img.png"

# Wizard step labels (Storeganizer-only flow)
WIZARD_STEPS = {
    1: "Welcome",
    2: "Pocket Configuration",
    3: "Upload Inventory",
    4: "Filter & Refine",
    5: "Plan & Optimize",
    6: "Review & Export",
}


# ===========================
# Session State Management
# ===========================

def init_session_state():
    """Initialize session state with Storeganizer defaults."""
    defaults = {
        "wizard_step": 1,
        "plan_name": "Storeganizer Plan",
        "chat_history": [],
        "inventory_raw": None,
        "inventory_filtered": None,
        "inventory_filename": None,
        "column_status": None,
        "rejection_reasons": {},
        "blocks": [],
        "columns_summary": None,
        "planning_df": None,
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
        "forecast_threshold": config.DEFAULT_FORECAST_THRESHOLD,
        "allow_extra_width": config.ALLOW_SQUEEZE_PACKAGING,
        "remove_fragile": config.DEFAULT_REMOVE_FRAGILE,
        "bay_count": 5,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_inventory_state():
    """Clear inventory-dependent session data."""
    st.session_state["inventory_raw"] = None
    st.session_state["inventory_filtered"] = None
    st.session_state["inventory_filename"] = None
    st.session_state["column_status"] = None
    st.session_state["rejection_reasons"] = {}
    st.session_state["blocks"] = []
    st.session_state["columns_summary"] = None
    st.session_state["planning_df"] = None


# ===========================
# Utility helpers
# ===========================

def download_df(label: str, df: pd.DataFrame, filename: str, help: str | None = None):
    """Render a CSV download button for a dataframe."""
    if df is None or df.empty:
        st.button(label, disabled=True, help="Nothing to download yet")
        return

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
        help=help,
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


def go_to_step(step: int):
    """Update wizard step within bounds."""
    step = max(1, min(step, len(WIZARD_STEPS)))
    st.session_state["wizard_step"] = step


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
        "reject_count": sum(st.session_state.get("rejection_reasons", {}).values()),
        "sku_count": len(filtered) if isinstance(filtered, pd.DataFrame) else None,
        "columns_per_bay": st.session_state.get("columns_per_bay"),
        "rows_per_column": st.session_state.get("rows_per_column"),
        "max_weight_per_column": st.session_state.get("max_weight_per_column"),
    }


# ===========================
# Sidebar: Lena Chat
# ===========================

def render_lena_chat():
    """Sidebar with Lena's RAG chat."""
    context = get_chat_context()
    with st.sidebar:
        st.image(str(LENA_AVATAR_PATH), width=120)
        st.markdown("### Lena — Storeganizer Analyst")
        st.caption(config.LENA_PERSONA)

        if st.button("Reset chat", key="reset_chat"):
            st.session_state["chat_history"] = []

        st.markdown("**Suggested prompts**")
        prompt_cols = st.columns(2)
        for idx, prompt in enumerate(config.SUGGESTED_PROMPTS):
            if prompt_cols[idx % 2].button(prompt, key=f"suggest_{idx}"):
                st.session_state["lena_input"] = prompt

        user_message = st.text_input("Ask Lena", key="lena_input", placeholder="e.g., What are the pocket limits?")
        if st.button("Send to Lena", key="send_lena"):
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
# Step 1: Welcome
# ===========================

def render_step_welcome():
    st.subheader("Step 1 — Welcome")
    st.markdown(APP_TAGLINE)
    st.markdown(
        """
        Upload your SKU list, apply eligibility filters, and automatically map items to Storeganizer
        bays, columns, and rows. This flow is Storeganizer-only—no competitor branching, just the
        fastest path from data to planogram.
        """
    )

    cards = st.columns(3)
    cards[0].metric("Default pocket (mm)", f"{config.DEFAULT_POCKET_WIDTH}W × {config.DEFAULT_POCKET_DEPTH}D × {config.DEFAULT_POCKET_HEIGHT}H")
    cards[1].metric("Pocket weight limit", f"{config.DEFAULT_POCKET_WEIGHT_LIMIT} kg")
    cards[2].metric("Columns × Rows per bay", f"{config.DEFAULT_COLUMNS_PER_BAY} × {config.DEFAULT_ROWS_PER_COLUMN}")

    st.markdown("---")
    st.markdown(
        """
        **Flow overview**
        1) Configure Storeganizer pockets and bay structure  
        2) Upload inventory (CSV or Excel)  
        3) Filter by fit, weight, velocity, and demand  
        4) Generate plan metrics and bay allocation  
        5) Visualize and export the planogram
        """
    )
    if st.button("Start planning", type="primary"):
        go_to_step(2)


# ===========================
# Step 2: Pocket Configuration
# ===========================

def render_step_configuration():
    st.subheader("Step 2 — Choose Your Configuration")
    st.caption("Select a standard Storeganizer configuration or customize your own.")

    def apply_config(config_key: str):
        cfg = config.STANDARD_CONFIGS.get(config_key)
        if not cfg:
            return
        st.session_state["selected_config_size"] = config_key
        st.session_state["pocket_width"] = cfg["pocket_width"]
        st.session_state["pocket_depth"] = cfg["pocket_depth"]
        st.session_state["pocket_height"] = cfg["pocket_height"]
        st.session_state["pocket_weight_limit"] = cfg["pocket_weight_limit"]
        st.session_state["columns_per_bay"] = cfg["columns_per_bay"]
        st.session_state["rows_per_column"] = cfg["rows_per_column"]
        st.session_state["show_custom_config"] = False
        st.rerun()

    # ===== SECTION 1: Configuration Cards =====
    st.markdown("### Standard Configurations")
    cols = st.columns(4)
    for idx, (config_key, cfg) in enumerate(config.STANDARD_CONFIGS.items()):
        with cols[idx]:
            is_selected = st.session_state.get("selected_config_size", config.DEFAULT_CONFIG_SIZE) == config_key
            if st.button(
                cfg["name"],
                key=f"config_{config_key}",
                use_container_width=True,
                type="primary" if is_selected else "secondary",
            ):
                apply_config(config_key)
            desc = f"{cfg['description']} ({cfg['pocket_width']}×{cfg['pocket_depth']}×{cfg['pocket_height']} mm)"
            st.caption(desc)
            st.image(cfg["image"], use_container_width=True)
            st.caption(f"Cells/bay: {cfg['cells_per_bay']}")

    st.markdown("---")

    # ===== SECTION 2: Bay Count =====
    st.markdown("### How Many Bays?")
    col1, _ = st.columns([2, 1])
    with col1:
        num_bays = st.number_input(
            "Number of bays",
            min_value=1,
            max_value=50,
            value=int(st.session_state.get("num_bays", 5)),
            key="num_bays",
            help="Total number of Storeganizer bays in your warehouse",
        )
    st.session_state["bay_count"] = int(num_bays)

    # ===== SECTION 3: Capacity Summary =====
    st.markdown("### Your Configuration Summary")
    selected_key = st.session_state.get("selected_config_size", config.DEFAULT_CONFIG_SIZE)
    selected_cfg = config.STANDARD_CONFIGS.get(selected_key)
    cells_per_bay = (
        selected_cfg["cells_per_bay"]
        if selected_cfg
        else int(st.session_state.get("columns_per_bay", 0)) * int(st.session_state.get("rows_per_column", 0))
    )
    total_cells = int(num_bays) * int(cells_per_bay or 0)
    estimated_skus = int(total_cells * 0.8)  # Assume 80% utilization

    metric_cols = st.columns(3)
    metric_cols[0].metric("Total Cells", format_metric(total_cells))
    metric_cols[1].metric("Estimated SKUs", format_metric(estimated_skus), help="Assuming 80% utilization")
    price_per_bay = selected_cfg.get("price_per_bay_eur") if selected_cfg else None
    if price_per_bay:
        metric_cols[2].metric("Estimated Price", f"€{int(price_per_bay * int(num_bays)):,}")
    else:
        metric_cols[2].metric("Estimated Price", "TBD", help="Contact Storeganizer for pricing")

    selection_name = selected_cfg["name"] if selected_cfg else "Custom"
    st.info(f"💡 Configuration: **{selection_name}** × {int(num_bays)} bays")

    st.markdown("---")

    # ===== SECTION 4: Advanced / Custom =====
    prior_state = {
        "pocket_width": st.session_state.get("pocket_width"),
        "pocket_depth": st.session_state.get("pocket_depth"),
        "pocket_height": st.session_state.get("pocket_height"),
        "pocket_weight_limit": st.session_state.get("pocket_weight_limit"),
        "columns_per_bay": st.session_state.get("columns_per_bay"),
        "rows_per_column": st.session_state.get("rows_per_column"),
        "max_weight_per_column": st.session_state.get("max_weight_per_column"),
    }
    with st.expander("⚙️ Advanced: Custom Configuration", expanded=False):
        st.caption("Override standard configurations with custom pocket dimensions and structure.")
        adv_cols = st.columns(3)
        with adv_cols[0]:
            st.number_input(
                "Pocket width (mm)",
                min_value=100.0,
                max_value=2000.0,
                value=float(st.session_state["pocket_width"]),
                step=5.0,
                key="pocket_width",
            )
            st.number_input(
                "Pocket depth (mm)",
                min_value=100.0,
                max_value=2000.0,
                value=float(st.session_state["pocket_depth"]),
                step=5.0,
                key="pocket_depth",
            )
            st.number_input(
                "Pocket height (mm)",
                min_value=100.0,
                max_value=2000.0,
                value=float(st.session_state["pocket_height"]),
                step=5.0,
                key="pocket_height",
            )
        with adv_cols[1]:
            st.number_input(
                "Pocket weight limit (kg)",
                min_value=1.0,
                max_value=200.0,
                value=float(st.session_state["pocket_weight_limit"]),
                step=0.5,
                key="pocket_weight_limit",
            )
            st.number_input(
                "Columns per bay",
                min_value=1,
                max_value=30,
                value=int(st.session_state["columns_per_bay"]),
                key="columns_per_bay",
            )
            st.number_input(
                "Rows per column",
                min_value=1,
                max_value=20,
                value=int(st.session_state["rows_per_column"]),
                key="rows_per_column",
            )
        with adv_cols[2]:
            st.number_input(
                "Max weight per column (kg)",
                min_value=5.0,
                max_value=1000.0,
                value=float(st.session_state["max_weight_per_column"]),
                step=5.0,
                key="max_weight_per_column",
                help="Flag columns exceeding this threshold.",
            )
        new_state = {
            "pocket_width": st.session_state.get("pocket_width"),
            "pocket_depth": st.session_state.get("pocket_depth"),
            "pocket_height": st.session_state.get("pocket_height"),
            "pocket_weight_limit": st.session_state.get("pocket_weight_limit"),
            "columns_per_bay": st.session_state.get("columns_per_bay"),
            "rows_per_column": st.session_state.get("rows_per_column"),
            "max_weight_per_column": st.session_state.get("max_weight_per_column"),
        }
        if new_state != prior_state:
            st.session_state["selected_config_size"] = "custom"
            st.session_state["show_custom_config"] = True

def render_step_upload():
    st.subheader("Step 3 — Upload Inventory")
    st.caption("CSV or Excel accepted. Column aliases are handled automatically.")

    uploaded_file = st.file_uploader("Upload inventory file", type=["csv", "xlsx", "xls"])
    if uploaded_file:
        try:
            df = data_ingest.load_inventory_file(uploaded_file)
            df = add_assq_columns(df)
            st.session_state["inventory_raw"] = df
            st.session_state["inventory_filename"] = uploaded_file.name
            st.session_state["column_status"] = data_ingest.get_column_status(df)
            st.success(f"Loaded {len(df)} rows from {uploaded_file.name}")
        except ValueError as exc:
            st.error(f"File error: {exc}")
        except Exception as exc:  # pragma: no cover - defensive log
            st.error(f"Unexpected error reading file: {exc}")

    if st.session_state.get("inventory_raw") is not None:
        status = st.session_state.get("column_status") or {}
        col_a, col_b = st.columns(2)
        col_a.markdown("**Required columns present**")
        col_a.write(status.get("required_present", []))
        col_b.markdown("**Missing required columns**")
        col_b.write(status.get("required_missing", []))
        st.markdown("**Optional columns**")
        st.write(status.get("optional_present", []))

        st.markdown("**Preview**")
        st.dataframe(st.session_state["inventory_raw"].head(10))
        if "assq_units" in st.session_state["inventory_raw"].columns:
            review_count = int(st.session_state["inventory_raw"].get("assq_needs_review", pd.Series(dtype=bool)).sum())
            median_assq = pd.to_numeric(st.session_state["inventory_raw"]["assq_units"], errors="coerce").median()
            median_txt = int(median_assq) if pd.notna(median_assq) else "?"
            st.caption(f"ASSQ (auto stacking): median {median_txt} units; {review_count} flagged for review.")

    cols = st.columns([1, 1, 1])
    if cols[0].button("Reset upload"):
        reset_inventory_state()
    if cols[2].button("Next: Filter & refine", type="primary", disabled=st.session_state.get("inventory_raw") is None):
        go_to_step(4)


# ===========================
# Step 4: Filter & Refine
# ===========================

def render_step_filter():
    st.subheader("Step 4 — Filter & Refine")
    raw_df = st.session_state.get("inventory_raw")
    if raw_df is None:
        st.warning("Upload inventory first.")
        return

    st.caption("Tidy this up: clear planning mode, simple demand filters, tuck the heavy stuff into an expander.")

    st.markdown("### Planning mode")
    st.radio(
        "Pocket allocation",
        options=["Single pocket per SKU (recommended)", "Multi-pocket stacking (coming soon)"],
        index=0,
        disabled=False,
        help="Single pocket: one SKU per pocket. Multi-pocket stacking available in future release.",
    )
    # Always use single pocket mode
    st.session_state["single_pocket_per_sku"] = True

    st.markdown("### Demand & velocity")
    with st.form("eligibility_form"):
        cols = st.columns([1, 1])
        with cols[0]:
            st.selectbox(
                "Velocity band",
                options=["All", "A", "B", "C"],
                index=["All", "A", "B", "C"].index(st.session_state.get("velocity_band_filter", "All")),
                key="velocity_band_filter",
            )
            st.number_input(
                "Forecast threshold (weekly demand)",
                min_value=0.0,
                max_value=10000.0,
                value=float(st.session_state.get("forecast_threshold", config.DEFAULT_FORECAST_THRESHOLD)),
                step=0.5,
                key="forecast_threshold",
            )
        with cols[1]:
            st.checkbox(
                "Allow soft packaging squeeze (+10% width)",
                value=bool(st.session_state.get("allow_extra_width", False)),
                key="allow_extra_width",
            )
            st.checkbox(
                "Remove fragile items",
                value=bool(st.session_state.get("remove_fragile", False)),
                key="remove_fragile",
            )

        with st.expander("Advanced: size & weight filters", expanded=False):
            adv_cols = st.columns(3)
            with adv_cols[0]:
                st.number_input(
                    "Max width (mm)",
                    min_value=50.0,
                    max_value=5000.0,
                    value=float(st.session_state.get("pocket_width", config.DEFAULT_POCKET_WIDTH)),
                    key="elig_max_w",
                )
                st.number_input(
                    "Max depth (mm)",
                    min_value=50.0,
                    max_value=5000.0,
                    value=float(st.session_state.get("pocket_depth", config.DEFAULT_POCKET_DEPTH)),
                    step=5.0,
                    key="elig_max_d",
                )
            with adv_cols[1]:
                st.number_input(
                    "Max height (mm)",
                    min_value=50.0,
                    max_value=5000.0,
                    value=float(st.session_state.get("pocket_height", config.DEFAULT_POCKET_HEIGHT)),
                    step=5.0,
                    key="elig_max_h",
                )
                st.number_input(
                    "Pocket weight limit (kg)",
                    min_value=1.0,
                    max_value=200.0,
                    value=float(st.session_state.get("pocket_weight_limit", config.DEFAULT_POCKET_WEIGHT_LIMIT)),
                    step=0.5,
                    key="pocket_weight_limit",
                )
            with adv_cols[2]:
                st.number_input(
                    "Max weight per column (kg)",
                    min_value=5.0,
                    max_value=1000.0,
                    value=float(st.session_state.get("max_weight_per_column", config.DEFAULT_COLUMN_WEIGHT_LIMIT)),
                    step=5.0,
                    key="max_weight_per_column",
                )

        submitted = st.form_submit_button("Apply filters")

    st.caption(
        f"Using velocity: {st.session_state.get('velocity_band_filter', 'All')}, "
        f"forecast ≥ {st.session_state.get('forecast_threshold', config.DEFAULT_FORECAST_THRESHOLD)}, "
        f"size W/D/H ≤ {st.session_state.get('elig_max_w', config.DEFAULT_POCKET_WIDTH)}/"
        f"{st.session_state.get('elig_max_d', config.DEFAULT_POCKET_DEPTH)}/"
        f"{st.session_state.get('elig_max_h', config.DEFAULT_POCKET_HEIGHT)} mm, "
        f"squeeze={'on' if st.session_state.get('allow_extra_width') else 'off'}, "
        f"fragile={'excluded' if st.session_state.get('remove_fragile') else 'included'}."
    )

    if submitted:
        apply_filters()

    # Quick peek at potential fragile items before exclusion
    if raw_df is not None and "description" in raw_df.columns:
        fragile_pattern = "|".join(config.FRAGILE_KEYWORDS)
        potential_fragile = raw_df[raw_df["description"].str.contains(fragile_pattern, case=False, na=False)]
        if len(potential_fragile) > 0:
            st.info(f"Potential fragile SKUs: {len(potential_fragile)} flagged by keywords ({', '.join(config.FRAGILE_KEYWORDS)}).")
            st.dataframe(potential_fragile[["sku_code", "description"]].head(10), use_container_width=True)

    filtered = st.session_state.get("inventory_filtered")
    if filtered is not None:
        dropped = sum(st.session_state.get("rejection_reasons", {}).values())
        st.success(f"Eligible items: {len(filtered)} rows. Rejected: {dropped}.")
        st.text(eligibility.get_rejection_summary(st.session_state.get("rejection_reasons", {})))
        if "assq_needs_review" in filtered.columns:
            review_count = int(filtered["assq_needs_review"].sum())
            if review_count > 0:
                st.warning(f"{review_count} SKUs fall back to single-unit stacking. Please review later.")
        st.dataframe(filtered.head(20), use_container_width=True)

    if st.button("Next: Plan & optimize", type="primary", disabled=filtered is None or len(filtered) == 0):
        go_to_step(5)


def apply_filters():
    """Run eligibility filters and persist filtered dataframe."""
    raw_df = st.session_state.get("inventory_raw")
    if raw_df is None:
        return

    # Compute velocity bands using defaults to enable velocity filtering
    df_with_assq = add_assq_columns(raw_df)
    st.session_state["inventory_raw"] = df_with_assq

    df_with_velocity = allocation.compute_planning_metrics(
        df_with_assq,
        units_per_column=config.DEFAULT_UNITS_PER_COLUMN,
        max_weight_per_column_kg=st.session_state.get("max_weight_per_column", config.DEFAULT_COLUMN_WEIGHT_LIMIT),
        per_sku_units_col="assq_units",
    )

    filtered_df, rejected_count, rejection_reasons = eligibility.apply_all_filters(
        df_with_velocity,
        max_width=st.session_state.get("elig_max_w", config.DEFAULT_POCKET_WIDTH),
        max_depth=st.session_state.get("elig_max_d", config.DEFAULT_POCKET_DEPTH),
        max_height=st.session_state.get("elig_max_h", config.DEFAULT_POCKET_HEIGHT),
        max_weight_kg=st.session_state.get("pocket_weight_limit", config.DEFAULT_POCKET_WEIGHT_LIMIT),
        velocity_band=st.session_state.get("velocity_band_filter", "All"),
        max_weekly_demand=st.session_state.get("forecast_threshold", config.DEFAULT_FORECAST_THRESHOLD),
        allow_squeeze=st.session_state.get("allow_extra_width", False),
        remove_fragile=st.session_state.get("remove_fragile", False),
    )

    st.session_state["inventory_filtered"] = filtered_df
    st.session_state["rejection_reasons"] = rejection_reasons

    # Auto-set bay count suggestion based on filtered SKU count
    suggested_bays = allocation.calculate_bay_requirements(
        sku_count=len(filtered_df),
        columns_per_bay=st.session_state.get("columns_per_bay"),
        rows_per_column=st.session_state.get("rows_per_column"),
    )
    st.session_state["bay_count"] = suggested_bays
    st.session_state["num_bays"] = suggested_bays


# ===========================
# Step 5: Plan & Optimize
# ===========================

def render_step_plan():
    st.subheader("Step 5 — Plan & Optimize")
    df = st.session_state.get("inventory_filtered")
    if df is None or len(df) == 0:
        st.warning("Apply filters first.")
        return

    cols = st.columns(3)
    with cols[0]:
        st.number_input(
            "Max weight per column (kg)",
            min_value=5.0,
            max_value=1000.0,
            value=float(st.session_state.get("max_weight_per_column", config.DEFAULT_COLUMN_WEIGHT_LIMIT)),
            step=5.0,
            key="max_weight_per_column",
        )
    with cols[1]:
        st.number_input(
            "Columns per bay",
            min_value=1,
            max_value=30,
            value=int(st.session_state.get("columns_per_bay", config.DEFAULT_COLUMNS_PER_BAY)),
            key="columns_per_bay",
        )
    with cols[2]:
        st.number_input(
            "Rows per column",
            min_value=1,
            max_value=20,
            value=int(st.session_state.get("rows_per_column", config.DEFAULT_ROWS_PER_COLUMN)),
            key="rows_per_column",
        )

    st.number_input(
        "Bays to allocate",
        min_value=1,
        max_value=300,
        value=int(st.session_state.get("bay_count", 4)),
        key="bay_count",
        help="You can increase bay count if columns are overweight.",
    )

    if st.button("Run planning", type="primary"):
        try:
            # Check if single pocket per SKU mode is enabled
            single_pocket_mode = st.session_state.get("single_pocket_per_sku", True)

            if single_pocket_mode:
                # Simple mode: 1 pocket per SKU (no ASSQ calculation)
                df_with_assq = df.copy()
                df_with_assq["assq_units"] = 1
                units_per_column = 1

                # Compute planning metrics first
                planning_df = allocation.compute_planning_metrics(
                    df_with_assq,
                    units_per_column=units_per_column,
                    max_weight_per_column_kg=st.session_state["max_weight_per_column"],
                    per_sku_units_col="assq_units",
                )

                # Override: exactly 1 unit per SKU (1 pocket per SKU)
                planning_df["units_required"] = 1
                planning_df["columns_required"] = 1
            else:
                # Multi-pocket mode: calculate ASSQ for stacking
                df_with_assq = add_assq_columns(df)
                units_per_column = config.DEFAULT_UNITS_PER_COLUMN

                planning_df = allocation.compute_planning_metrics(
                    df_with_assq,
                    units_per_column=units_per_column,
                    max_weight_per_column_kg=st.session_state["max_weight_per_column"],
                    per_sku_units_col="assq_units",
                )
            # Build layout with calculated planning data
            planning_df, blocks, columns_summary = allocation.build_layout(
                planning_df,
                bays=int(st.session_state["bay_count"]),
                columns_per_bay=int(st.session_state["columns_per_bay"]),
                rows_per_column=int(st.session_state["rows_per_column"]),
                units_per_column=units_per_column,
                max_weight_per_column_kg=float(st.session_state["max_weight_per_column"]),
                per_sku_units_col="assq_units",
            )
        except ValueError as exc:
            st.error(f"Planning error: {exc}")
            return

        st.session_state["planning_df"] = planning_df
        st.session_state["blocks"] = blocks
        st.session_state["columns_summary"] = columns_summary

    planning_df = st.session_state.get("planning_df")
    blocks = st.session_state.get("blocks", [])
    cols_summary = st.session_state.get("columns_summary")

    if planning_df is not None:
        overweight_cols = 0
        if cols_summary is not None and "overweight_flag" in cols_summary.columns:
            overweight_cols = int(cols_summary["overweight_flag"].sum())
        metrics_cols = st.columns(4)
        metrics_cols[0].metric("SKUs planned", format_metric(len(planning_df)))
        metrics_cols[1].metric("Bays", format_metric(st.session_state.get("bay_count", 0)))
        metrics_cols[2].metric("Columns per bay", format_metric(st.session_state.get("columns_per_bay")))
        metrics_cols[3].metric("Overweight columns", format_metric(overweight_cols))
        if "assq_needs_review" in planning_df.columns:
            review_count = int(planning_df["assq_needs_review"].sum())
            st.caption(f"Auto stacking (ASSQ) applied. {review_count} SKUs flagged for manual review.")

        st.markdown("**Planning preview (top 15 rows)**")
        st.dataframe(planning_df.head(15), use_container_width=True)

    if st.button("Next: Review & export", type="primary", disabled=st.session_state.get("planning_df") is None):
        go_to_step(6)


# ===========================
# Step 6: Review & Export
# ===========================

def render_step_review():
    st.subheader("Step 6 — Review & Export")
    planning_df = st.session_state.get("planning_df")
    blocks = st.session_state.get("blocks", [])
    columns_summary = st.session_state.get("columns_summary")

    if planning_df is None or not blocks:
        st.warning("Run planning first.")
        return

    overweight_cols = int(columns_summary["overweight_flag"].sum()) if columns_summary is not None else 0
    st.markdown(
        f"**Plan overview:** {len(planning_df)} SKUs across {st.session_state.get('bay_count')} bays, "
        f"{st.session_state.get('columns_per_bay')} columns/bay. Overweight columns: {overweight_cols}."
    )

    color_mode = st.selectbox("Color planogram by", ["By SKU", "By velocity"], index=0)
    planogram_2d.render_planogram(
        blocks=blocks,
        columns_summary=columns_summary,
        color_mode="Velocity band" if "velocity" in color_mode.lower() else "By SKU",
    )

    st.markdown("---")
    st.markdown("**Exports**")
    export_cols = st.columns(3)
    with export_cols[0]:
        download_df("Download planning table", planning_df, "storeganizer_planning.csv")
    with export_cols[1]:
        download_df("Download columns summary", columns_summary, "storeganizer_columns.csv")
    with export_cols[2]:
        blocks_df = blocks_to_dataframe(blocks)
        download_df("Download planogram blocks", blocks_df, "storeganizer_blocks.csv")

    st.markdown("---")
    st.markdown("**3D visualization (placeholder)**")
    suggestion = get_configuration_suggestions(
        bay_count=int(st.session_state.get("bay_count", 0)),
        sku_count=len(planning_df),
    )
    st.info(f"Suggested configuration: {suggestion['size_category']} ({suggestion['reason']})")
    if st.checkbox("Preview Storeganizer 3D viewer (coming soon)"):
        st.components.v1.html(embed_3d_viewer({}, height="650px"), height=680)


# ===========================
# Main
# ===========================

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="📦")
    init_session_state()

    st.title(APP_TITLE)
    render_stepper()

    render_lena_chat()

    step = st.session_state["wizard_step"]
    with st.container():
        if step == 1:
            render_step_welcome()
        elif step == 2:
            render_step_configuration()
        elif step == 3:
            render_step_upload()
        elif step == 4:
            render_step_filter()
        elif step == 5:
            render_step_plan()
        elif step == 6:
            render_step_review()
        else:
            render_step_welcome()

    st.markdown("---")
    nav_cols = st.columns(2)
    with nav_cols[0]:
        st.button("Previous", disabled=step == 1, on_click=lambda: go_to_step(step - 1))
    with nav_cols[1]:
        st.button("Next", disabled=step == len(WIZARD_STEPS), on_click=lambda: go_to_step(step + 1))


if __name__ == "__main__":
    main()
