"""
Export utilities for Storeganizer planning outputs.

Generates professional Excel reports for Dimitri's warehouse implementation.
"""
import io
from typing import List
import pandas as pd
from core.allocation import CellBlock
from config import storeganizer as storeganizer_config


def filter_preview_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out excluded columns from preview DataFrames."""
    if df is None or df.empty:
        return df
    cols = [c for c in df.columns if not any(
        excl.lower() in c.lower() for excl in storeganizer_config.EXCLUDED_PREVIEW_COLUMNS
    ) and c not in storeganizer_config.EXCLUDED_SINGLE_COLUMNS]
    return df[cols]


def create_full_article_report(
    original_df: pd.DataFrame,
    eligible_df: pd.DataFrame,
    rejection_reasons: dict,
    planning_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Create full article report showing ALL SKUs (eligible + rejected).

    Columns:
    - SKU Code
    - Description
    - Width (mm)
    - Depth (mm)
    - Height (mm)
    - Weight (kg)
    - Weekly Demand
    - Status (Eligible / Rejected)
    - Rejection Reason
    - Bay
    - Column
    - Row
    - Cell Label
    - Velocity Band

    Args:
        original_df: Original uploaded inventory
        eligible_df: Filtered eligible SKUs
        rejection_reasons: Dict of rejection counts by reason
        planning_df: Planning results (optional)

    Returns:
        DataFrame with full article report
    """
    # Start with original data
    report = original_df.copy()

    # Add status column
    eligible_skus = set(eligible_df['sku_code'].astype(str))
    report['Status'] = report['sku_code'].astype(str).apply(
        lambda x: 'Eligible' if x in eligible_skus else 'Rejected'
    )

    # Add rejection reason - use detailed reasons from eligibility filter if available
    def get_rejection_reason(row):
        if row['Status'] == 'Eligible':
            return ''

        # First check if we already have a detailed rejection reason
        if 'rejection_reason' in row.index and row.get('rejection_reason'):
            return row['rejection_reason']

        sku = str(row['sku_code'])

        # Fall back to heuristics only if no detailed reason provided
        # (only hardcoded filters: dimensions and weight)
        if row.get('width_mm', 0) == 0 or row.get('depth_mm', 0) == 0 or row.get('height_mm', 0) == 0:
            return 'Missing dimensions'

        # Check if oversized (uses largest pocket - Large: 450x500x450)
        if row.get('width_mm', 0) > 450 or row.get('depth_mm', 0) > 500 or row.get('height_mm', 0) > 450:
            return f"Oversized (W:{row.get('width_mm', 0):.0f} D:{row.get('depth_mm', 0):.0f} H:{row.get('height_mm', 0):.0f})"

        # Check weight (hardcoded 20kg limit)
        if row.get('weight_kg', 0) > 20:
            return f"Overweight ({row.get('weight_kg', 0):.1f}kg > 20kg)"

        return 'Filtered by configuration'

    report['Rejection Reason'] = report.apply(get_rejection_reason, axis=1)

    # Add planning data if available
    report['Bay'] = ''
    report['Column'] = ''
    report['Row'] = ''
    report['Cell Label'] = ''
    report['Velocity Band'] = ''

    if planning_df is not None and len(planning_df) > 0:
        # Merge planning data for eligible SKUs
        planning_lookup = planning_df.set_index('sku_code')[['velocity_band']].to_dict()['velocity_band']

        for idx, row in report.iterrows():
            sku = str(row['sku_code'])
            if sku in planning_lookup:
                report.at[idx, 'Velocity Band'] = planning_lookup.get(sku, '')

    # Select and order columns
    output_cols = [
        'sku_code', 'description', 'width_mm', 'depth_mm', 'height_mm',
        'weight_kg', 'weekly_demand', 'Status', 'Rejection Reason',
        'Bay', 'Column', 'Row', 'Cell Label', 'Velocity Band'
    ]

    # Only include columns that exist
    available_cols = [c for c in output_cols if c in report.columns]
    result = report[available_cols].copy()

    # Rename for clarity
    result.columns = [
        'SKU Code', 'Description', 'Width (mm)', 'Depth (mm)', 'Height (mm)',
        'Weight (kg)', 'Weekly Demand', 'Status', 'Rejection Reason',
        'Bay', 'Column', 'Row', 'Cell Label', 'Velocity Band'
    ][:len(available_cols)]

    return result


def create_rejection_report(rejected_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create report of SKUs excluded by eligibility filters.

    Includes only columns relevant to eligibility review.
    """
    if rejected_df is None or len(rejected_df) == 0:
        return pd.DataFrame()

    cols = [
        "sku_code",
        "description",
        "width_mm",
        "depth_mm",
        "height_mm",
        "weight_kg",
        "weekly_demand",
        "stock_weeks",
        "rejection_reason",
    ]

    work = rejected_df.copy()
    for col in cols:
        if col not in work.columns:
            work[col] = ""

    report = work[cols].copy()
    report.columns = [
        "SKU Code",
        "Description",
        "Width (mm)",
        "Depth (mm)",
        "Height (mm)",
        "Weight (kg)",
        "Weekly Demand",
        "Stock Weeks",
        "Rejection Reason",
    ]
    return report


def create_planogram_layout(blocks: List[CellBlock]) -> pd.DataFrame:
    """
    Create planogram layout file for warehouse implementation.

    Format matches old project output:
    - Article Number
    - Article Name
    - Bay
    - Column
    - Row
    - Cell Label (B01-C01-R01)
    - Weight (kg)
    - Velocity Band

    Args:
        blocks: List of CellBlock objects

    Returns:
        DataFrame with planogram layout
    """
    if not blocks:
        return pd.DataFrame()

    records = []
    for block in blocks:
        # Expand multi-row blocks into individual rows
        for row_offset in range(block.row_span):
            row_1based = block.row_start + row_offset + 1
            column_1based = block.column_index + 1

            records.append({
                'Article Number': block.sku_code,
                'Article Name': block.description,
                'Bay': block.bay,
                'Column': column_1based,
                'Row': row_1based,
                'Cell Label': f"B{block.bay:02d}-C{column_1based:02d}-R{row_1based:02d}",
                'Weight (kg)': round(block.sku_weight_kg, 2),
                'Velocity Band': block.velocity_band,
            })

    return pd.DataFrame(records)


def export_to_excel(df: pd.DataFrame, sheet_name: str = "Sheet1") -> bytes:
    """
    Export DataFrame to Excel bytes for download.

    Args:
        df: DataFrame to export
        sheet_name: Name of the Excel sheet

    Returns:
        Excel file as bytes
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    return output.getvalue()


def export_multi_sheet_excel(sheets: dict) -> bytes:
    """
    Export multiple DataFrames to a single Excel file with multiple sheets.

    Args:
        sheets: Dict of {sheet_name: dataframe}

    Returns:
        Excel file as bytes
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    return output.getvalue()


def analyze_file_health(df: pd.DataFrame) -> dict:
    """
    Step 1 relevance check for Storeganizer.

    Categorizes articles into:
    1. IRRELEVANT - Too large/heavy for any pocket (don't waste time on these)
    2. RELEVANT + INCOMPLETE - Could fit but missing data (need attention)
    3. READY - Relevant with complete data (good to go)

    This helps users focus on articles that actually matter.
    """
    if df is None or len(df) == 0:
        return {
            "total": 0,
            "irrelevant": 0,
            "relevant": 0,
            "ready": 0,
            "needs_data": 0,
            "columns_found": {},
            "data_ranges": {},
            "top_ready": pd.DataFrame(),
            "top_needs_data": pd.DataFrame(),
            "top_irrelevant": pd.DataFrame(),
            "missing_breakdown": {},
            "irrelevant_breakdown": {},
            "has_demand": False,
            "demand_col": None,
        }

    total = len(df)

    # === PARSE NUMERIC COLUMNS ===
    width = pd.to_numeric(df.get("width_mm"), errors="coerce") if "width_mm" in df.columns else pd.Series([None] * total, index=df.index)
    depth = pd.to_numeric(df.get("depth_mm"), errors="coerce") if "depth_mm" in df.columns else pd.Series([None] * total, index=df.index)
    height = pd.to_numeric(df.get("height_mm"), errors="coerce") if "height_mm" in df.columns else pd.Series([None] * total, index=df.index)
    weight = pd.to_numeric(df.get("weight_kg"), errors="coerce") if "weight_kg" in df.columns else pd.Series([None] * total, index=df.index)

    # === CHECK FOR DEMAND COLUMN ===
    demand_col_found = None
    has_demand = False
    for demand_col in ["weekly_demand", "ews", "assq_units", "forecast"]:
        if demand_col in df.columns:
            demand_vals = pd.to_numeric(df[demand_col], errors="coerce")
            if demand_vals.notna().any() and (demand_vals > 0).any():
                has_demand = True
                demand_col_found = demand_col
                break

    # === COLUMN DETECTION ===
    columns_found = {
        "sku_code": "sku_code" in df.columns,
        "description": "description" in df.columns,
        "width_mm": "width_mm" in df.columns,
        "depth_mm": "depth_mm" in df.columns,
        "height_mm": "height_mm" in df.columns,
        "weight_kg": "weight_kg" in df.columns,
        "demand": demand_col_found,
    }

    # === STEP 1: IDENTIFY IRRELEVANT ARTICLES ===
    # These will NEVER fit in Storeganizer - don't waste time on them
    # Largest pocket: 500×450×450mm, Max weight: 20kg

    has_dimensions = width.notna() & depth.notna() & height.notna() & (width > 0) & (depth > 0) & (height > 0)
    has_weight = weight.notna() & (weight > 0)

    # Irrelevant = has dimensions AND (too big OR too heavy)
    too_wide = has_dimensions & (width > 500)
    too_deep = has_dimensions & (depth > 450)
    too_tall = has_dimensions & (height > 450)
    too_heavy = has_weight & (weight > 20.0)

    irrelevant_mask = too_wide | too_deep | too_tall | too_heavy

    # === STEP 2: OF REMAINING, CHECK DATA COMPLETENESS ===
    relevant_mask = ~irrelevant_mask

    # Missing data checks (only for relevant articles)
    missing_width = relevant_mask & (width.isna() | (width <= 0))
    missing_depth = relevant_mask & (depth.isna() | (depth <= 0))
    missing_height = relevant_mask & (height.isna() | (height <= 0))
    missing_weight = relevant_mask & (weight.isna() | (weight <= 0))

    missing_any = missing_width | missing_depth | missing_height | missing_weight

    # Ready = relevant + complete dimensions + complete weight
    ready_mask = relevant_mask & ~missing_any

    # Needs data = relevant but missing something
    needs_data_mask = relevant_mask & missing_any

    # === COUNTS ===
    irrelevant_count = int(irrelevant_mask.sum())
    relevant_count = int(relevant_mask.sum())
    ready_count = int(ready_mask.sum())
    needs_data_count = int(needs_data_mask.sum())

    # === BREAKDOWNS ===
    irrelevant_breakdown = {}
    if too_wide.any():
        irrelevant_breakdown["too wide (>500mm)"] = int(too_wide.sum())
    if too_deep.any():
        irrelevant_breakdown["too deep (>450mm)"] = int(too_deep.sum())
    if too_tall.any():
        irrelevant_breakdown["too tall (>450mm)"] = int(too_tall.sum())
    if too_heavy.any():
        irrelevant_breakdown["too heavy (>20kg)"] = int(too_heavy.sum())

    missing_breakdown = {}
    if missing_width.any():
        missing_breakdown["width_mm"] = int(missing_width.sum())
    if missing_depth.any():
        missing_breakdown["depth_mm"] = int(missing_depth.sum())
    if missing_height.any():
        missing_breakdown["height_mm"] = int(missing_height.sum())
    if missing_weight.any():
        missing_breakdown["weight_kg"] = int(missing_weight.sum())

    # Check demand for relevant articles
    if has_demand and demand_col_found:
        demand_vals = pd.to_numeric(df[demand_col_found], errors="coerce")
        missing_demand = relevant_mask & (demand_vals.isna() | (demand_vals <= 0))
        if missing_demand.any():
            missing_breakdown[f"{demand_col_found} (forecast)"] = int(missing_demand.sum())

    # === DATA RANGES (from articles with data) ===
    data_ranges = {}
    if width.notna().any() and (width > 0).any():
        valid_width = width[width > 0]
        data_ranges["width_mm"] = {"min": int(valid_width.min()), "max": int(valid_width.max())}
    if depth.notna().any() and (depth > 0).any():
        valid_depth = depth[depth > 0]
        data_ranges["depth_mm"] = {"min": int(valid_depth.min()), "max": int(valid_depth.max())}
    if height.notna().any() and (height > 0).any():
        valid_height = height[height > 0]
        data_ranges["height_mm"] = {"min": int(valid_height.min()), "max": int(valid_height.max())}
    if weight.notna().any() and (weight > 0).any():
        valid_weight = weight[weight > 0]
        data_ranges["weight_kg"] = {"min": round(float(valid_weight.min()), 2), "max": round(float(valid_weight.max()), 2)}

    # === TOP 15 PREVIEWS (show ALL columns from df) ===
    # Top Ready - all columns, no filtering
    top_ready = pd.DataFrame()
    if ready_mask.any():
        top_ready = df[ready_mask].head(15).copy()

    # Top Needs Data - all columns + Missing indicator
    top_needs_data = pd.DataFrame()
    if needs_data_mask.any():
        needs_df = df[needs_data_mask].head(15).copy()

        def get_missing(row):
            missing = []
            if pd.isna(row.get("width_mm")) or row.get("width_mm", 0) <= 0:
                missing.append("width")
            if pd.isna(row.get("depth_mm")) or row.get("depth_mm", 0) <= 0:
                missing.append("depth")
            if pd.isna(row.get("height_mm")) or row.get("height_mm", 0) <= 0:
                missing.append("height")
            if pd.isna(row.get("weight_kg")) or row.get("weight_kg", 0) <= 0:
                missing.append("weight")
            return ", ".join(missing) if missing else ""

        needs_df["Missing"] = needs_df.apply(get_missing, axis=1)
        top_needs_data = needs_df

    # Top Irrelevant - all columns + Why Irrelevant indicator
    top_irrelevant = pd.DataFrame()
    if irrelevant_mask.any():
        irr_df = df[irrelevant_mask].head(15).copy()

        def get_reason(row):
            reasons = []
            w = pd.to_numeric(row.get("width_mm"), errors="coerce")
            d = pd.to_numeric(row.get("depth_mm"), errors="coerce")
            h = pd.to_numeric(row.get("height_mm"), errors="coerce")
            wt = pd.to_numeric(row.get("weight_kg"), errors="coerce")
            if w and w > 500:
                reasons.append(f"width {w:.0f}mm")
            if d and d > 450:
                reasons.append(f"depth {d:.0f}mm")
            if h and h > 450:
                reasons.append(f"height {h:.0f}mm")
            if wt and wt > 20:
                reasons.append(f"weight {wt:.1f}kg")
            return ", ".join(reasons) if reasons else ""

        irr_df["Why Irrelevant"] = irr_df.apply(get_reason, axis=1)
        top_irrelevant = irr_df

    # Filter out excluded columns from previews
    top_ready = filter_preview_columns(top_ready)
    top_needs_data = filter_preview_columns(top_needs_data)
    top_irrelevant = filter_preview_columns(top_irrelevant)

    return {
        "total": total,
        "irrelevant": irrelevant_count,
        "relevant": relevant_count,
        "ready": ready_count,
        "needs_data": needs_data_count,
        "columns_found": columns_found,
        "data_ranges": data_ranges,
        "top_ready": top_ready,
        "top_needs_data": top_needs_data,
        "top_irrelevant": top_irrelevant,
        "missing_breakdown": missing_breakdown,
        "irrelevant_breakdown": irrelevant_breakdown,
        "has_demand": has_demand,
        "demand_col": demand_col_found,
    }


def create_incomplete_articles_export(df: pd.DataFrame, health_stats: dict = None) -> bytes:
    """
    Create Excel export for RELEVANT articles needing data.

    IMPORTANT: This only exports articles that COULD fit in Storeganizer
    but are missing dimension/weight data. Irrelevant (oversized/overweight)
    articles are NOT included - no point filling in their data.

    Args:
        df: Original DataFrame
        health_stats: Optional pre-computed health stats from analyze_file_health()

    Returns:
        Excel bytes with "Needs Data" and "ReadMe" sheets
    """
    if df is None or len(df) == 0:
        return None

    # Get health stats if not provided
    if health_stats is None:
        health_stats = analyze_file_health(df)

    total = health_stats["total"]
    ready = health_stats["ready"]
    needs_data = health_stats["needs_data"]
    irrelevant = health_stats["irrelevant"]
    missing_breakdown = health_stats.get("missing_breakdown", {})

    # If no articles need data, nothing to export
    if needs_data == 0:
        return None

    # === BUILD MASK FOR RELEVANT + INCOMPLETE ARTICLES ===
    width = pd.to_numeric(df.get("width_mm"), errors="coerce") if "width_mm" in df.columns else pd.Series([None] * len(df), index=df.index)
    depth = pd.to_numeric(df.get("depth_mm"), errors="coerce") if "depth_mm" in df.columns else pd.Series([None] * len(df), index=df.index)
    height = pd.to_numeric(df.get("height_mm"), errors="coerce") if "height_mm" in df.columns else pd.Series([None] * len(df), index=df.index)
    weight = pd.to_numeric(df.get("weight_kg"), errors="coerce") if "weight_kg" in df.columns else pd.Series([None] * len(df), index=df.index)

    has_dimensions = width.notna() & depth.notna() & height.notna() & (width > 0) & (depth > 0) & (height > 0)
    has_weight = weight.notna() & (weight > 0)

    # Irrelevant = oversized OR overweight (with known dimensions/weight)
    too_big = has_dimensions & ((width > 500) | (depth > 450) | (height > 450))
    too_heavy = has_weight & (weight > 20.0)
    irrelevant_mask = too_big | too_heavy

    # Relevant = NOT irrelevant
    relevant_mask = ~irrelevant_mask

    # Missing data (only check for relevant articles)
    missing_width = relevant_mask & (width.isna() | (width <= 0))
    missing_depth = relevant_mask & (depth.isna() | (depth <= 0))
    missing_height = relevant_mask & (height.isna() | (height <= 0))
    missing_weight = relevant_mask & (weight.isna() | (weight <= 0))
    missing_any = missing_width | missing_depth | missing_height | missing_weight

    # Final mask: relevant AND missing data
    needs_data_mask = relevant_mask & missing_any

    if not needs_data_mask.any():
        return None

    # Create export DataFrame
    needs_data_df = df[needs_data_mask].copy()

    def get_missing_fields(row):
        missing = []
        row_width = pd.to_numeric(row.get("width_mm"), errors="coerce") if "width_mm" in row.index else None
        row_depth = pd.to_numeric(row.get("depth_mm"), errors="coerce") if "depth_mm" in row.index else None
        row_height = pd.to_numeric(row.get("height_mm"), errors="coerce") if "height_mm" in row.index else None
        row_weight = pd.to_numeric(row.get("weight_kg"), errors="coerce") if "weight_kg" in row.index else None

        if pd.isna(row_width) or row_width <= 0:
            missing.append("width_mm")
        if pd.isna(row_depth) or row_depth <= 0:
            missing.append("depth_mm")
        if pd.isna(row_height) or row_height <= 0:
            missing.append("height_mm")
        if pd.isna(row_weight) or row_weight <= 0:
            missing.append("weight_kg")

        return ", ".join(missing) if missing else ""

    needs_data_df["Missing Fields"] = needs_data_df.apply(get_missing_fields, axis=1)

    # === BUILD CONTEXTUAL README ===
    readme_lines = [
        ["STOREGANIZER DATA REQUEST", ""],
        ["", ""],
        ["=" * 50, ""],
        ["FILE SUMMARY", ""],
        ["=" * 50, ""],
        [f"Total articles in file:", total],
        [f"Ready for Storeganizer:", ready],
        [f"Need data (in this export):", needs_data],
        [f"Irrelevant (too big/heavy):", irrelevant],
        ["", ""],
        ["=" * 50, ""],
        ["WHY THESE ARTICLES?", ""],
        ["=" * 50, ""],
        ["These articles COULD fit in Storeganizer pockets", ""],
        ["but are missing required dimension/weight data.", ""],
        ["", ""],
        ["Oversized/overweight articles are NOT included.", ""],
        ["They will never fit, so no point filling in data.", ""],
        ["", ""],
    ]

    # Missing data breakdown
    if missing_breakdown:
        readme_lines.append(["MISSING DATA BREAKDOWN:", ""])
        for field, count in missing_breakdown.items():
            if "forecast" not in field.lower():  # Only show dimension/weight fields
                readme_lines.append([f"  {count} articles", f"missing {field}"])
        readme_lines.append(["", ""])

    # Required fields
    readme_lines.extend([
        ["=" * 50, ""],
        ["REQUIRED FIELDS", ""],
        ["=" * 50, ""],
        ["width_mm", "Product width in millimeters"],
        ["depth_mm", "Product depth in millimeters"],
        ["height_mm", "Product height in millimeters"],
        ["weight_kg", "Product weight in kilograms"],
        ["", ""],
        ["POCKET SIZE LIMITS:", ""],
        ["Largest (L):", "500mm W × 450mm D × 450mm H"],
        ["Medium (M):", "300mm W × 450mm D × 300mm H"],
        ["Small (S):", "300mm W × 300mm D × 300mm H"],
        ["Extra Small (XS):", "260mm W × 300mm D × 150mm H"],
        ["Max weight:", "20 kg per pocket"],
        ["", ""],
    ])

    # Instructions
    readme_lines.extend([
        ["=" * 50, ""],
        ["INSTRUCTIONS", ""],
        ["=" * 50, ""],
        ["1. Fill in the missing dimension/weight data", ""],
        ["2. Return the completed file for re-upload", ""],
        ["3. Articles will be assigned to appropriate pockets", ""],
        ["", ""],
    ])

    readme_df = pd.DataFrame(readme_lines, columns=["Field", "Value"])

    # Generate Excel
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        needs_data_df.to_excel(writer, sheet_name="Needs Data", index=False)
        readme_df.to_excel(writer, sheet_name="ReadMe", index=False, header=False)

    return output.getvalue()
