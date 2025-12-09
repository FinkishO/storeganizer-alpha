"""
Export utilities for Storeganizer planning outputs.

Generates professional Excel reports for Dimitri's warehouse implementation.
"""
import io
from typing import List
import pandas as pd
from core.allocation import CellBlock


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

    # Add rejection reason (simplified)
    def get_rejection_reason(row):
        if row['Status'] == 'Eligible':
            return ''

        sku = str(row['sku_code'])

        # Check dimension rejection
        if row.get('width_mm', 0) == 0 or row.get('depth_mm', 0) == 0 or row.get('height_mm', 0) == 0:
            return 'Missing dimensions'

        # Check if oversized (simple heuristic - actual limits from config)
        if row.get('width_mm', 0) > 450 or row.get('depth_mm', 0) > 500 or row.get('height_mm', 0) > 450:
            return 'Oversized'

        # Check weight
        if row.get('weight_kg', 0) > 20:
            return 'Overweight'

        # Check demand
        if row.get('weekly_demand', 0) > 4.0:
            return 'High demand (fast mover)'

        return 'Other'

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
