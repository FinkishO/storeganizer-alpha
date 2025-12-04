"""
Allocation and mapping logic for Storeganizer.

Handles:
- Planning metrics calculation (units required, columns required, velocity ranking)
- Bay/column/row allocation algorithm
- Weight distribution and overweight flagging
- Family grouping for optimal placement
"""

from dataclasses import dataclass
from math import ceil
from typing import List, Tuple

import numpy as np
import pandas as pd

from config import storeganizer as config


@dataclass
class CellBlock:
    """
    Represents a single SKU placement block in the planogram.

    A block occupies a specific column in a bay, spanning one or more rows.
    """
    sku_code: str
    description: str
    bay: int
    bay_label: str
    column_index: int
    row_start: int
    row_span: int
    units_in_block: int
    velocity_rank: int
    velocity_band: str
    overweight_flag: bool = False
    column_weight_kg: float = 0.0
    sku_weight_kg: float = 0.0  # Individual SKU weight (not cumulative)


def extract_family_identifier(description: str) -> str:
    """
    Extract a family identifier from product description.

    Heuristic: Take first 2-3 words as family grouping.
    Helps keep related products together in layout.

    Args:
        description: Product description string

    Returns:
        Family identifier (e.g., "KITCHEN TOWEL" from "KITCHEN TOWEL BLUE 50CM")
    """
    parts = description.split()
    if len(parts) > 2:
        return " ".join(parts[:2])
    return description


def compute_planning_metrics(
    df: pd.DataFrame,
    units_per_column: int,
    max_weight_per_column_kg: float,
    per_sku_units_col: str | None = None,
) -> pd.DataFrame:
    """
    Add planning-related columns to SKU DataFrame.

    Calculates:
    - units_required = ceil(weekly_demand * stock_weeks)
    - columns_required = ceil(units_required / units_per_column)
    - total_weight_per_column = weight_kg * units_per_column
    - overweight_flag = total_weight_per_column > max_weight_per_column_kg
    - velocity_rank = dense rank of weekly_demand (descending)
    - velocity_band = A/B/C split on configured percentiles
    - family_identifier = common leading words in description
    - units_per_column_plan = per-SKU stacking capacity (from per_sku_units_col or default)

    Args:
        df: SKU DataFrame with required columns
        units_per_column: Target units to stack per column (fallback if per_sku_units_col missing)
        max_weight_per_column_kg: Column weight limit
        per_sku_units_col: Optional column name containing per-SKU capacity

    Returns:
        DataFrame with added planning metric columns
    """
    if units_per_column <= 0:
        raise ValueError("units_per_column must be greater than zero")

    df = df.copy()

    # Ensure numeric types
    weekly = pd.to_numeric(df["weekly_demand"], errors="coerce").fillna(0.0)
    weeks = pd.to_numeric(df["stock_weeks"], errors="coerce").fillna(0.0)
    weight = pd.to_numeric(df["weight_kg"], errors="coerce").fillna(0.0)

    # Determine per-SKU stacking capacity
    units_capacity = pd.Series(units_per_column, index=df.index, dtype=float)
    if per_sku_units_col and per_sku_units_col in df.columns:
        units_capacity = pd.to_numeric(df[per_sku_units_col], errors="coerce").fillna(units_per_column)
    units_capacity = units_capacity.clip(lower=1)
    df["units_per_column_plan"] = units_capacity.astype(int)

    # Calculate requirements
    df["units_required"] = np.ceil(weekly * weeks).astype(int)
    df["columns_required"] = np.ceil(df["units_required"] / units_capacity).astype(int)
    df["total_weight_per_column"] = weight * units_capacity
    df["overweight_flag"] = df["total_weight_per_column"] > float(max_weight_per_column_kg)

    # Velocity ranking
    df["velocity_rank"] = weekly.rank(method="dense", ascending=False).astype(int)
    df["family_identifier"] = df["description"].apply(extract_family_identifier)

    # Velocity bands (A/B/C)
    if len(df) == 0:
        df["velocity_band"] = pd.Series(dtype=str)
        return df

    p_high = np.percentile(weekly, config.VELOCITY_BAND_A_PERCENTILE)
    p_mid = np.percentile(weekly, config.VELOCITY_BAND_B_PERCENTILE)

    def assign_band(demand: float) -> str:
        if demand >= p_high:
            return "A"
        if demand >= p_mid:
            return "B"
        return "C"

    df["velocity_band"] = weekly.apply(assign_band)

    return df


def build_layout(
    df: pd.DataFrame,
    bays: int,
    columns_per_bay: int,
    rows_per_column: int,
    units_per_column: int,
    max_weight_per_column_kg: float,
    per_sku_units_col: str | None = None,
) -> Tuple[pd.DataFrame, List[CellBlock], pd.DataFrame]:
    """
    Allocate SKUs into bay/column/row blocks.

    Algorithm:
    1. Sort SKUs by family identifier, then by demand (groups families together)
    2. For each SKU, allocate units into blocks:
       - Use per-SKU stacking capacity when provided (per_sku_units_col)
       - Try to fit full capacity in one column; otherwise split across columns
       - Fill columns sequentially, respecting row capacity
    3. Track weight per column and flag overweight conditions

    Args:
        df: SKU DataFrame with planning metrics already computed
        bays: Number of bays in layout
        columns_per_bay: Columns per bay
        rows_per_column: Rows per column
        units_per_column: Target units per column (fallback if per_sku_units_col missing)
        max_weight_per_column_kg: Column weight limit for flagging
        per_sku_units_col: Optional column containing per-SKU stacking capacity

    Returns:
        Tuple of:
        - Original DataFrame (unchanged)
        - List of CellBlock objects representing allocation
        - DataFrame summarizing column utilization and weight
    """
    df = df.copy()

    # Validate required columns
    required = ["units_required", "weight_kg", "velocity_rank", "velocity_band", "weekly_demand"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"DataFrame missing required column '{c}' for layout")

    # Initialize column data structures
    columns = []
    for bay in range(1, bays + 1):
        for cidx in range(columns_per_bay):
            columns.append({
                "bay": bay,
                "bay_label": f"Bay {bay}",
                "column_index": cidx,
                "used_rows": 0,
                "remaining_rows": rows_per_column,
                "total_weight_kg": 0.0,
                "blocks": [],
            })

    blocks: List[CellBlock] = []

    # Sort by family first (grouping), then by demand (prioritize high movers)
    sort_cols = ["family_identifier", "weekly_demand"]
    ascending = [True, False]
    sku_rows = df.sort_values(by=sort_cols, ascending=ascending).to_dict(orient="records")

    # Allocate each SKU
    for sku in sku_rows:
        units_remaining = max(0, int(sku.get("units_required", 0)))
        sku_code = str(sku.get("sku_code", ""))
        description = str(sku.get("description", ""))
        weight_kg = float(sku.get("weight_kg", 0.0))
        velocity_rank = int(sku.get("velocity_rank", 0))
        velocity_band = str(sku.get("velocity_band", "C"))
        sku_units_per_column = int(
            sku.get(per_sku_units_col, units_per_column)
            if per_sku_units_col and per_sku_units_col in sku
            else units_per_column
        )
        if sku_units_per_column <= 0:
            sku_units_per_column = units_per_column
        sku_units_per_column = max(1, int(sku_units_per_column))
        units_per_row_for_sku = max(1, sku_units_per_column // rows_per_column) if rows_per_column else sku_units_per_column

        # Allocate units into blocks across columns
        while units_remaining > 0:
            units_in_block = min(units_remaining, sku_units_per_column)

            # Calculate rows needed for this block
            row_span = ceil(units_in_block / units_per_row_for_sku)
            if row_span > rows_per_column:
                row_span = rows_per_column
                units_in_block = min(units_in_block, row_span * units_per_row_for_sku)

            # Find column with enough space
            target_idx = None
            for idx, col in enumerate(columns):
                if col["remaining_rows"] >= row_span:
                    target_idx = idx
                    break

            # If no perfect fit, find any column with space
            if target_idx is None:
                for idx, col in enumerate(columns):
                    if col["remaining_rows"] > 0:
                        row_span = min(row_span, col["remaining_rows"])
                        units_in_block = min(units_in_block, row_span * units_per_row_for_sku)
                        target_idx = idx
                        break

            # Fallback to first column if all full
            if target_idx is None:
                target_idx = 0

            col = columns[target_idx]
            row_start = col["used_rows"]

            # Create block
            block = CellBlock(
                sku_code=sku_code,
                description=description,
                bay=col["bay"],
                bay_label=col["bay_label"],
                column_index=col["column_index"],
                row_start=row_start,
                row_span=row_span,
                units_in_block=int(units_in_block),
                velocity_rank=velocity_rank,
                velocity_band=velocity_band,
                overweight_flag=False,
                column_weight_kg=0.0,
                sku_weight_kg=float(weight_kg),  # Store individual SKU weight
            )

            # Update column state
            col["blocks"].append(block)
            col["used_rows"] += row_span
            col["remaining_rows"] = max(rows_per_column - col["used_rows"], 0)

            # Track weight
            weight_added = float(units_in_block) * weight_kg
            col["total_weight_kg"] += weight_added
            block.column_weight_kg = col["total_weight_kg"]

            blocks.append(block)
            units_remaining -= units_in_block

    # Generate column summaries and flag overweight
    summaries = []
    for col in columns:
        overweight = col["total_weight_kg"] > float(max_weight_per_column_kg)

        # Update all blocks in this column with overweight flag
        for b in col["blocks"]:
            b.overweight_flag = overweight
            b.column_weight_kg = col["total_weight_kg"]

        summaries.append({
            "bay": col["bay"],
            "bay_label": col["bay_label"],
            "column_index": col["column_index"],
            "column_label": f"B{col['bay']}-C{col['column_index']+1}",
            "total_weight_kg": round(col["total_weight_kg"], 3),
            "utilization_rows": f"{col['used_rows']}/{rows_per_column}",
            "utilization_ratio": round(col["used_rows"] / float(rows_per_column), 3) if rows_per_column > 0 else 0,
            "rows_per_column": rows_per_column,
            "overweight_flag": overweight,
        })

    columns_summary_df = pd.DataFrame(summaries)

    return df, blocks, columns_summary_df


def calculate_bay_requirements(
    sku_count: int,
    columns_per_bay: int = None,
    rows_per_column: int = None,
) -> int:
    """
    Calculate estimated bay requirements based on SKU count.

    Args:
        sku_count: Number of SKUs to accommodate
        columns_per_bay: Columns per bay (defaults to config)
        rows_per_column: Rows per column (defaults to config)

    Returns:
        Estimated number of bays needed
    """
    columns_per_bay = columns_per_bay or config.DEFAULT_COLUMNS_PER_BAY
    rows_per_column = rows_per_column or config.DEFAULT_ROWS_PER_COLUMN

    cells_per_bay = columns_per_bay * rows_per_column
    return ceil(sku_count / cells_per_bay) if sku_count > 0 else 1
