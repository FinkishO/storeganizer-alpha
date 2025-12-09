"""
Eligibility filtering for Storeganizer.

Determines which SKUs can fit in Storeganizer pockets based on:
- Dimensions (width, depth, height)
- Weight limits
- Velocity bands (A/B/C)
- Forecast thresholds
- Fragile item exclusions
"""

from typing import List, Tuple
import pandas as pd

from config import storeganizer as config


def apply_dimension_filter(
    df: pd.DataFrame,
    max_width: float,
    max_depth: float,
    max_height: float,
    allow_squeeze: bool = False,
) -> pd.DataFrame:
    """
    Filter SKUs based on dimensional constraints.

    Args:
        df: DataFrame with width_mm, depth_mm, height_mm columns
        max_width: Maximum pocket width (mm)
        max_depth: Maximum pocket depth (mm)
        max_height: Maximum pocket height (mm)
        allow_squeeze: If True, allow 10% width overage for soft packaging

    Returns:
        Filtered DataFrame
    """
    df_work = df.copy()

    # Ensure numeric types
    for col in ["width_mm", "depth_mm", "height_mm"]:
        if col not in df_work.columns:
            df_work[col] = 0
        df_work[col] = pd.to_numeric(df_work[col], errors="coerce").fillna(0)

    # Apply squeeze multiplier if enabled
    width_multiplier = config.SQUEEZE_WIDTH_MULTIPLIER if allow_squeeze else 1.0
    effective_max_width = max_width * width_multiplier

    # Dimensional filters
    # Reject SKUs with zero/missing dimensions (invalid data) OR dimensions exceeding limits
    mask = (
        (df_work["width_mm"] > 0) &  # Must have valid width
        (df_work["depth_mm"] > 0) &  # Must have valid depth
        (df_work["height_mm"] > 0) &  # Must have valid height
        (df_work["width_mm"] <= effective_max_width) &
        (df_work["depth_mm"] <= max_depth) &
        (df_work["height_mm"] <= max_height)
    )

    return df_work[mask]


def apply_weight_filter(
    df: pd.DataFrame,
    max_weight_kg: float,
) -> pd.DataFrame:
    """
    Filter SKUs based on weight constraints.

    Args:
        df: DataFrame with weight_kg column
        max_weight_kg: Maximum weight per pocket (kg)

    Returns:
        Filtered DataFrame
    """
    df_work = df.copy()

    if "weight_kg" not in df_work.columns:
        df_work["weight_kg"] = 0
    df_work["weight_kg"] = pd.to_numeric(df_work["weight_kg"], errors="coerce").fillna(0)

    # Reject SKUs with zero/missing weight (invalid data) OR weight exceeding limit
    return df_work[(df_work["weight_kg"] > 0) & (df_work["weight_kg"] <= max_weight_kg)]


def apply_velocity_filter(
    df: pd.DataFrame,
    velocity_band: str = "All",
) -> pd.DataFrame:
    """
    Filter SKUs based on velocity band (A/B/C).

    Args:
        df: DataFrame with velocity_band column
        velocity_band: "All", "A", "B", or "C"
            - "A": Only A items
            - "B": A and B items
            - "C": A, B, and C items (all)
            - "All": No filtering

    Returns:
        Filtered DataFrame
    """
    if velocity_band == "All" or "velocity_band" not in df.columns:
        return df

    bands_to_include = []
    if velocity_band == "A":
        bands_to_include = ["A"]
    elif velocity_band == "B":
        bands_to_include = ["A", "B"]
    elif velocity_band == "C":
        bands_to_include = ["A", "B", "C"]
    else:
        return df

    return df[df["velocity_band"].isin(bands_to_include)]


def apply_forecast_filter(
    df: pd.DataFrame,
    max_weekly_demand: float,
) -> pd.DataFrame:
    """
    Filter SKUs based on forecast/demand threshold.

    Args:
        df: DataFrame with weekly_demand column
        max_weekly_demand: Maximum weekly demand threshold

    Returns:
        Filtered DataFrame
    """
    df_work = df.copy()

    if "weekly_demand" not in df_work.columns:
        return df_work

    df_work["weekly_demand"] = pd.to_numeric(df_work["weekly_demand"], errors="coerce").fillna(0)
    return df_work[df_work["weekly_demand"] <= max_weekly_demand]


def apply_fragile_filter(
    df: pd.DataFrame,
    remove_fragile: bool = False,
) -> pd.DataFrame:
    """
    Filter out fragile items based on keywords in description.

    Args:
        df: DataFrame with description column
        remove_fragile: If True, remove items matching fragile keywords

    Returns:
        Filtered DataFrame
    """
    if not remove_fragile or "description" not in df.columns:
        return df

    pattern = "|".join(config.FRAGILE_KEYWORDS)
    fragile_mask = ~df["description"].str.contains(pattern, case=False, na=False)

    return df[fragile_mask]


def apply_all_filters(
    df: pd.DataFrame,
    max_width: float = None,
    max_depth: float = None,
    max_height: float = None,
    max_weight_kg: float = None,
    velocity_band: str = "All",
    max_weekly_demand: float = None,
    allow_squeeze: bool = False,
    remove_fragile: bool = False,
) -> Tuple[pd.DataFrame, int, dict]:
    """
    Apply all eligibility filters to SKU DataFrame.

    Uses config defaults if parameters not provided.

    Args:
        df: Input DataFrame
        max_width: Maximum pocket width (mm), defaults to config
        max_depth: Maximum pocket depth (mm), defaults to config
        max_height: Maximum pocket height (mm), defaults to config
        max_weight_kg: Maximum weight per pocket (kg), defaults to config
        velocity_band: Velocity filter ("All", "A", "B", "C")
        max_weekly_demand: Forecast threshold, defaults to config
        allow_squeeze: Allow 10% width overage for soft packaging
        remove_fragile: Remove fragile items

    Returns:
        Tuple of (filtered_df, rejected_count, rejection_reasons_dict)
    """
    if df is None or len(df) == 0:
        return df, 0, {}

    # Use config defaults if not specified
    max_width = max_width or config.DEFAULT_POCKET_WIDTH
    max_depth = max_depth or config.DEFAULT_POCKET_DEPTH
    max_height = max_height or config.DEFAULT_POCKET_HEIGHT
    max_weight_kg = max_weight_kg or config.DEFAULT_POCKET_WEIGHT_LIMIT
    max_weekly_demand = max_weekly_demand or config.DEFAULT_FORECAST_THRESHOLD

    original_count = len(df)
    rejection_reasons = {}

    # Track rejections at each stage
    df_filtered = df.copy()

    # Dimension filter
    before = len(df_filtered)
    df_filtered = apply_dimension_filter(df_filtered, max_width, max_depth, max_height, allow_squeeze)
    if len(df_filtered) < before:
        rejection_reasons["dimensions"] = before - len(df_filtered)

    # Weight filter
    before = len(df_filtered)
    df_filtered = apply_weight_filter(df_filtered, max_weight_kg)
    if len(df_filtered) < before:
        rejection_reasons["weight"] = before - len(df_filtered)

    # Velocity filter
    before = len(df_filtered)
    df_filtered = apply_velocity_filter(df_filtered, velocity_band)
    if len(df_filtered) < before:
        rejection_reasons["velocity_band"] = before - len(df_filtered)

    # Forecast filter
    before = len(df_filtered)
    df_filtered = apply_forecast_filter(df_filtered, max_weekly_demand)
    if len(df_filtered) < before:
        rejection_reasons["forecast_threshold"] = before - len(df_filtered)

    # Fragile filter
    before = len(df_filtered)
    df_filtered = apply_fragile_filter(df_filtered, remove_fragile)
    if len(df_filtered) < before:
        rejection_reasons["fragile_items"] = before - len(df_filtered)

    rejected_count = original_count - len(df_filtered)

    return df_filtered, rejected_count, rejection_reasons


def apply_all_filters_detailed(
    df: pd.DataFrame,
    max_width: float = None,
    max_depth: float = None,
    max_height: float = None,
    max_weight_kg: float = None,
    velocity_band: str = "All",
    max_weekly_demand: float = None,
    allow_squeeze: bool = False,
    remove_fragile: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, int, dict]:
    """
    Apply eligibility filters and return both eligible and rejected SKUs with reasons.

    Args mirror apply_all_filters, but this additionally returns the rejected items
    with a human-readable rejection_reason column.

    Returns:
        eligible_df, rejected_df, rejected_count, rejection_reasons_dict
    """
    if df is None or len(df) == 0:
        return df, pd.DataFrame(), 0, {}

    # Defaults
    max_width = max_width or config.DEFAULT_POCKET_WIDTH
    max_depth = max_depth or config.DEFAULT_POCKET_DEPTH
    max_height = max_height or config.DEFAULT_POCKET_HEIGHT
    max_weight_kg = max_weight_kg or config.DEFAULT_POCKET_WEIGHT_LIMIT
    max_weekly_demand = max_weekly_demand or config.DEFAULT_FORECAST_THRESHOLD

    df_work = df.copy()

    # Normalized numeric columns
    width = pd.to_numeric(df_work.get("width_mm"), errors="coerce").fillna(0.0)
    depth = pd.to_numeric(df_work.get("depth_mm"), errors="coerce").fillna(0.0)
    height = pd.to_numeric(df_work.get("height_mm"), errors="coerce").fillna(0.0)
    weight = pd.to_numeric(df_work.get("weight_kg"), errors="coerce").fillna(0.0)
    weekly_demand = pd.to_numeric(df_work.get("weekly_demand"), errors="coerce").fillna(0.0)

    width_multiplier = config.SQUEEZE_WIDTH_MULTIPLIER if allow_squeeze else 1.0
    effective_max_width = max_width * width_multiplier

    # Individual checks
    dimension_ok = (
        (width > 0)
        & (depth > 0)
        & (height > 0)
        & (width <= effective_max_width)
        & (depth <= max_depth)
        & (height <= max_height)
    )

    weight_ok = (weight > 0) & (weight <= max_weight_kg)

    if velocity_band == "All" or "velocity_band" not in df_work.columns:
        velocity_ok = pd.Series(True, index=df_work.index)
    else:
        if velocity_band == "A":
            allowed = ["A"]
        elif velocity_band == "B":
            allowed = ["A", "B"]
        elif velocity_band == "C":
            allowed = ["A", "B", "C"]
        else:
            allowed = []
        velocity_ok = df_work["velocity_band"].isin(allowed)

    if "weekly_demand" not in df_work.columns:
        forecast_ok = pd.Series(True, index=df_work.index)
    else:
        forecast_ok = weekly_demand <= max_weekly_demand

    if not remove_fragile or "description" not in df_work.columns:
        fragile_ok = pd.Series(True, index=df_work.index)
    else:
        pattern = "|".join(config.FRAGILE_KEYWORDS)
        fragile_ok = ~df_work["description"].str.contains(pattern, case=False, na=False)

    # Build reason strings
    reason_strings = []
    for idx in df_work.index:
        row_reasons: List[str] = []
        if not dimension_ok.loc[idx]:
            row_reasons.append("Dimensions exceed limits or missing")
        if not weight_ok.loc[idx]:
            row_reasons.append("Over weight limit")
        if not velocity_ok.loc[idx]:
            row_reasons.append(f"Velocity band excluded ({df_work.at[idx, 'velocity_band']})")
        if not forecast_ok.loc[idx]:
            row_reasons.append("Above forecast threshold")
        if not fragile_ok.loc[idx]:
            row_reasons.append("Fragile item excluded")
        reason_strings.append("; ".join(row_reasons))

    reason_series = pd.Series(reason_strings, index=df_work.index)

    eligible_mask = dimension_ok & weight_ok & velocity_ok & forecast_ok & fragile_ok
    eligible_df = df_work[eligible_mask].copy()
    rejected_df = df_work[~eligible_mask].copy()
    rejected_df["rejection_reason"] = reason_series[~eligible_mask]

    # Use the existing sequential logic for aggregated counts
    _, _, rejection_reasons = apply_all_filters(
        df,
        max_width=max_width,
        max_depth=max_depth,
        max_height=max_height,
        max_weight_kg=max_weight_kg,
        velocity_band=velocity_band,
        max_weekly_demand=max_weekly_demand,
        allow_squeeze=allow_squeeze,
        remove_fragile=remove_fragile,
    )

    rejected_count = len(rejected_df)
    return eligible_df, rejected_df, rejected_count, rejection_reasons


def get_rejection_summary(rejection_reasons: dict) -> str:
    """
    Generate human-readable summary of rejection reasons.

    Args:
        rejection_reasons: Dict from apply_all_filters

    Returns:
        Formatted string summary
    """
    if not rejection_reasons:
        return "No items rejected"

    lines = ["Rejection breakdown:"]
    for reason, count in rejection_reasons.items():
        lines.append(f"  - {reason.replace('_', ' ').title()}: {count} items")

    return "\n".join(lines)
