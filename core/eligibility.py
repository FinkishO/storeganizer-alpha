"""
Eligibility filtering for Storeganizer.

Determines which SKUs can fit in Storeganizer pockets based on:
- Dimensions (width, depth, height)
- Weight limits
- Velocity bands (A/B/C)
- Forecast thresholds
- Fragile item exclusions
"""

from typing import Tuple
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
