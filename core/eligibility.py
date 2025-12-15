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
    DEPRECATED: Use apply_stockweeks_filter instead.

    Filter SKUs based on forecast/demand threshold.
    This doesn't account for pocket capacity (ASSQ) - use stockweeks instead.
    """
    df_work = df.copy()

    if "weekly_demand" not in df_work.columns:
        return df_work

    df_work["weekly_demand"] = pd.to_numeric(df_work["weekly_demand"], errors="coerce").fillna(0)
    return df_work[df_work["weekly_demand"] <= max_weekly_demand]


def apply_stockweeks_filter(
    df: pd.DataFrame,
    min_stockweeks: float = None,
    max_stockweeks: float = None,
) -> pd.DataFrame:
    """
    Filter SKUs based on stockweeks (ASSQ / EWS).

    Stockweeks = how many weeks of stock fit in one pocket.
    - If stockweeks < min → item sells too fast (constant replenishment)
    - If stockweeks > max → item is too slow (wasting pocket space)

    Args:
        df: DataFrame with assq_units and weekly_demand columns
        min_stockweeks: Minimum weeks of stock (default from config)
        max_stockweeks: Maximum weeks of stock (default from config)

    Returns:
        Filtered DataFrame
    """
    min_stockweeks = min_stockweeks if min_stockweeks is not None else config.MIN_STOCKWEEKS
    max_stockweeks = max_stockweeks if max_stockweeks is not None else config.MAX_STOCKWEEKS

    df_work = df.copy()

    # Need both columns for stockweeks calculation
    if "weekly_demand" not in df_work.columns:
        return df_work

    # Get ASSQ - use assq_units if available, otherwise default to 1
    if "assq_units" in df_work.columns:
        assq = pd.to_numeric(df_work["assq_units"], errors="coerce").fillna(1)
    else:
        assq = pd.Series(1, index=df_work.index)

    demand = pd.to_numeric(df_work["weekly_demand"], errors="coerce").fillna(0)

    # Calculate stockweeks (avoid division by zero)
    # If demand is 0, stockweeks is infinite (item never sells - keep it for now)
    stockweeks = assq / demand.where(demand > 0, 1)
    stockweeks = stockweeks.where(demand > 0, float('inf'))

    # Store stockweeks in dataframe for reporting
    df_work["stockweeks"] = stockweeks.round(2)

    # Filter: keep items within acceptable stockweeks range
    # Items with 0 demand (infinite stockweeks) pass the min filter but may fail max
    mask = (stockweeks >= min_stockweeks) & (stockweeks <= max_stockweeks)

    return df_work[mask]


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
    max_weekly_demand: float = None,  # DEPRECATED - use stockweeks
    min_stockweeks: float = None,
    max_stockweeks: float = None,
    use_stockweeks_filter: bool = None,
    allow_squeeze: bool = False,
    remove_fragile: bool = False,
) -> Tuple[pd.DataFrame, int, dict]:
    """
    Apply all eligibility filters to SKU DataFrame.

    Hardcoded filters (always applied):
    - Dimensions: Must fit pocket (configurable size)
    - Weight: Must be under pocket weight limit (default 15kg safe, 20kg max)

    Configurable filters:
    - Stockweeks: Min/max weeks of stock coverage
    - Velocity band: A/B/C filtering
    - Fragile: Exclude fragile items by keyword

    Args:
        df: Input DataFrame
        max_width: Maximum pocket width (mm), defaults to config
        max_depth: Maximum pocket depth (mm), defaults to config
        max_height: Maximum pocket height (mm), defaults to config
        max_weight_kg: Maximum weight per pocket (kg), defaults to config
        velocity_band: Velocity filter ("All", "A", "B", "C")
        max_weekly_demand: DEPRECATED - use stockweeks filter instead
        min_stockweeks: Minimum weeks of stock (default from config)
        max_stockweeks: Maximum weeks of stock (default from config)
        use_stockweeks_filter: Use stockweeks instead of EWS filter
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
    use_stockweeks = use_stockweeks_filter if use_stockweeks_filter is not None else config.USE_STOCKWEEKS_FILTER

    original_count = len(df)
    rejection_reasons = {}

    # Track rejections at each stage
    df_filtered = df.copy()

    # === HARDCODED FILTERS (always applied) ===

    # 1. Dimension filter - HARDCODED: must fit pocket
    before = len(df_filtered)
    df_filtered = apply_dimension_filter(df_filtered, max_width, max_depth, max_height, allow_squeeze)
    if len(df_filtered) < before:
        rejection_reasons["dimensions_oversized"] = before - len(df_filtered)

    # 2. Weight filter - HARDCODED: must be under weight limit
    before = len(df_filtered)
    df_filtered = apply_weight_filter(df_filtered, max_weight_kg)
    if len(df_filtered) < before:
        rejection_reasons["weight_overweight"] = before - len(df_filtered)

    # === CONFIGURABLE FILTERS (can be disabled in UI) ===

    # 3. Velocity filter (optional)
    if velocity_band and velocity_band != "All":
        before = len(df_filtered)
        df_filtered = apply_velocity_filter(df_filtered, velocity_band)
        if len(df_filtered) < before:
            rejection_reasons["velocity_band"] = before - len(df_filtered)

    # 4. Stockweeks filter (new) OR legacy forecast filter
    before = len(df_filtered)
    if use_stockweeks:
        df_filtered = apply_stockweeks_filter(df_filtered, min_stockweeks, max_stockweeks)
        if len(df_filtered) < before:
            rejection_reasons["stockweeks_out_of_range"] = before - len(df_filtered)
    elif max_weekly_demand is not None:
        # DEPRECATED: Legacy forecast filter (only if explicitly provided)
        df_filtered = apply_forecast_filter(df_filtered, max_weekly_demand)
        if len(df_filtered) < before:
            rejection_reasons["forecast_threshold"] = before - len(df_filtered)

    # 5. Fragile filter (optional)
    if remove_fragile:
        before = len(df_filtered)
        df_filtered = apply_fragile_filter(df_filtered, remove_fragile)
        if len(df_filtered) < before:
            rejection_reasons["fragile_excluded"] = before - len(df_filtered)

    rejected_count = original_count - len(df_filtered)

    return df_filtered, rejected_count, rejection_reasons


def apply_all_filters_detailed(
    df: pd.DataFrame,
    max_width: float = None,
    max_depth: float = None,
    max_height: float = None,
    max_weight_kg: float = None,
    velocity_band: str = "All",
    max_weekly_demand: float = None,  # DEPRECATED
    min_stockweeks: float = None,
    max_stockweeks: float = None,
    use_stockweeks_filter: bool = None,
    allow_squeeze: bool = False,
    remove_fragile: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, int, dict]:
    """
    Apply eligibility filters and return both eligible and rejected SKUs with reasons.

    Returns individual rejection reasons per SKU for detailed reporting.

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
    use_stockweeks = use_stockweeks_filter if use_stockweeks_filter is not None else config.USE_STOCKWEEKS_FILTER
    min_sw = min_stockweeks if min_stockweeks is not None else config.MIN_STOCKWEEKS
    max_sw = max_stockweeks if max_stockweeks is not None else config.MAX_STOCKWEEKS

    df_work = df.copy()

    # Normalized numeric columns
    width = pd.to_numeric(df_work.get("width_mm"), errors="coerce").fillna(0.0)
    depth = pd.to_numeric(df_work.get("depth_mm"), errors="coerce").fillna(0.0)
    height = pd.to_numeric(df_work.get("height_mm"), errors="coerce").fillna(0.0)
    weight = pd.to_numeric(df_work.get("weight_kg"), errors="coerce").fillna(0.0)
    weekly_demand = pd.to_numeric(df_work.get("weekly_demand"), errors="coerce").fillna(0.0)

    width_multiplier = config.SQUEEZE_WIDTH_MULTIPLIER if allow_squeeze else 1.0
    effective_max_width = max_width * width_multiplier

    # === HARDCODED FILTERS ===

    # 1. Dimension check (must fit pocket)
    dimension_ok = (
        (width > 0)
        & (depth > 0)
        & (height > 0)
        & (width <= effective_max_width)
        & (depth <= max_depth)
        & (height <= max_height)
    )

    # 2. Weight check (must be under limit)
    weight_ok = (weight > 0) & (weight <= max_weight_kg)

    # === CONFIGURABLE FILTERS ===

    # 3. Velocity band filter (optional)
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

    # 4. Stockweeks filter (new) OR legacy forecast filter
    if use_stockweeks:
        # Calculate stockweeks = ASSQ / EWS
        if "assq_units" in df_work.columns:
            assq = pd.to_numeric(df_work["assq_units"], errors="coerce").fillna(1)
        else:
            assq = pd.Series(1, index=df_work.index)

        # Avoid division by zero
        stockweeks = assq / weekly_demand.where(weekly_demand > 0, 1)
        stockweeks = stockweeks.where(weekly_demand > 0, float('inf'))
        df_work["stockweeks"] = stockweeks.round(2)

        stockweeks_ok = (stockweeks >= min_sw) & (stockweeks <= max_sw)
    else:
        stockweeks_ok = pd.Series(True, index=df_work.index)
        stockweeks = pd.Series(0, index=df_work.index)
        # Legacy forecast filter (deprecated)
        if max_weekly_demand is not None and "weekly_demand" in df_work.columns:
            stockweeks_ok = weekly_demand <= max_weekly_demand

    # 5. Fragile filter (optional)
    if not remove_fragile or "description" not in df_work.columns:
        fragile_ok = pd.Series(True, index=df_work.index)
    else:
        pattern = "|".join(config.FRAGILE_KEYWORDS)
        fragile_ok = ~df_work["description"].str.contains(pattern, case=False, na=False)

    # Build reason strings for EACH article
    reason_strings = []
    for idx in df_work.index:
        row_reasons: List[str] = []
        if not dimension_ok.loc[idx]:
            row_reasons.append(f"Oversized (W:{width.loc[idx]:.0f} D:{depth.loc[idx]:.0f} H:{height.loc[idx]:.0f})")
        if not weight_ok.loc[idx]:
            row_reasons.append(f"Overweight ({weight.loc[idx]:.1f}kg > {max_weight_kg}kg)")
        if not velocity_ok.loc[idx] and velocity_band != "All":
            vb = df_work.at[idx, 'velocity_band'] if 'velocity_band' in df_work.columns else '?'
            row_reasons.append(f"Velocity band {vb} excluded")
        if not stockweeks_ok.loc[idx]:
            sw = stockweeks.loc[idx]
            if sw < min_sw:
                row_reasons.append(f"Sells too fast ({sw:.1f} weeks < {min_sw} min)")
            elif sw > max_sw:
                row_reasons.append(f"Too slow ({sw:.1f} weeks > {max_sw} max)")
        if not fragile_ok.loc[idx]:
            row_reasons.append("Fragile item excluded")
        reason_strings.append("; ".join(row_reasons))

    reason_series = pd.Series(reason_strings, index=df_work.index)

    eligible_mask = dimension_ok & weight_ok & velocity_ok & stockweeks_ok & fragile_ok
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
        min_stockweeks=min_sw,
        max_stockweeks=max_sw,
        use_stockweeks_filter=use_stockweeks,
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


def apply_cascading_pocket_allocation(
    df: pd.DataFrame,
    max_weight_kg: float = None,
    velocity_band: str = "All",
    min_stockweeks: float = None,
    max_stockweeks: float = None,
    use_stockweeks_filter: bool = None,
    allow_squeeze: bool = False,
    remove_fragile: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, int, dict]:
    """
    Apply cascading pocket allocation: try XS → S → M → L for each article.
    Assign each article to the SMALLEST pocket that fits its dimensions.

    Pocket sizes (from config.STANDARD_CONFIGS):
    - XS: 300×260×150mm (W×D×H)
    - S:  300×300×300mm
    - M:  450×300×300mm
    - L:  450×500×450mm

    Args:
        df: Input DataFrame with dimensions
        max_weight_kg: Maximum weight per pocket (same for all sizes)
        velocity_band: Velocity filter ("All", "A", "B", "C")
        min_stockweeks: Minimum weeks of stock
        max_stockweeks: Maximum weeks of stock
        use_stockweeks_filter: Use stockweeks filter
        allow_squeeze: Allow 10% width overage for soft packaging
        remove_fragile: Remove fragile items

    Returns:
        Tuple of (eligible_df, rejected_df, rejected_count, rejection_reasons_dict)
    """
    if df is None or len(df) == 0:
        return df, pd.DataFrame(), 0, {}

    # Use config defaults if not specified
    max_weight_kg = max_weight_kg or config.DEFAULT_POCKET_WEIGHT_LIMIT
    use_stockweeks = use_stockweeks_filter if use_stockweeks_filter is not None else config.USE_STOCKWEEKS_FILTER
    min_sw = min_stockweeks if min_stockweeks is not None else config.MIN_STOCKWEEKS
    max_sw = max_stockweeks if max_stockweeks is not None else config.MAX_STOCKWEEKS

    df_work = df.copy()

    # Normalized numeric columns
    width = pd.to_numeric(df_work.get("width_mm"), errors="coerce").fillna(0.0)
    depth = pd.to_numeric(df_work.get("depth_mm"), errors="coerce").fillna(0.0)
    height = pd.to_numeric(df_work.get("height_mm"), errors="coerce").fillna(0.0)
    weight = pd.to_numeric(df_work.get("weight_kg"), errors="coerce").fillna(0.0)
    weekly_demand = pd.to_numeric(df_work.get("weekly_demand"), errors="coerce").fillna(0.0)

    width_multiplier = config.SQUEEZE_WIDTH_MULTIPLIER if allow_squeeze else 1.0

    # Pocket sizes in order (smallest to largest)
    # NOTE: Using official Storeganizer specs from config.STANDARD_CONFIGS
    pocket_sizes = [
        ("XS", config.STANDARD_CONFIGS["xs"]["pocket_width"],
         config.STANDARD_CONFIGS["xs"]["pocket_depth"],
         config.STANDARD_CONFIGS["xs"]["pocket_height"]),
        ("Small", config.STANDARD_CONFIGS["small"]["pocket_width"],
         config.STANDARD_CONFIGS["small"]["pocket_depth"],
         config.STANDARD_CONFIGS["small"]["pocket_height"]),
        ("Medium", config.STANDARD_CONFIGS["medium"]["pocket_width"],
         config.STANDARD_CONFIGS["medium"]["pocket_depth"],
         config.STANDARD_CONFIGS["medium"]["pocket_height"]),
        ("Large", config.STANDARD_CONFIGS["large"]["pocket_width"],
         config.STANDARD_CONFIGS["large"]["pocket_depth"],
         config.STANDARD_CONFIGS["large"]["pocket_height"]),
    ]

    # Assign pocket size to each article
    pocket_assignments = []
    for idx in df_work.index:
        w = width.loc[idx]
        d = depth.loc[idx]
        h = height.loc[idx]

        # Skip invalid dimensions (will be rejected later)
        if w <= 0 or d <= 0 or h <= 0:
            pocket_assignments.append(None)
            continue

        # Try each pocket size in order (smallest to largest)
        assigned = None
        for size_name, max_w, max_d, max_h in pocket_sizes:
            effective_max_w = max_w * width_multiplier
            if w <= effective_max_w and d <= max_d and h <= max_h:
                assigned = size_name
                break

        pocket_assignments.append(assigned)

    df_work["pocket_size"] = pocket_assignments

    # === HARDCODED FILTERS (always applied) ===

    # 1. Dimension check - must fit at least ONE pocket size
    dimension_ok = df_work["pocket_size"].notna()

    # 2. Weight check (must be under limit)
    weight_ok = (weight > 0) & (weight <= max_weight_kg)

    # === CONFIGURABLE FILTERS ===

    # 3. Velocity band filter (optional)
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

    # 4. Stockweeks filter
    if use_stockweeks:
        # Calculate stockweeks = ASSQ / EWS
        if "assq_units" in df_work.columns:
            assq = pd.to_numeric(df_work["assq_units"], errors="coerce").fillna(1)
        else:
            assq = pd.Series(1, index=df_work.index)

        # Avoid division by zero
        stockweeks = assq / weekly_demand.where(weekly_demand > 0, 1)
        stockweeks = stockweeks.where(weekly_demand > 0, float('inf'))
        df_work["stockweeks"] = stockweeks.round(2)

        stockweeks_ok = (stockweeks >= min_sw) & (stockweeks <= max_sw)
    else:
        stockweeks_ok = pd.Series(True, index=df_work.index)
        stockweeks = pd.Series(0, index=df_work.index)

    # 5. Fragile filter (optional)
    if not remove_fragile or "description" not in df_work.columns:
        fragile_ok = pd.Series(True, index=df_work.index)
    else:
        pattern = "|".join(config.FRAGILE_KEYWORDS)
        fragile_ok = ~df_work["description"].str.contains(pattern, case=False, na=False)

    # Build reason strings for EACH article
    reason_strings = []
    for idx in df_work.index:
        row_reasons: List[str] = []
        if not dimension_ok.loc[idx]:
            row_reasons.append(f"Too large for any pocket (W:{width.loc[idx]:.0f} D:{depth.loc[idx]:.0f} H:{height.loc[idx]:.0f})")
        if not weight_ok.loc[idx]:
            row_reasons.append(f"Overweight ({weight.loc[idx]:.1f}kg > {max_weight_kg}kg)")
        if not velocity_ok.loc[idx] and velocity_band != "All":
            vb = df_work.at[idx, 'velocity_band'] if 'velocity_band' in df_work.columns else '?'
            row_reasons.append(f"Velocity band {vb} excluded")
        if not stockweeks_ok.loc[idx]:
            sw = stockweeks.loc[idx]
            if sw < min_sw:
                row_reasons.append(f"Sells too fast ({sw:.1f} weeks < {min_sw} min)")
            elif sw > max_sw:
                row_reasons.append(f"Too slow ({sw:.1f} weeks > {max_sw} max)")
        if not fragile_ok.loc[idx]:
            row_reasons.append("Fragile item excluded")
        reason_strings.append("; ".join(row_reasons))

    reason_series = pd.Series(reason_strings, index=df_work.index)

    eligible_mask = dimension_ok & weight_ok & velocity_ok & stockweeks_ok & fragile_ok
    eligible_df = df_work[eligible_mask].copy()
    rejected_df = df_work[~eligible_mask].copy()
    rejected_df["rejection_reason"] = reason_series[~eligible_mask]

    # Count rejections by reason
    rejection_reasons = {}
    if (~dimension_ok).any():
        rejection_reasons["dimensions_oversized"] = int((~dimension_ok).sum())
    if (~weight_ok).any():
        rejection_reasons["weight_overweight"] = int((~weight_ok).sum())
    if velocity_band != "All" and (~velocity_ok).any():
        rejection_reasons["velocity_band"] = int((~velocity_ok).sum())
    if use_stockweeks and (~stockweeks_ok).any():
        rejection_reasons["stockweeks_out_of_range"] = int((~stockweeks_ok).sum())
    if remove_fragile and (~fragile_ok).any():
        rejection_reasons["fragile_excluded"] = int((~fragile_ok).sum())

    rejected_count = len(rejected_df)
    return eligible_df, rejected_df, rejected_count, rejection_reasons
