"""
SKILL: ASSQ Calculator (Assigned Sales Space Quantity)

PURPOSE:
Calculate how many units/boxes fit in a single pocket based on:
- Article dimensions (CP or MP)
- Pocket dimensions (XS/S/M/L)
- Weight limits
- Rotation options
- Air buffer (25% rule)

TERMINOLOGY:
- Speedcell = Cell = Pocket (synonyms throughout codebase)
- CP = Consumer Pack (single unit packaging)
- MP = Multipack (multiple units in one box)
- ASSQ = units that fit in one pocket
- EWS = Expected Weekly Sales

INPUTS:
- article_length, article_width, article_height (mm)
- article_weight (kg per unit/box)
- pocket_length, pocket_width, pocket_height (mm)
- pocket_weight_limit (kg, default 20)
- allow_rotation (bool) - can swap length ↔ width
- allow_flip (bool) - can swap height with length/width
- air_buffer_pct (float) - default 0.25 (25% air = 75% fill)
- min_height_mm (float) - minimum 15mm to avoid extreme stacking

OUTPUTS:
- assq: int (units that fit)
- rotation_used: bool
- flip_used: bool
- fill_efficiency: float (0-1)
- weight_limited: bool (True if weight was the constraint)
- fit_details: dict with rows_deep, cols_wide, layers_high

ALGORITHM (from CPH official logic):

1. TRY ORIGINAL ORIENTATION
   - rows_deep = floor(pocket_depth / article_length)
   - cols_wide = floor(pocket_width / article_width)
   - layers_high = floor(pocket_height / max(article_height, min_height_mm))
   - raw_fit = rows_deep × cols_wide × layers_high

2. IF ROTATION ALLOWED, TRY ROTATED
   - Swap article_length ↔ article_width
   - Recalculate rows_deep, cols_wide
   - Keep better result

3. IF FLIP ALLOWED, TRY FLIPPED (height swapped)
   - Try height as length, original height as width, etc.
   - Only if improves fit

4. APPLY WEIGHT LIMIT
   - max_by_weight = floor(pocket_weight_limit / article_weight)
   - assq = min(raw_fit, max_by_weight)

5. APPLY AIR BUFFER
   - assq = floor(assq × (1 - air_buffer_pct))
   - Ensures 25% headroom for easy picking

6. RETURN RESULT
   - assq (capped at 1 minimum if fits at all)
   - Metadata about how it fit

ERROR HANDLING:
- Zero/negative dimensions → return assq=0, error logged
- Weight > pocket limit → return assq=0
- Article doesn't fit any orientation → return assq=0

SELF-ANNEALING:
- Log cases where calculated ASSQ differs from Emmie's "CP Speedcell Qty"
- Track rotation/flip decisions for learning
"""

from typing import Tuple, Dict, Optional
from dataclasses import dataclass
import math

# Self-annealing error log
ERROR_LOG = []


@dataclass
class ASSQResult:
    """Result of ASSQ calculation."""
    assq: int
    fits: bool
    rotation_used: bool
    flip_used: bool
    weight_limited: bool
    fill_efficiency: float
    fit_details: Dict
    error: Optional[str] = None


def log_error(article_id: str, expected: int, actual: int, context: dict):
    """Log discrepancy for self-annealing learning."""
    ERROR_LOG.append({
        "article_id": article_id,
        "expected": expected,
        "actual": actual,
        "context": context,
    })


def calculate_fit_in_orientation(
    article_l: float,
    article_w: float,
    article_h: float,
    pocket_l: float,
    pocket_w: float,
    pocket_h: float,
    min_height_mm: float = 15.0,
) -> Tuple[int, Dict]:
    """
    Calculate how many units fit in given orientation.

    Returns:
        (units_fit, details_dict)
    """
    # Check if article fits at all
    if article_l > pocket_l or article_w > pocket_w or article_h > pocket_h:
        return 0, {"reason": "dimensions_exceed"}

    # Effective height (minimum 15mm to avoid extreme stacking)
    effective_h = max(article_h, min_height_mm)

    # Calculate how many fit in each dimension
    rows_deep = math.floor(pocket_l / article_l) if article_l > 0 else 0
    cols_wide = math.floor(pocket_w / article_w) if article_w > 0 else 0
    layers_high = math.floor(pocket_h / effective_h) if effective_h > 0 else 0

    units_fit = rows_deep * cols_wide * layers_high

    details = {
        "rows_deep": rows_deep,
        "cols_wide": cols_wide,
        "layers_high": layers_high,
        "article_dims": (article_l, article_w, article_h),
        "pocket_dims": (pocket_l, pocket_w, pocket_h),
    }

    return units_fit, details


def calculate_assq(
    article_length: float,
    article_width: float,
    article_height: float,
    article_weight: float,
    pocket_length: float,
    pocket_width: float,
    pocket_height: float,
    pocket_weight_limit: float = 20.0,
    allow_rotation: bool = True,
    allow_flip: bool = False,
    air_buffer_pct: float = 0.25,
    min_height_mm: float = 15.0,
) -> ASSQResult:
    """
    Calculate ASSQ (Assigned Sales Space Quantity) for an article in a pocket.

    Args:
        article_length: Article length in mm (depth direction)
        article_width: Article width in mm
        article_height: Article height in mm
        article_weight: Weight per unit in kg
        pocket_length: Pocket depth in mm
        pocket_width: Pocket width in mm
        pocket_height: Pocket height in mm
        pocket_weight_limit: Max weight per pocket in kg (default 20)
        allow_rotation: If True, try swapping length ↔ width
        allow_flip: If True, try swapping height with other dims
        air_buffer_pct: Headroom buffer (0.25 = 25% air, 75% fill)
        min_height_mm: Minimum height for stacking calc (avoid extremes)

    Returns:
        ASSQResult with assq count and metadata
    """

    # Validate inputs
    if any(d <= 0 for d in [article_length, article_width, article_height]):
        return ASSQResult(
            assq=0, fits=False, rotation_used=False, flip_used=False,
            weight_limited=False, fill_efficiency=0.0, fit_details={},
            error="Invalid article dimensions (zero or negative)"
        )

    if article_weight <= 0:
        return ASSQResult(
            assq=0, fits=False, rotation_used=False, flip_used=False,
            weight_limited=False, fill_efficiency=0.0, fit_details={},
            error="Invalid article weight (zero or negative)"
        )

    if article_weight > pocket_weight_limit:
        return ASSQResult(
            assq=0, fits=False, rotation_used=False, flip_used=False,
            weight_limited=True, fill_efficiency=0.0, fit_details={},
            error="Single unit exceeds pocket weight limit"
        )

    best_fit = 0
    best_details = {}
    rotation_used = False
    flip_used = False

    # Try original orientation
    fit, details = calculate_fit_in_orientation(
        article_length, article_width, article_height,
        pocket_length, pocket_width, pocket_height,
        min_height_mm
    )
    if fit > best_fit:
        best_fit = fit
        best_details = details

    # Try rotated (swap length ↔ width)
    if allow_rotation:
        fit_rot, details_rot = calculate_fit_in_orientation(
            article_width, article_length, article_height,  # swapped L/W
            pocket_length, pocket_width, pocket_height,
            min_height_mm
        )
        if fit_rot > best_fit:
            best_fit = fit_rot
            best_details = details_rot
            rotation_used = True

    # Try flipped orientations (swap height with length or width)
    if allow_flip:
        # Height becomes length
        fit_flip1, details_flip1 = calculate_fit_in_orientation(
            article_height, article_width, article_length,
            pocket_length, pocket_width, pocket_height,
            min_height_mm
        )
        if fit_flip1 > best_fit:
            best_fit = fit_flip1
            best_details = details_flip1
            flip_used = True
            rotation_used = False

        # Height becomes width
        fit_flip2, details_flip2 = calculate_fit_in_orientation(
            article_length, article_height, article_width,
            pocket_length, pocket_width, pocket_height,
            min_height_mm
        )
        if fit_flip2 > best_fit:
            best_fit = fit_flip2
            best_details = details_flip2
            flip_used = True
            rotation_used = False

    # Check if article fits at all
    if best_fit == 0:
        return ASSQResult(
            assq=0, fits=False, rotation_used=rotation_used, flip_used=flip_used,
            weight_limited=False, fill_efficiency=0.0, fit_details=best_details,
            error="Article dimensions don't fit pocket"
        )

    # Apply weight limit
    max_by_weight = math.floor(pocket_weight_limit / article_weight)
    weight_limited = best_fit > max_by_weight
    raw_assq = min(best_fit, max_by_weight)

    # Apply air buffer (25% air = multiply by 0.75)
    final_assq = math.floor(raw_assq * (1 - air_buffer_pct))

    # Ensure at least 1 if it fits
    final_assq = max(1, final_assq) if raw_assq > 0 else 0

    # Calculate fill efficiency
    article_volume = article_length * article_width * article_height
    pocket_volume = pocket_length * pocket_width * pocket_height
    fill_efficiency = (final_assq * article_volume) / pocket_volume if pocket_volume > 0 else 0

    return ASSQResult(
        assq=final_assq,
        fits=True,
        rotation_used=rotation_used,
        flip_used=flip_used,
        weight_limited=weight_limited,
        fill_efficiency=fill_efficiency,
        fit_details=best_details,
        error=None
    )


def calculate_pockets_needed(
    assq: int,
    weekly_demand: float,
    stock_weeks: float,
    max_pockets_per_article: int = 2,
) -> Tuple[int, bool, str]:
    """
    Calculate how many pockets needed for an article.

    Args:
        assq: Units that fit in one pocket
        weekly_demand: Expected weekly sales (EWS)
        stock_weeks: Desired weeks of stock coverage
        max_pockets_per_article: Maximum pockets allowed per article

    Returns:
        (pockets_needed, fits_constraint, reason)
    """
    if assq <= 0:
        return 0, False, "ASSQ is zero - article doesn't fit"

    if weekly_demand <= 0:
        return 1, True, "No demand - 1 pocket for display"

    units_needed = weekly_demand * stock_weeks
    pockets_raw = math.ceil(units_needed / assq)

    if pockets_raw <= max_pockets_per_article:
        return pockets_raw, True, f"Fits in {pockets_raw} pocket(s)"
    else:
        return pockets_raw, False, f"Needs {pockets_raw} pockets, exceeds max {max_pockets_per_article}"


# ============================================
# POCKET DIMENSIONS (from config/storeganizer.py)
# ============================================

POCKET_SIZES = {
    "xs": {"depth": 260, "width": 300, "height": 150},      # XS: 300x260x150
    "small": {"depth": 300, "width": 300, "height": 300},   # Small: 300x300x300
    "medium": {"depth": 300, "width": 450, "height": 300},  # Medium: 450x300x300
    "large": {"depth": 500, "width": 450, "height": 450},   # Large: 450x500x450
}


# ============================================
# TEST CASES (from Emmie's file)
# ============================================

def run_tests():
    """Run validation tests against known good data."""

    print("=" * 60)
    print("ASSQ Calculator Tests")
    print("=" * 60)

    # Test case 1: ÄNDLIG knife set (Consumer Pack)
    # From Emmie: CP 385x165x24mm, CP Speedcell Qty = 54
    # This article is long (385mm) - needs flip to fit well
    # With flip: 165(L) x 24(W) x 385(H) in Large (500x450x450)
    # = floor(500/165) x floor(450/24) x floor(450/385) = 3 x 18 x 1 = 54
    pocket = POCKET_SIZES["large"]
    result = calculate_assq(
        article_length=385, article_width=165, article_height=24,
        article_weight=0.33,
        pocket_length=pocket["depth"], pocket_width=pocket["width"], pocket_height=pocket["height"],
        allow_rotation=True,
        allow_flip=True,  # Enable flip for long flat items
        air_buffer_pct=0.0  # Test without buffer first
    )
    print(f"\nTest 1a (ÄNDLIG knife, Large pocket, no buffer):")
    print(f"  Expected: 54, Got: {result.assq}")
    print(f"  Rotation: {result.rotation_used}, Flip: {result.flip_used}")
    print(f"  Details: {result.fit_details}")

    # Test 1b: Same but with 25% air buffer
    result_buffered = calculate_assq(
        article_length=385, article_width=165, article_height=24,
        article_weight=0.33,
        pocket_length=pocket["depth"], pocket_width=pocket["width"], pocket_height=pocket["height"],
        allow_rotation=True,
        allow_flip=True,
        air_buffer_pct=0.25
    )
    print(f"\nTest 1b (ÄNDLIG knife, Large pocket, 25% buffer):")
    print(f"  Raw 54 × 0.75 = 40.5 → Expected: 40, Got: {result_buffered.assq}")

    # Test case 2: Small item in XS pocket
    # Something that SHOULD fit XS well
    pocket_xs = POCKET_SIZES["xs"]
    result2 = calculate_assq(
        article_length=100, article_width=80, article_height=50,
        article_weight=0.2,
        pocket_length=pocket_xs["depth"], pocket_width=pocket_xs["width"], pocket_height=pocket_xs["height"],
        allow_rotation=True,
        air_buffer_pct=0.25
    )
    # 260/100=2, 300/80=3, 150/50=3 → 2×3×3=18, ×0.75=13.5→13
    print(f"\nTest 2 (Small item in XS):")
    print(f"  Expected: 13, Got: {result2.assq}")
    print(f"  Details: {result2.fit_details}")

    # Test case 3: Large article that shouldn't fit XS
    result3 = calculate_assq(
        article_length=400, article_width=350, article_height=200,
        article_weight=5.0,
        pocket_length=pocket_xs["depth"], pocket_width=pocket_xs["width"], pocket_height=pocket_xs["height"],
        allow_rotation=True,
    )
    print(f"\nTest 3 (Large article in XS):")
    print(f"  Expected: 0 (doesn't fit), Got: {result3.assq}")
    print(f"  Fits: {result3.fits}, Error: {result3.error}")

    # Test case 4: Weight-limited article
    pocket_m = POCKET_SIZES["medium"]
    result4 = calculate_assq(
        article_length=100, article_width=100, article_height=100,
        article_weight=5.0,  # Heavy! 20kg limit / 5kg = max 4
        pocket_length=pocket_m["depth"], pocket_width=pocket_m["width"], pocket_height=pocket_m["height"],
        allow_rotation=True,
        air_buffer_pct=0.25
    )
    # 300/100=3, 450/100=4, 300/100=3 → 3×4×3=36 by volume
    # But 20/5=4 by weight, then 4×0.75=3
    print(f"\nTest 4 (Weight-limited article):")
    print(f"  Expected: 3 (weight limited), Got: {result4.assq}")
    print(f"  Weight limited: {result4.weight_limited}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_tests()
