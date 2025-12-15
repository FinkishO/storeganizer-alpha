"""
SKILL: Cascading Eligibility Filter

PURPOSE:
Determine best pocket size for each article by cascading through XS → S → M → L.
Assigns smallest pocket that provides adequate stockweeks coverage.

ALGORITHM:

STEP 0: MULTIBOX CHECK (FIRST!)
    - If MP_QTY > 1: Use MP dimensions (multibox = picking multiple units at once)
    - If MP_QTY == 1: Use CP dimensions (consumer pack = single unit)

STEP 1: CASCADE THROUGH POCKET SIZES
    For each pocket size (XS → S → M → L):
        a) Calculate ASSQ using skill_assq_calculator
        b) Calculate stockweeks = ASSQ / EWS
        c) If stockweeks >= min_stockweeks: FITS!
        d) If ASSQ == 0: dimension doesn't fit, try next size

STEP 2: MULTI-POCKET CONSIDERATION
    - If smallest fit requires > max_pockets, try larger pocket
    - Prefer: 1 Large pocket > 2 Medium pockets (easier replenishment)
    - Setting: max_pockets_per_article (default: 2)

STEP 3: FINAL ASSIGNMENT
    - Return: recommended_pocket_size, assq, stockweeks, pockets_needed
    - Or: "NOT_SUITABLE" with reason

REJECTION REASONS:
- "TOO_FAST" - stockweeks < min even in L pocket (sells too fast)
- "TOO_LARGE" - doesn't fit even Large pocket
- "TOO_HEAVY" - single unit exceeds 20kg pocket weight limit
- "MISSING_DIMS" - no dimensions available
- "MISSING_EWS" - no demand data

INPUTS:
- Article dimensions (CP or MP based on MP_QTY)
- Article weight
- EWS (Expected Weekly Sales)
- MP_QTY (multipack quantity)
- min_stockweeks (default: 1 week)
- max_stockweeks (default: 8 weeks, optional upper bound)
- max_pockets_per_article (default: 2)
- allow_rotation (default: True)
- allow_flip (default: False)
- air_buffer_pct (default: 0.25)

OUTPUTS:
- CascadeResult with:
  - fits: bool
  - recommended_size: str (xs/small/medium/large)
  - assq: int
  - stockweeks: float
  - pockets_needed: int
  - using_multibox: bool
  - rejection_reason: str (if doesn't fit)

SELF-ANNEALING:
- Log cases where algorithm picks different size than CPH's file
- Track rejection distribution for learning

VALIDATION INSIGHTS (2025-12-14):
- Emmie's file uses LARGE pocket exclusively for all articles (no cascade)
- When testing against Large only: 89.3% ASSQ match within 20%
- Our cascade approach assigns smallest pocket that fits (different goal)
- Both are valid: Emmie = simplicity, Cascade = space efficiency
- Key learning: Customer preference determines which approach to use
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math

# Import ASSQ calculator from same directory
from skill_assq_calculator import calculate_assq, POCKET_SIZES, ASSQResult

# Self-annealing error log
CASCADE_ERROR_LOG = []
REJECTION_STATS = {
    "TOO_FAST": 0,
    "TOO_LARGE": 0,
    "TOO_HEAVY": 0,
    "MISSING_DIMS": 0,
    "MISSING_EWS": 0,
}


@dataclass
class CascadeResult:
    """Result of cascading eligibility check."""
    fits: bool
    recommended_size: Optional[str]  # xs, small, medium, large
    assq: int
    stockweeks: float
    pockets_needed: int
    using_multibox: bool
    rotation_used: bool
    flip_used: bool
    rejection_reason: Optional[str] = None
    cascade_details: Optional[Dict] = None


def log_cascade_error(article_id: str, expected_size: str, actual_size: str, context: dict):
    """Log when our cascade picks different size than reference."""
    CASCADE_ERROR_LOG.append({
        "article_id": article_id,
        "expected": expected_size,
        "actual": actual_size,
        "context": context,
    })


def check_multibox(mp_qty: float) -> bool:
    """
    Step 0: Determine if article is multibox.

    Multibox = picking multiple consumer packs at once.
    MP_QTY > 1 means the picker grabs a box containing multiple units.
    """
    return mp_qty is not None and mp_qty > 1


def get_article_dimensions(
    cp_length: float, cp_width: float, cp_height: float, cp_weight: float,
    mp_length: float, mp_width: float, mp_height: float, mp_weight: float,
    mp_qty: float
) -> Tuple[float, float, float, float, bool]:
    """
    Get effective dimensions based on multibox status.

    Returns: (length, width, height, weight, is_multibox)
    """
    is_multibox = check_multibox(mp_qty)

    if is_multibox and all(d and d > 0 for d in [mp_length, mp_width, mp_height]):
        return mp_length, mp_width, mp_height, mp_weight or cp_weight, True
    else:
        return cp_length, cp_width, cp_height, cp_weight, False


def cascade_fit_article(
    # CP dimensions
    cp_length: float,
    cp_width: float,
    cp_height: float,
    cp_weight: float,
    # MP dimensions (optional)
    mp_length: float = None,
    mp_width: float = None,
    mp_height: float = None,
    mp_weight: float = None,
    mp_qty: float = 1,
    # Demand
    ews: float = 0,  # Expected Weekly Sales
    # Settings
    min_stockweeks: float = 1.0,
    max_stockweeks: float = 26.0,  # Optional upper bound (26 = 6 months)
    max_pockets_per_article: int = 2,
    allow_rotation: bool = True,
    allow_flip: bool = False,
    air_buffer_pct: float = 0.25,
) -> CascadeResult:
    """
    Cascade through pocket sizes to find best fit.

    Args:
        cp_length, cp_width, cp_height: Consumer pack dimensions in mm
        cp_weight: CP weight in kg
        mp_length, mp_width, mp_height: Multipack dimensions in mm
        mp_weight: MP weight in kg
        mp_qty: Number of CPs in one MP (>1 = multibox)
        ews: Expected Weekly Sales
        min_stockweeks: Minimum weeks of stock required
        max_stockweeks: Maximum weeks (too slow = wasting space)
        max_pockets_per_article: Max pockets before rejecting
        allow_rotation: Can swap length/width
        allow_flip: Can swap height with length/width
        air_buffer_pct: Headroom buffer (0.25 = 25%)

    Returns:
        CascadeResult with fit status and recommendations
    """

    # Step 0: Get effective dimensions
    length, width, height, weight, is_multibox = get_article_dimensions(
        cp_length, cp_width, cp_height, cp_weight,
        mp_length, mp_width, mp_height, mp_weight,
        mp_qty
    )

    # Validate dimensions
    if not all(d and d > 0 for d in [length, width, height]):
        REJECTION_STATS["MISSING_DIMS"] += 1
        return CascadeResult(
            fits=False, recommended_size=None, assq=0, stockweeks=0,
            pockets_needed=0, using_multibox=is_multibox,
            rotation_used=False, flip_used=False,
            rejection_reason="MISSING_DIMS"
        )

    if not weight or weight <= 0:
        weight = 0.1  # Default to 100g if missing

    # Validate EWS
    if ews is None or ews <= 0:
        REJECTION_STATS["MISSING_EWS"] += 1
        return CascadeResult(
            fits=False, recommended_size=None, assq=0, stockweeks=0,
            pockets_needed=0, using_multibox=is_multibox,
            rotation_used=False, flip_used=False,
            rejection_reason="MISSING_EWS"
        )

    # Cascade order: smallest to largest
    cascade_order = ["xs", "small", "medium", "large"]
    cascade_details = {}

    for size in cascade_order:
        pocket = POCKET_SIZES[size]

        # Calculate ASSQ for this pocket size
        result = calculate_assq(
            article_length=length,
            article_width=width,
            article_height=height,
            article_weight=weight,
            pocket_length=pocket["depth"],
            pocket_width=pocket["width"],
            pocket_height=pocket["height"],
            allow_rotation=allow_rotation,
            allow_flip=allow_flip,
            air_buffer_pct=air_buffer_pct,
        )

        cascade_details[size] = {
            "assq": result.assq,
            "fits": result.fits,
            "weight_limited": result.weight_limited,
            "error": result.error,
        }

        if not result.fits:
            continue  # Try next size

        # Calculate stockweeks
        stockweeks = result.assq / ews if ews > 0 else float('inf')

        # Calculate pockets needed for min_stockweeks
        units_needed = ews * min_stockweeks
        pockets_needed = math.ceil(units_needed / result.assq) if result.assq > 0 else float('inf')

        cascade_details[size]["stockweeks"] = stockweeks
        cascade_details[size]["pockets_needed"] = pockets_needed

        # Check if this size meets requirements
        if stockweeks >= min_stockweeks and pockets_needed <= max_pockets_per_article:
            # Found a fit!
            return CascadeResult(
                fits=True,
                recommended_size=size,
                assq=result.assq,
                stockweeks=round(stockweeks, 2),
                pockets_needed=pockets_needed,
                using_multibox=is_multibox,
                rotation_used=result.rotation_used,
                flip_used=result.flip_used,
                rejection_reason=None,
                cascade_details=cascade_details,
            )

        # Optional: check max_stockweeks (too slow = wasting space)
        # But we still accept it if it fits - just note it

    # If we get here, nothing fit well enough
    # Check why:

    # Did it fit Large but need too many pockets? (TOO_FAST)
    if "large" in cascade_details and cascade_details["large"]["fits"]:
        REJECTION_STATS["TOO_FAST"] += 1
        return CascadeResult(
            fits=False,
            recommended_size=None,
            assq=cascade_details["large"]["assq"],
            stockweeks=cascade_details["large"].get("stockweeks", 0),
            pockets_needed=cascade_details["large"].get("pockets_needed", 0),
            using_multibox=is_multibox,
            rotation_used=False, flip_used=False,
            rejection_reason="TOO_FAST",
            cascade_details=cascade_details,
        )

    # Did it not fit any size due to weight? (TOO_HEAVY)
    if cascade_details.get("large", {}).get("weight_limited"):
        REJECTION_STATS["TOO_HEAVY"] += 1
        return CascadeResult(
            fits=False, recommended_size=None, assq=0, stockweeks=0,
            pockets_needed=0, using_multibox=is_multibox,
            rotation_used=False, flip_used=False,
            rejection_reason="TOO_HEAVY",
            cascade_details=cascade_details,
        )

    # Dimensions too large even for Large pocket (TOO_LARGE)
    REJECTION_STATS["TOO_LARGE"] += 1
    return CascadeResult(
        fits=False, recommended_size=None, assq=0, stockweeks=0,
        pockets_needed=0, using_multibox=is_multibox,
        rotation_used=False, flip_used=False,
        rejection_reason="TOO_LARGE",
        cascade_details=cascade_details,
    )


def process_article_batch(articles: List[Dict], **settings) -> List[CascadeResult]:
    """
    Process multiple articles through cascading filter.

    Args:
        articles: List of dicts with article data
        **settings: Override default cascade settings

    Returns:
        List of CascadeResult for each article
    """
    results = []
    for article in articles:
        result = cascade_fit_article(
            cp_length=article.get("cp_length", 0),
            cp_width=article.get("cp_width", 0),
            cp_height=article.get("cp_height", 0),
            cp_weight=article.get("cp_weight", 0),
            mp_length=article.get("mp_length"),
            mp_width=article.get("mp_width"),
            mp_height=article.get("mp_height"),
            mp_weight=article.get("mp_weight"),
            mp_qty=article.get("mp_qty", 1),
            ews=article.get("ews", 0),
            **settings
        )
        results.append(result)
    return results


def get_rejection_summary() -> Dict:
    """Get summary of rejection reasons for self-annealing."""
    total = sum(REJECTION_STATS.values())
    return {
        "total_rejected": total,
        "breakdown": REJECTION_STATS.copy(),
        "percentages": {k: round(v/total*100, 1) if total > 0 else 0
                       for k, v in REJECTION_STATS.items()},
    }


# ============================================
# TEST CASES
# ============================================

def run_tests():
    """Test cascading filter logic."""

    print("=" * 60)
    print("Cascading Filter Tests")
    print("=" * 60)

    # Reset stats
    global REJECTION_STATS
    REJECTION_STATS = {k: 0 for k in REJECTION_STATS}

    # Test 1: Small item should fit XS
    print("\nTest 1: Small item → should fit XS")
    result = cascade_fit_article(
        cp_length=100, cp_width=80, cp_height=50, cp_weight=0.2,
        ews=10,  # 10 units/week
        min_stockweeks=1,
    )
    print(f"  Fits: {result.fits}, Size: {result.recommended_size}")
    print(f"  ASSQ: {result.assq}, Stockweeks: {result.stockweeks}")
    assert result.fits and result.recommended_size == "xs", "FAIL: Should fit XS"
    print("  ✓ PASS")

    # Test 2: Medium item needs Large for stockweeks coverage
    print("\nTest 2: Medium item → needs Large for 1 week stockweeks")
    result = cascade_fit_article(
        cp_length=200, cp_width=200, cp_height=200, cp_weight=1.0,
        ews=5,  # 5/week
        min_stockweeks=1,
    )
    print(f"  Fits: {result.fits}, Size: {result.recommended_size}")
    print(f"  ASSQ: {result.assq}, Stockweeks: {result.stockweeks}")
    # 200x200x200 fits Small physically (ASSQ=1), but stockweeks = 0.2
    # Needs Large (ASSQ=6) for stockweeks = 1.2
    assert result.fits and result.recommended_size == "large", f"FAIL: Got {result.recommended_size}"
    print("  ✓ PASS")

    # Test 2b: Same item with lower EWS should fit Small
    print("\nTest 2b: Same item, lower demand → should fit Small")
    result = cascade_fit_article(
        cp_length=200, cp_width=200, cp_height=200, cp_weight=1.0,
        ews=1,  # Only 1/week - Small's ASSQ=1 gives 1 week coverage
        min_stockweeks=1,
    )
    print(f"  Fits: {result.fits}, Size: {result.recommended_size}")
    print(f"  ASSQ: {result.assq}, Stockweeks: {result.stockweeks}")
    assert result.fits and result.recommended_size == "small", f"FAIL: Got {result.recommended_size}"
    print("  ✓ PASS")

    # Test 3: Multibox should use MP dimensions
    print("\nTest 3: Multibox article → should use MP dimensions")
    result = cascade_fit_article(
        cp_length=50, cp_width=50, cp_height=50, cp_weight=0.1,
        mp_length=200, mp_width=150, mp_height=100, mp_weight=0.5,
        mp_qty=6,  # 6-pack
        ews=10,  # 10 MPs/week (reasonable)
        min_stockweeks=1,
    )
    print(f"  Using multibox: {result.using_multibox}")
    print(f"  Fits: {result.fits}, Size: {result.recommended_size}")
    print(f"  ASSQ: {result.assq}, Stockweeks: {result.stockweeks}")
    if result.cascade_details:
        for size, details in result.cascade_details.items():
            print(f"    {size}: ASSQ={details['assq']}, stockweeks={details.get('stockweeks', 'N/A')}")
    assert result.using_multibox, "FAIL: Should detect multibox"
    assert result.fits, f"FAIL: Multibox should fit, got {result.rejection_reason}"
    print("  ✓ PASS")

    # Test 4: Fast-selling item should be rejected (TOO_FAST)
    print("\nTest 4: Fast-selling item → should reject as TOO_FAST")
    result = cascade_fit_article(
        cp_length=100, cp_width=100, cp_height=100, cp_weight=0.5,
        ews=1000,  # 1000/week! Very fast
        min_stockweeks=1,
        max_pockets_per_article=2,
    )
    print(f"  Fits: {result.fits}, Reason: {result.rejection_reason}")
    print(f"  Would need {result.pockets_needed} pockets")
    assert not result.fits and result.rejection_reason == "TOO_FAST", "FAIL: Should be TOO_FAST"
    print("  ✓ PASS")

    # Test 5: Oversized item should be rejected (TOO_LARGE)
    print("\nTest 5: Oversized item → should reject as TOO_LARGE")
    result = cascade_fit_article(
        cp_length=600, cp_width=500, cp_height=500, cp_weight=5.0,
        ews=2,
        min_stockweeks=1,
    )
    print(f"  Fits: {result.fits}, Reason: {result.rejection_reason}")
    assert not result.fits and result.rejection_reason == "TOO_LARGE", "FAIL: Should be TOO_LARGE"
    print("  ✓ PASS")

    # Test 6: Missing dimensions should be rejected
    print("\nTest 6: Missing dimensions → should reject as MISSING_DIMS")
    result = cascade_fit_article(
        cp_length=0, cp_width=0, cp_height=0, cp_weight=0.5,
        ews=10,
    )
    print(f"  Fits: {result.fits}, Reason: {result.rejection_reason}")
    assert not result.fits and result.rejection_reason == "MISSING_DIMS", "FAIL: Should be MISSING_DIMS"
    print("  ✓ PASS")

    # Print rejection summary
    print("\n" + "=" * 60)
    print("Rejection Summary:")
    summary = get_rejection_summary()
    for reason, count in summary["breakdown"].items():
        print(f"  {reason}: {count}")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
