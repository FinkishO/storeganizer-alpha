"""
ROI Calculator for Storeganizer investments.

Replicates the Storeganizer ROI Excel template logic to calculate:
- Space savings (reduced square meters needed)
- Utility cost reductions
- Labor efficiency gains
- ROI percentage and payback period
- Net Present Value (NPV) over 5 years

Based on Dimitri's Excel template with WACC=7.61% and inflation=1%.

Usage Example:
    from core.roi_calculator import calculate_roi, ROIInputs

    # Full detailed calculation
    inputs = ROIInputs(
        rack_width_m=2.7,
        rack_depth_m=1.0,
        locations_per_rack_before=40,
        locations_per_rack_after=90,
        num_racks_before=30,
        num_racks_after=15,
        cost_per_sqm_annual=90.0,
        wage_per_hour=28.0,
        efficiency_increase_pct=0.15,
        total_investment=35000.0
    )

    results = calculate_roi(inputs)
    print(results.format_summary())

    # Quick calculation
    from core.roi_calculator import quick_roi
    results = quick_roi(
        investment=35000,
        sqm_before=81,
        sqm_after=40.5
    )
    print(f"ROI: {results.roi_pct:.1f}%, Payback: {results.payback_years:.1f} years")
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# Financial constants from Excel template
WACC = 0.0761  # Weighted Average Cost of Capital
INFLATION = 0.01  # Annual inflation rate
NPV_PROJECTION_YEARS = 5  # Standard 5-year NPV calculation


@dataclass
class ROIInputs:
    """
    Input parameters for ROI calculation.

    Space Parameters:
        rack_width_m: Width of each rack in meters
        rack_depth_m: Depth of each rack in meters
        locations_per_rack_before: Storage locations per rack (current setup)
        locations_per_rack_after: Storage locations per rack (with Storeganizer)
        num_racks_before: Number of racks needed (current setup)
        num_racks_after: Number of racks needed (with Storeganizer)

    Cost Parameters:
        cost_per_sqm_annual: Annual real estate cost per square meter (€)
        utility_cost_per_sqm_annual: Annual utility cost per square meter (€)

    Labor Parameters:
        picks_per_hour_per_person: Pick rate per staff member per hour
        num_staff: Number of picking staff
        wage_per_hour: Hourly wage per staff member (€)
        efficiency_increase_pct: Labor efficiency gain (e.g., 0.15 = 15%)

    Investment Parameters:
        total_investment: Total upfront investment in Storeganizer (€)
    """
    # Space
    rack_width_m: float = 2.7
    rack_depth_m: float = 1.0
    locations_per_rack_before: int = 40
    locations_per_rack_after: int = 90
    num_racks_before: int = 30
    num_racks_after: int = 15

    # Costs
    cost_per_sqm_annual: float = 90.0
    utility_cost_per_sqm_annual: float = 10.0

    # Labor
    picks_per_hour_per_person: int = 20
    num_staff: int = 2
    wage_per_hour: float = 28.0
    efficiency_increase_pct: float = 0.15  # 15% efficiency gain

    # Investment
    total_investment: float = 35000.0

    def validate(self) -> list[str]:
        """
        Validate inputs and return list of warnings/errors.

        Returns:
            List of validation messages (empty if all valid)
        """
        issues = []

        # Check non-negative values
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, (int, float)) and field_value < 0:
                issues.append(f"ERROR: {field_name} cannot be negative (got {field_value})")

        # Check zero values that would break calculations
        if self.num_staff == 0:
            issues.append("ERROR: num_staff cannot be zero")

        if self.total_investment == 0:
            issues.append("ERROR: total_investment cannot be zero (needed for payback calculation)")

        # Sanity warnings
        if self.efficiency_increase_pct > 0.5:
            issues.append(f"WARNING: efficiency_increase_pct is {self.efficiency_increase_pct:.1%} - " +
                         "gains above 50% are unusually high")

        # Logical consistency checks
        if self.locations_per_rack_after <= self.locations_per_rack_before:
            issues.append(f"WARNING: locations_per_rack_after ({self.locations_per_rack_after}) " +
                         f"should exceed locations_per_rack_before ({self.locations_per_rack_before})")

        if self.num_racks_after >= self.num_racks_before:
            issues.append(f"WARNING: num_racks_after ({self.num_racks_after}) " +
                         f"should be less than num_racks_before ({self.num_racks_before})")

        return issues


@dataclass
class ROIResults:
    """
    Calculated ROI results.

    Space Metrics:
        total_sqm_before: Total square meters (current)
        total_sqm_after: Total square meters (with Storeganizer)
        sqm_saved: Square meters saved
        total_locations_before: Total storage locations (current)
        total_locations_after: Total storage locations (with Storeganizer)
        locations_per_sqm_before: Storage density (current)
        locations_per_sqm_after: Storage density (with Storeganizer)
        walking_distance_reduction_m: Estimated walking distance reduction

    Annual Savings:
        annual_space_saving: Annual real estate cost savings (€)
        annual_utility_saving: Annual utility cost savings (€)
        annual_labor_saving: Annual labor cost savings (€)
        total_annual_saving: Total annual savings (€)

    Labor Metrics:
        annual_picks: Estimated annual pick volume
        annual_staff_cost_before: Annual staff costs (current, €)
        annual_staff_cost_after: Annual staff costs (with Storeganizer, €)

    ROI Metrics:
        roi_pct: Return on investment (percentage)
        payback_years: Years to recover investment
        npv: Net Present Value over 5 years (€)

    Validation:
        warnings: List of sanity check warnings
    """
    # Space metrics
    total_sqm_before: float
    total_sqm_after: float
    sqm_saved: float
    total_locations_before: int
    total_locations_after: int
    locations_per_sqm_before: float
    locations_per_sqm_after: float
    walking_distance_reduction_m: float

    # Annual savings
    annual_space_saving: float
    annual_utility_saving: float
    annual_labor_saving: float
    total_annual_saving: float

    # Labor metrics
    annual_picks: int
    annual_staff_cost_before: float
    annual_staff_cost_after: float

    # ROI metrics
    roi_pct: float
    payback_years: float
    npv: float

    # Validation
    warnings: list[str] = field(default_factory=list)

    def format_summary(self) -> str:
        """Generate a human-readable summary of ROI results."""
        lines = [
            "=" * 60,
            "STOREGANIZER ROI ANALYSIS",
            "=" * 60,
            "",
            "SPACE OPTIMIZATION:",
            f"  Before: {self.total_sqm_before:.1f} m² → After: {self.total_sqm_after:.1f} m²",
            f"  Space saved: {self.sqm_saved:.1f} m² ({self.sqm_saved/self.total_sqm_before*100:.1f}% reduction)",
            f"  Storage density: {self.locations_per_sqm_before:.1f} → {self.locations_per_sqm_after:.1f} locations/m²",
            f"  Walking distance reduced by: {self.walking_distance_reduction_m:.1f} m",
            "",
            "ANNUAL SAVINGS:",
            f"  Real estate: €{self.annual_space_saving:,.2f}",
            f"  Utilities: €{self.annual_utility_saving:,.2f}",
            f"  Labor: €{self.annual_labor_saving:,.2f}",
            f"  TOTAL: €{self.total_annual_saving:,.2f} per year",
            "",
            "ROI METRICS:",
            f"  ROI: {self.roi_pct:.1f}%",
            f"  Payback period: {self.payback_years:.2f} years",
            f"  NPV (5-year): €{self.npv:,.2f}",
            "=" * 60,
        ]

        if self.warnings:
            lines.extend([
                "",
                "⚠️  WARNINGS:",
            ])
            for warning in self.warnings:
                lines.append(f"  - {warning}")
            lines.append("=" * 60)

        return "\n".join(lines)


def calculate_roi(inputs: ROIInputs) -> ROIResults:
    """
    Calculate ROI for a Storeganizer investment.

    Args:
        inputs: ROIInputs dataclass with all required parameters

    Returns:
        ROIResults dataclass with calculated metrics

    Raises:
        ValueError: If inputs fail validation with errors
    """
    # Validate inputs
    validation_messages = inputs.validate()
    errors = [msg for msg in validation_messages if msg.startswith("ERROR")]
    warnings = [msg for msg in validation_messages if msg.startswith("WARNING")]

    if errors:
        error_text = "\n".join(errors)
        raise ValueError(f"Invalid ROI inputs:\n{error_text}")

    for warning in warnings:
        logger.warning(warning)

    # --- SPACE CALCULATIONS ---

    # Square meters per rack
    sqm_per_rack = inputs.rack_width_m * inputs.rack_depth_m

    # Total square meters before and after
    total_sqm_before = sqm_per_rack * inputs.num_racks_before
    total_sqm_after = sqm_per_rack * inputs.num_racks_after
    sqm_saved = total_sqm_before - total_sqm_after

    # Total storage locations before and after
    total_locations_before = inputs.locations_per_rack_before * inputs.num_racks_before
    total_locations_after = inputs.locations_per_rack_after * inputs.num_racks_after

    # Storage density (locations per square meter)
    locations_per_sqm_before = total_locations_before / total_sqm_before
    locations_per_sqm_after = total_locations_after / total_sqm_after

    # Walking distance reduction (simplified: proportional to sqm reduction)
    # Typically 50% reduction due to vertical storage vs. horizontal
    walking_distance_reduction_m = sqm_saved

    # --- ANNUAL COST CALCULATIONS ---

    # Space costs
    annual_space_cost_before = total_sqm_before * inputs.cost_per_sqm_annual
    annual_space_cost_after = total_sqm_after * inputs.cost_per_sqm_annual
    annual_space_saving = annual_space_cost_before - annual_space_cost_after

    # Utility costs
    annual_utility_before = total_sqm_before * inputs.utility_cost_per_sqm_annual
    annual_utility_after = total_sqm_after * inputs.utility_cost_per_sqm_annual
    annual_utility_saving = annual_utility_before - annual_utility_after

    # --- LABOR CALCULATIONS ---

    # Annual pick volume estimate
    # Formula: picks/hour * staff * 8 hours/day * 260 work days * 0.8 (80% utilization)
    annual_picks = int(
        inputs.picks_per_hour_per_person *
        inputs.num_staff *
        8 *  # hours per day
        260 *  # work days per year
        0.8  # 80% utilization factor
    )

    # Annual staff costs
    annual_staff_cost_before = inputs.num_staff * inputs.wage_per_hour * 8 * 260

    # After Storeganizer: reduced by efficiency gain percentage
    # Note: efficiency_increase_pct represents the cost reduction, not staff reduction
    # A 15% efficiency gain means labor costs are 85% of original
    annual_staff_cost_after = annual_staff_cost_before * (1 - inputs.efficiency_increase_pct)

    # Annual labor savings
    annual_labor_saving = annual_staff_cost_before - annual_staff_cost_after

    # --- TOTAL SAVINGS ---

    total_annual_saving = annual_space_saving + annual_utility_saving + annual_labor_saving

    # --- ROI CALCULATIONS ---

    # ROI percentage: (annual savings / investment) * 100
    roi_pct = (total_annual_saving / inputs.total_investment) * 100

    # Payback period: investment / annual savings
    payback_years = inputs.total_investment / total_annual_saving

    # NPV calculation (5-year projection with WACC and inflation)
    npv = calculate_npv(
        annual_saving=total_annual_saving,
        investment=inputs.total_investment,
        years=NPV_PROJECTION_YEARS,
        wacc=WACC,
        inflation=INFLATION
    )

    # --- SANITY CHECKS ---

    if roi_pct > 500:
        warnings.append(f"ROI is {roi_pct:.1f}% - exceptionally high, verify inputs")

    if payback_years > 10:
        warnings.append(f"Payback period is {payback_years:.1f} years - unusually long")

    if npv < 0:
        warnings.append(f"NPV is negative (€{npv:,.2f}) - investment may not be worthwhile")

    # --- RETURN RESULTS ---

    return ROIResults(
        # Space metrics
        total_sqm_before=total_sqm_before,
        total_sqm_after=total_sqm_after,
        sqm_saved=sqm_saved,
        total_locations_before=total_locations_before,
        total_locations_after=total_locations_after,
        locations_per_sqm_before=locations_per_sqm_before,
        locations_per_sqm_after=locations_per_sqm_after,
        walking_distance_reduction_m=walking_distance_reduction_m,

        # Annual savings
        annual_space_saving=annual_space_saving,
        annual_utility_saving=annual_utility_saving,
        annual_labor_saving=annual_labor_saving,
        total_annual_saving=total_annual_saving,

        # Labor metrics
        annual_picks=annual_picks,
        annual_staff_cost_before=annual_staff_cost_before,
        annual_staff_cost_after=annual_staff_cost_after,

        # ROI metrics
        roi_pct=roi_pct,
        payback_years=payback_years,
        npv=npv,

        # Validation
        warnings=warnings
    )


def calculate_npv(
    annual_saving: float,
    investment: float,
    years: int = NPV_PROJECTION_YEARS,
    wacc: float = WACC,
    inflation: float = INFLATION
) -> float:
    """
    Calculate Net Present Value over specified period.

    NPV accounts for:
    - Initial investment (negative cash flow at t=0)
    - Annual savings adjusted for inflation
    - Discounting future cash flows by WACC

    Formula:
        NPV = -Investment + Σ(year=1 to N) [Saving * (1+inflation)^year / (1+WACC)^year]

    Args:
        annual_saving: First-year annual savings (€)
        investment: Initial investment (€)
        years: Number of years to project (default: 5)
        wacc: Weighted Average Cost of Capital (default: 7.61%)
        inflation: Annual inflation rate (default: 1%)

    Returns:
        Net Present Value (€)
    """
    npv = -investment  # Initial investment is negative cash flow

    for year in range(1, years + 1):
        # Adjust savings for inflation
        adjusted_saving = annual_saving * ((1 + inflation) ** year)

        # Discount to present value
        discounted_value = adjusted_saving / ((1 + wacc) ** year)

        npv += discounted_value

    return npv


def test_roi_calculator():
    """
    Test ROI calculator against Excel template values.

    Expected results (from Dimitri's template):
    - Annual saving: ~€19,243
    - ROI: ~45.5%
    - Payback: ~1.82 years
    """
    print("Testing ROI calculator against Excel template...\n")

    # Use exact values from Excel template
    inputs = ROIInputs(
        # Space
        rack_width_m=2.7,
        rack_depth_m=1.0,
        locations_per_rack_before=40,
        locations_per_rack_after=90,
        num_racks_before=30,
        num_racks_after=15,

        # Costs
        cost_per_sqm_annual=90.0,
        utility_cost_per_sqm_annual=10.0,

        # Labor
        picks_per_hour_per_person=20,
        num_staff=2,
        wage_per_hour=28.0,
        efficiency_increase_pct=0.15,

        # Investment
        total_investment=35000.0
    )

    results = calculate_roi(inputs)

    # Print formatted results
    print(results.format_summary())

    # Detailed breakdown for debugging
    print("\nDETAILED BREAKDOWN:")
    print(f"  Space saving: €{results.annual_space_saving:,.2f}")
    print(f"  Utility saving: €{results.annual_utility_saving:,.2f}")
    print(f"  Labor saving: €{results.annual_labor_saving:,.2f}")
    print(f"    (Staff cost before: €{results.annual_staff_cost_before:,.2f})")
    print(f"    (Staff cost after: €{results.annual_staff_cost_after:,.2f})")
    print(f"  Total: €{results.total_annual_saving:,.2f}")

    # Validate against expected values
    print("\nValidation against Excel template:")
    print(f"  Expected annual saving: ~€19,243")
    print(f"  Calculated: €{results.total_annual_saving:,.2f}")
    print(f"  Match: {'✓' if abs(results.total_annual_saving - 19243) < 1000 else '✗'}")

    print(f"\n  Expected ROI: ~45.5%")
    print(f"  Calculated: {results.roi_pct:.1f}%")
    print(f"  Match: {'✓' if abs(results.roi_pct - 45.5) < 5 else '✗'}")

    print(f"\n  Expected payback: ~1.82 years")
    print(f"  Calculated: {results.payback_years:.2f} years")
    print(f"  Match: {'✓' if abs(results.payback_years - 1.82) < 0.2 else '✗'}")

    print("\nNOTE: Minor differences from Excel template may be due to:")
    print("  - Rounding differences in intermediate calculations")
    print("  - Different assumptions about work days/hours")
    print("  - Excel template using different efficiency calculation method")

    return results


def quick_roi(
    investment: float,
    sqm_before: float,
    sqm_after: float,
    labor_savings_pct: float = 0.15,
    cost_per_sqm: float = 90.0
) -> ROIResults:
    """
    Quick ROI calculation with simplified inputs.

    Useful for rapid "what-if" scenarios.

    Args:
        investment: Total investment (€)
        sqm_before: Square meters before (m²)
        sqm_after: Square meters after (m²)
        labor_savings_pct: Labor cost reduction (default: 15%)
        cost_per_sqm: Annual real estate cost per m² (default: €90)

    Returns:
        ROIResults with calculated metrics

    Example:
        >>> results = quick_roi(
        ...     investment=35000,
        ...     sqm_before=81,
        ...     sqm_after=40.5,
        ...     labor_savings_pct=0.15,
        ...     cost_per_sqm=90
        ... )
        >>> print(f"ROI: {results.roi_pct:.1f}%")
    """
    # Back-calculate rack configuration from sqm
    rack_sqm = 2.7  # Assume standard 2.7m² per rack
    num_racks_before = int(sqm_before / rack_sqm)
    num_racks_after = int(sqm_after / rack_sqm)

    inputs = ROIInputs(
        # Space (back-calculated)
        rack_width_m=2.7,
        rack_depth_m=1.0,
        locations_per_rack_before=40,
        locations_per_rack_after=90,
        num_racks_before=num_racks_before,
        num_racks_after=num_racks_after,

        # Costs
        cost_per_sqm_annual=cost_per_sqm,
        utility_cost_per_sqm_annual=10.0,

        # Labor
        picks_per_hour_per_person=20,
        num_staff=2,
        wage_per_hour=28.0,
        efficiency_increase_pct=labor_savings_pct,

        # Investment
        total_investment=investment
    )

    return calculate_roi(inputs)


if __name__ == "__main__":
    # Run test
    logging.basicConfig(level=logging.INFO)
    test_roi_calculator()
