"""
Validation test: Run cascading filter against Emmie's 1,434 suitable articles.

Compares our ASSQ calculations against Emmie's "CP Speedcell Qty" column.
"""

import pandas as pd
import sys
sys.path.insert(0, '/home/flinux/storeganizer-planner/storeganizer-alpha/skills')

from skill_cascading_filter import cascade_fit_article, get_rejection_summary, REJECTION_STATS

# Column mapping from Emmie's file (actual column names from Excel)
# Header is on row 1 (use header=1 when loading)
COLUMN_MAP = {
    # CP dimensions (mm)
    "cp_length": "CP Length",
    "cp_width": "CP Width",
    "cp_height": "CP Height",
    "cp_weight": "CP Max Weight",  # NOT "CP Gross Weight"
    # MP dimensions (mm)
    "mp_length": "MP Length",
    "mp_width": "MP Width",
    "mp_height": "MP Height",
    "mp_weight": "MP Weight",  # NOT "MP Gross Weight"
    "mp_qty": "MP QTY",
    # Demand
    "ews": "Total EWS",
    # Reference values from Emmie's calculations
    "emmie_assq": "CP Speedcell Qty",
    "emmie_speedcells": "Speedcells",  # Number of pockets assigned
    "article_name": "Article Name",
    "article_no": "Article Number",
}


def load_emmie_data(filepath: str) -> pd.DataFrame:
    """Load and filter Emmie's suitable articles."""
    # Header is on row 1, not row 0
    df = pd.read_excel(filepath, header=1)

    print(f"Columns found: {list(df.columns[:10])}...")

    # Filter to suitable articles (those with Speedcells > 0)
    if "Speedcells" in df.columns:
        df_suitable = df[df["Speedcells"] > 0].copy()
        print(f"Loaded {len(df_suitable)} suitable articles from {len(df)} total")
        return df_suitable
    else:
        print("WARNING: No 'Speedcells' column found, using all rows")
        return df


def safe_float(val, default=0.0):
    """Convert value to float safely."""
    try:
        if pd.isna(val):
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


def run_validation(df: pd.DataFrame):
    """Run cascading filter on all articles and compare to Emmie's values."""

    results = []
    matches = 0
    within_20pct = 0
    total_processed = 0

    # Reset rejection stats
    global REJECTION_STATS
    REJECTION_STATS = {k: 0 for k in REJECTION_STATS}

    for idx, row in df.iterrows():
        # Extract article data using column map
        article_no = row.get(COLUMN_MAP["article_no"], idx)

        # Get dimensions (using mapped column names)
        cp_l = safe_float(row.get(COLUMN_MAP["cp_length"]))
        cp_w = safe_float(row.get(COLUMN_MAP["cp_width"]))
        cp_h = safe_float(row.get(COLUMN_MAP["cp_height"]))
        cp_wt = safe_float(row.get(COLUMN_MAP["cp_weight"]), 0.1)

        mp_l = safe_float(row.get(COLUMN_MAP["mp_length"]))
        mp_w = safe_float(row.get(COLUMN_MAP["mp_width"]))
        mp_h = safe_float(row.get(COLUMN_MAP["mp_height"]))
        mp_wt = safe_float(row.get(COLUMN_MAP["mp_weight"]))
        mp_qty = safe_float(row.get(COLUMN_MAP["mp_qty"]), 1)

        ews = safe_float(row.get(COLUMN_MAP["ews"]), 0.1)

        # Reference values
        emmie_assq = safe_float(row.get(COLUMN_MAP["emmie_assq"]))
        emmie_speedcells = safe_float(row.get(COLUMN_MAP["emmie_speedcells"]))

        # Run cascade
        # Settings adjusted to match Emmie's approach:
        # - max_pockets=3 (Emmie uses up to 3)
        # - min_stockweeks=0 (no minimum requirement, just fit by dimensions)
        # - allow_flip=True for flat items
        # - air_buffer=0 to match Emmie's ASSQ values
        result = cascade_fit_article(
            cp_length=cp_l, cp_width=cp_w, cp_height=cp_h, cp_weight=cp_wt,
            mp_length=mp_l, mp_width=mp_w, mp_height=mp_h, mp_weight=mp_wt,
            mp_qty=mp_qty,
            ews=ews,
            min_stockweeks=0.0,  # No stockweeks requirement
            max_pockets_per_article=10,  # Effectively unlimited
            allow_rotation=True,
            allow_flip=True,  # Enable flip for flat items
            air_buffer_pct=0.0,  # Test without buffer to match Emmie's values
        )

        total_processed += 1

        # Compare to Emmie's ASSQ
        our_assq = result.assq

        if emmie_assq > 0 and our_assq > 0:
            ratio = our_assq / emmie_assq
            if 0.95 <= ratio <= 1.05:  # Within 5%
                matches += 1
            if 0.8 <= ratio <= 1.2:  # Within 20%
                within_20pct += 1

        results.append({
            "article_no": article_no,
            "emmie_assq": emmie_assq,
            "our_assq": our_assq,
            "our_size": result.recommended_size,
            "our_stockweeks": result.stockweeks,
            "fits": result.fits,
            "rejection": result.rejection_reason,
            "multibox": result.using_multibox,
            "rotation": result.rotation_used,
            "flip": result.flip_used,
        })

    return results, matches, within_20pct, total_processed


def print_summary(results, matches, within_20pct, total):
    """Print validation summary."""

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    # Fit statistics
    fits = sum(1 for r in results if r["fits"])
    rejects = sum(1 for r in results if not r["fits"])

    print(f"\nProcessed: {total} articles")
    print(f"Fits: {fits} ({fits/total*100:.1f}%)")
    print(f"Rejected: {rejects} ({rejects/total*100:.1f}%)")

    # ASSQ comparison
    print(f"\nASSQ Match (within 5%): {matches} ({matches/total*100:.1f}%)")
    print(f"ASSQ Match (within 20%): {within_20pct} ({within_20pct/total*100:.1f}%)")

    # Size distribution
    print("\nRecommended Size Distribution:")
    size_counts = {}
    for r in results:
        size = r["our_size"] or "REJECTED"
        size_counts[size] = size_counts.get(size, 0) + 1
    for size in ["xs", "small", "medium", "large", "REJECTED"]:
        count = size_counts.get(size, 0)
        print(f"  {size.upper():10s}: {count:4d} ({count/total*100:.1f}%)")

    # Rejection breakdown
    print("\nRejection Reasons:")
    summary = get_rejection_summary()
    for reason, count in summary["breakdown"].items():
        if count > 0:
            print(f"  {reason}: {count}")

    # Multibox usage
    multibox_count = sum(1 for r in results if r["multibox"])
    print(f"\nMultibox articles: {multibox_count} ({multibox_count/total*100:.1f}%)")

    # Sample discrepancies
    print("\nSample Discrepancies (our_assq vs emmie_assq):")
    discrepancies = [r for r in results if r["fits"] and r["emmie_assq"] > 0
                     and abs(r["our_assq"] - r["emmie_assq"]) > r["emmie_assq"] * 0.2]
    for d in discrepancies[:10]:
        print(f"  Art {d['article_no']}: ours={d['our_assq']}, emmie={d['emmie_assq']}, "
              f"size={d['our_size']}, multibox={d['multibox']}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    filepath = "/home/flinux/storeganizer-planner/storeganizer-alpha/ref/eligibility_emm.xlsx"

    print("Loading Emmie's eligibility file...")
    df = load_emmie_data(filepath)

    print(f"\nRunning cascading filter on {len(df)} articles...")
    results, matches, within_20pct, total = run_validation(df)

    print_summary(results, matches, within_20pct, total)

    # Save detailed results for analysis
    results_df = pd.DataFrame(results)
    output_path = "/home/flinux/storeganizer-planner/storeganizer-alpha/ref/validation_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path}")
