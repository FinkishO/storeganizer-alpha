"""
Generate sample planogram output for Phil/Lee demonstration.

Uses Emmie's 1,434 suitable articles to show what our service can produce.
"""

import pandas as pd
import sys
import math
sys.path.insert(0, '/home/flinux/storeganizer-planner/storeganizer-alpha/skills')

from skill_assq_calculator import calculate_assq, POCKET_SIZES

# Pocket structure for Large configuration (Emmie's approach)
LARGE_CONFIG = {
    "pocket_dims": POCKET_SIZES["large"],
    "columns_per_row_pattern": [6, 4],  # 6 front + 4 back = 10 columns
    "rows_deep": 2,
    "pockets_per_column": 4,  # 4 vertical pockets
    "total_columns": 10,
    "cells_per_bay": 40,  # 10 cols × 4 pockets
}


def load_suitable_articles(filepath: str) -> pd.DataFrame:
    """Load Emmie's suitable articles."""
    df = pd.read_excel(filepath, header=1)
    suitable = df[df['Speedcells'] > 0].copy()
    print(f"Loaded {len(suitable)} suitable articles")
    return suitable


def calculate_velocity_band(ews: float) -> str:
    """Assign velocity band based on EWS."""
    if ews >= 10:
        return 'A'  # Fast movers
    elif ews >= 2:
        return 'B'  # Medium movers
    else:
        return 'C'  # Slow movers


def allocate_to_cells(articles_df: pd.DataFrame) -> pd.DataFrame:
    """
    Allocate articles to cells using greedy approach.

    Returns DataFrame with cell assignments.
    """
    pocket = POCKET_SIZES["large"]
    config = LARGE_CONFIG

    records = []
    bay = 1
    column = 1
    row = 1  # Row = pocket level (1-4 for Large)
    cells_used = 0

    # Sort by EWS (fast movers first - they go in easy-access positions)
    sorted_df = articles_df.sort_values('Total EWS', ascending=False)

    for idx, article in sorted_df.iterrows():
        # Get dimensions
        cp_l = float(article.get('CP Length', 0) or 0)
        cp_w = float(article.get('CP Width', 0) or 0)
        cp_h = float(article.get('CP Height', 0) or 0)
        cp_wt = float(article.get('CP Max Weight', 0.1) or 0.1)
        ews = float(article.get('Total EWS', 0) or 0)

        if cp_l == 0 or cp_w == 0 or cp_h == 0:
            continue

        # Calculate ASSQ
        result = calculate_assq(
            article_length=cp_l, article_width=cp_w, article_height=cp_h,
            article_weight=cp_wt,
            pocket_length=pocket['depth'], pocket_width=pocket['width'],
            pocket_height=pocket['height'],
            allow_rotation=True,
            allow_flip=True,
            air_buffer_pct=0.0  # Match Emmie's values
        )

        if not result.fits:
            continue

        # Number of pockets needed (based on Emmie's Speedcells column)
        speedcells = int(article.get('Speedcells', 1) or 1)

        for _ in range(speedcells):
            cell_label = f"B{bay:02d}-C{column:02d}-R{row:02d}"

            records.append({
                'Article Number': article.get('Article Number', ''),
                'Article Name': article.get('Article Name', ''),
                'Bay': bay,
                'Column': column,
                'Row': row,
                'Cell Label': cell_label,
                'CP Length (mm)': int(cp_l),
                'CP Width (mm)': int(cp_w),
                'CP Height (mm)': int(cp_h),
                'Weight (kg)': round(cp_wt, 2),
                'ASSQ': result.assq,
                'EWS': round(ews, 2),
                'Stockweeks': round(result.assq / ews, 1) if ews > 0 else 999,
                'Velocity Band': calculate_velocity_band(ews),
                'Pocket Size': 'Large',
            })

            # Move to next cell
            row += 1
            if row > config['pockets_per_column']:
                row = 1
                column += 1
                if column > config['total_columns']:
                    column = 1
                    bay += 1

            cells_used += 1

    print(f"Allocated {len(records)} cells across {bay} bays")
    return pd.DataFrame(records)


def generate_summary_stats(planogram_df: pd.DataFrame) -> dict:
    """Generate summary statistics for the planogram."""
    return {
        'Total Articles': planogram_df['Article Number'].nunique(),
        'Total Cells Used': len(planogram_df),
        'Total Bays': planogram_df['Bay'].max(),
        'Velocity A (Fast)': len(planogram_df[planogram_df['Velocity Band'] == 'A']),
        'Velocity B (Medium)': len(planogram_df[planogram_df['Velocity Band'] == 'B']),
        'Velocity C (Slow)': len(planogram_df[planogram_df['Velocity Band'] == 'C']),
        'Avg Stockweeks': round(planogram_df['Stockweeks'].mean(), 1),
        'Avg ASSQ': round(planogram_df['ASSQ'].mean(), 1),
    }


def main():
    filepath = "/home/flinux/storeganizer-planner/storeganizer-alpha/ref/eligibility_emm.xlsx"
    output_path = "/home/flinux/storeganizer-planner/storeganizer-alpha/ref/sample_planogram_output.xlsx"

    print("=" * 60)
    print("STOREGANIZER PLANOGRAM GENERATOR")
    print("Sample output for Phil/Lee demonstration")
    print("=" * 60)

    # Load data
    articles = load_suitable_articles(filepath)

    # Allocate to cells
    planogram = allocate_to_cells(articles)

    # Generate summary
    stats = generate_summary_stats(planogram)
    print("\nSUMMARY:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Export to Excel
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Main planogram sheet
        planogram.to_excel(writer, sheet_name='Planogram', index=False)

        # Summary sheet
        summary_df = pd.DataFrame([stats])
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # Velocity breakdown sheet
        velocity_summary = planogram.groupby('Velocity Band').agg({
            'Article Number': 'nunique',
            'ASSQ': 'mean',
            'EWS': 'mean',
            'Stockweeks': 'mean',
        }).round(2)
        velocity_summary.columns = ['Articles', 'Avg ASSQ', 'Avg EWS', 'Avg Stockweeks']
        velocity_summary.to_excel(writer, sheet_name='Velocity Breakdown')

    print(f"\nOutput saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
