"""
Planogram visualization as grid layout (matches reference format).

Renders cells as rectangles with visible labels, not stacked bars.
"""
from typing import Dict, List
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from core.allocation import CellBlock


def generate_sku_color_map(blocks: List[CellBlock]) -> Dict[str, str]:
    """
    Generate consistent color mapping for SKUs.

    Each unique SKU gets a distinct color from a professional palette.
    """
    # Use a clean palette (avoiding neon/garish colors)
    palette = [
        "#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3",
        "#fdb462", "#b3de69", "#fccde5", "#d9d9d9", "#bc80bd",
        "#ccebc5", "#ffed6f", "#a6cee3", "#b2df8a", "#fb9a99",
        "#fdbf6f", "#cab2d6", "#ffff99", "#1f78b4", "#33a02c",
    ]

    unique_skus = sorted(set(b.sku_code for b in blocks))
    return {sku: palette[i % len(palette)] for i, sku in enumerate(unique_skus)}


def render_planogram(blocks: List[CellBlock], columns_summary: pd.DataFrame, color_mode: str = "By SKU"):
    """
    Render planogram as grid layout with cell labels.

    Format matches reference planogram:
    - Grid cells with borders
    - Cell label (B01-C01-R01)
    - SKU code
    - Description (truncated)
    - Unit count
    - Color by SKU
    """
    if not blocks:
        st.info("No blocks to render.")
        return

    rows_per_column = int(columns_summary["rows_per_column"].iloc[0]) if "rows_per_column" in columns_summary else 10
    bays = sorted(set(b.bay for b in blocks))

    color_map = generate_sku_color_map(blocks)

    for bay in bays:
        bay_blocks = [b for b in blocks if b.bay == bay]

        # Get unique columns for this bay
        columns_in_bay = sorted(set(b.column_index for b in bay_blocks))

        fig = go.Figure()

        # Draw grid cells
        for block in bay_blocks:
            col_idx = block.column_index
            row_idx = block.row_start

            # Cell coordinates (column on x-axis, row on y-axis inverted)
            x0 = col_idx
            x1 = col_idx + 1
            y0 = rows_per_column - row_idx - 1  # Invert y-axis (row 0 at top)
            y1 = y0 + 1

            # Color by SKU
            fill_color = color_map.get(block.sku_code, "#cccccc")
            border_color = "#c0392b" if block.overweight_flag else "#2c3e50"
            border_width = 3 if block.overweight_flag else 1

            # Cell label: B01-C01-R01 format
            cell_label = f"B{block.bay:02d}-C{col_idx+1:02d}-R{row_idx+1:02d}"

            # Truncate description
            desc_short = (block.description[:16] + "...") if len(block.description) > 16 else block.description

            # Cell text content
            cell_text = (
                f"<b>{cell_label}</b><br>"
                f"{block.sku_code}<br>"
                f"{desc_short}<br>"
                f"Unit:{block.units_in_block}"
            )

            # Draw rectangle
            fig.add_shape(
                type="rect",
                x0=x0, x1=x1, y0=y0, y1=y1,
                line=dict(color=border_color, width=border_width),
                fillcolor=fill_color,
            )

            # Add text annotation
            fig.add_annotation(
                x=(x0 + x1) / 2,
                y=(y0 + y1) / 2,
                text=cell_text,
                showarrow=False,
                font=dict(size=9, color="#000000"),
                align="center",
                xanchor="center",
                yanchor="middle",
            )

        # Add column weight annotations at bottom
        bay_cols_summary = columns_summary[columns_summary["bay"] == bay]
        for _, col_row in bay_cols_summary.iterrows():
            col_idx = int(col_row["column_index"])
            weight_kg = col_row["total_weight_kg"]
            is_overweight = bool(col_row.get("overweight_flag", False))

            weight_text = f"{int(weight_kg)} kg"
            if is_overweight:
                weight_text = f"⚠ {weight_text}"

            # Position below grid
            fig.add_annotation(
                x=col_idx + 0.5,
                y=-0.5,
                text=f"<b>{weight_text}</b>",
                showarrow=False,
                font=dict(size=11, color="#c0392b" if is_overweight else "#2c3e50"),
                bgcolor="#fff3cd" if is_overweight else "#e9ecef",
                bordercolor="#c0392b" if is_overweight else "#6c757d",
                borderwidth=2 if is_overweight else 1,
                borderpad=4,
            )

        # Layout configuration
        fig.update_xaxes(
            range=[-0.5, max(columns_in_bay) + 1.5],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            title="",
        )

        fig.update_yaxes(
            range=[-1, rows_per_column],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            title="",
            scaleanchor="x",
            scaleratio=1,
        )

        fig.update_layout(
            title=dict(text=f"<b>Bay {bay}</b>", font=dict(size=18)),
            height=600,
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            margin=dict(l=20, r=20, t=60, b=80),
            showlegend=False,
        )

        st.plotly_chart(fig, use_container_width=True)

    # Legend: SKU colors
    st.markdown("### Legend: SKU Colors")
    legend_items = sorted(color_map.items())

    # Display in columns
    num_cols = 4
    cols = st.columns(num_cols)
    for i, (sku, color) in enumerate(legend_items):
        col = cols[i % num_cols]
        col.markdown(
            f"<div style='display:flex;align-items:center;gap:6px;margin:4px 0;'>"
            f"<div style='width:20px;height:16px;background:{color};border:1px solid #333;'></div>"
            f"<span style='font-size:12px;'>{sku}</span></div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    print("planogram_2d module loaded")
