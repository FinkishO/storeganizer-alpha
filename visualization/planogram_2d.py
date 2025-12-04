"""Visualization utilities for rendering the planogram in Streamlit using Plotly."""
from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from core.allocation import CellBlock


def _normalize_mode(mode: str) -> str:
    m = (mode or "").strip().lower()
    if "velocity" in m:
        return "velocity_band"
    return "sku"


def generate_color_map(blocks: List[CellBlock], mode: str = "sku") -> Dict[str, str]:
    """Generate color mapping keyed by sku_code or velocity_band."""
    normalized = _normalize_mode(mode)
    if normalized == "velocity_band":
        return {"A": "#FF6B6B", "B": "#FFA94D", "C": "#69DB7C"}
    if normalized == "hfb":
        palette = px.colors.qualitative.Set3
        keys = sorted({getattr(b, "description", "")[:2] for b in blocks})
        return {k: palette[i % len(palette)] for i, k in enumerate(keys)}

    palette = px.colors.qualitative.Set3
    keys = sorted({b.sku_code for b in blocks})
    return {k: palette[i % len(palette)] for i, k in enumerate(keys)}


def render_planogram(blocks: List[CellBlock], columns_summary: pd.DataFrame, color_mode: str = "By SKU"):
    """Render planogram into active Streamlit app."""
    if not blocks:
        st.info("No blocks to render.")
        return

    rows_per_column = int(columns_summary["rows_per_column"].iloc[0]) if "rows_per_column" in columns_summary else 10
    bays = sorted(set(b.bay for b in blocks))

    cmap = generate_color_map(blocks, mode=color_mode)
    normalized_mode = _normalize_mode(color_mode)

    def short_sku(s: str) -> str:
        s = s or ""
        if len(s) <= 6:
            return s
        return f"{s[:3]}***{s[-2:]}"

    def wrap_desc(d: str, width: int = 14, max_lines: int = 3) -> str:
        words = (d or "").split()
        lines = []
        current = ""
        for w in words:
            if len(current) + len(w) + 1 <= width:
                current = (current + " " + w).strip()
            else:
                lines.append(current)
                current = w
            if len(lines) >= max_lines:
                break
        if current and len(lines) < max_lines:
            lines.append(current)
        text = "\n".join(lines)
        if len(lines) == max_lines and " ".join(words).strip() != " ".join(lines):
            text += "\n…"
        return text

    for bay in bays:
        bay_blocks = [b for b in blocks if b.bay == bay]
        bay_blocks = sorted(bay_blocks, key=lambda b: (b.column_index, b.row_start))

        fig = go.Figure()

        for b in bay_blocks:
            col_label = f"C{b.column_index+1}"
            if normalized_mode == "hfb":
                key = (b.description or "")[:2]
            else:
                key = b.sku_code if normalized_mode == "sku" else b.velocity_band
            color = cmap.get(key, "#888888")
            border_color = "#c0392b" if b.overweight_flag else "#1b1b1b"

            label = f"{short_sku(b.sku_code)}\n{wrap_desc(b.description, width=14, max_lines=3)}"

            fig.add_bar(
                x=[col_label],
                y=[b.row_span],
                base=[b.row_start],
                marker=dict(color=color, line=dict(color=border_color, width=0.8)),
                name=b.sku_code,
                hovertext=(
                    f"{b.sku_code}: {b.description}<br>"
                    f"Units: {b.units_in_block}<br>"
                    f"Velocity: {b.velocity_band} (#{b.velocity_rank})<br>"
                    f"Column weight: {b.column_weight_kg:.2f} kg"
                ),
                hoverinfo="text",
                text=label,
                textposition="inside",
                insidetextanchor="middle",
                showlegend=False,
            )

        bay_cols_summary = columns_summary[columns_summary["bay"] == bay]
        for _, r in bay_cols_summary.iterrows():
            col_label = f"C{int(r['column_index'])+1}"
            weight_text = f"{r['total_weight_kg']} kg"
            flag = " ⚠" if bool(r.get("overweight_flag", False)) else ""
            fig.add_annotation(
                x=col_label,
                y=-0.4,
                text=f"{weight_text}{flag}",
                showarrow=False,
                yanchor="top",
                font=dict(size=11, color="#333"),
                bgcolor="#eef1f5",
                bordercolor="#b0b8c1",
                borderwidth=1,
                borderpad=4,
            )

        fig.update_layout(
            title_text=f"Bay {bay}",
            barmode="overlay",
            xaxis=dict(title="Columns", ticks="outside", tickfont=dict(size=12)),
            yaxis=dict(
                title="Rows (top to bottom)",
                range=[rows_per_column, 0],
                autorange=False,
                dtick=1,
                showgrid=True,
                gridcolor="#e9ecef",
            ),
            bargap=0.05,
            margin=dict(l=70, r=40, t=70, b=110),
            height=480,
            plot_bgcolor="#f8f9fb",
            paper_bgcolor="#ffffff",
        )

        st.plotly_chart(fig, use_container_width=True)

    if normalized_mode == "sku":
        unique_keys = sorted({b.sku_code for b in blocks})
        legend_items = [(k, cmap.get(k, "#777777")) for k in unique_keys]
        st.markdown("**Legend (SKU colors)**")
        cols = st.columns(4)
        for i, (sku, colhex) in enumerate(legend_items):
            c = cols[i % len(cols)]
            c.markdown(
                f"<div style='display:flex;align-items:center;gap:6px'>"
                f"<div style='width:16px;height:12px;background:{colhex};border:1px solid #999;'></div>"
                f"<span>{sku}</span></div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown("**Velocity Band Colors**")
        st.markdown("- A: coral  \n- B: amber  \n- C: green")


if __name__ == "__main__":
    print("visualization module loaded")
