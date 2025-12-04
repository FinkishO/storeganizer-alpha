"""
RAG-powered Lena chat assistant for Storeganizer.

Provides conversational help about:
- Storeganizer specifications and sizing
- Eligibility rules and filtering
- Bay/column capacity calculations
- Inventory file format requirements
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import List, Tuple
from functools import lru_cache

from rag.ingest_ref import ingest
from rag.rag_store import Document, get_connection, search
from config import storeganizer as config


REF_DIR = Path(__file__).parent.parent / "ref"

# Storeganizer-specific quick facts
STOREGANIZER_SPECS = {
    "sizing": f"Storeganizer pockets: up to ~{config.DEFAULT_POCKET_WIDTH} mm W x {config.DEFAULT_POCKET_DEPTH} mm D x {config.DEFAULT_POCKET_HEIGHT} mm H; weight ~{config.DEFAULT_POCKET_WEIGHT_LIMIT} kg per pocket; ~{config.DEFAULT_COLUMN_WEIGHT_LIMIT} kg per column (largest config).",
}

CAPABILITY_TEXT = (
    f"Lena (Storeganizer specialist): I can help with pocket sizing/eligibility, bay/column math, "
    f"weight flags, and inventory CSV/Excel formatting. Try asking: "
    f"'{config.SUGGESTED_PROMPTS[0]}' or '{config.SUGGESTED_PROMPTS[1]}'"
)

HELP_KEYWORDS = ("what can you help", "what can you do", "help me with", "capabilities")


@lru_cache(maxsize=1)
def list_ref_sources(limit: int = 30) -> List[str]:
    """List available reference materials."""
    paths: List[str] = []
    if not REF_DIR.exists():
        return paths

    for p in sorted(REF_DIR.rglob("*")):
        if p.is_dir():
            continue
        rel = str(p.relative_to(REF_DIR))
        paths.append(rel)
        if len(paths) >= limit:
            break
    return paths


def ensure_ingested():
    """Ensure RAG database is populated from ref/ folder."""
    db_path = Path(__file__).parent.parent / "rag_store.db"
    if not db_path.exists() or db_path.stat().st_size == 0:
        try:
            # Note: ingest_ref.py may need path adjustments
            ingest()
        except Exception:
            return


def query(message: str, limit: int = 4) -> List[Document]:
    """
    Query RAG store for relevant documents.

    Args:
        message: User question
        limit: Max number of results

    Returns:
        List of relevant Document objects
    """
    ensure_ingested()
    conn = get_connection()
    q = message.strip()[:300]
    return search(conn, q if q else "storeganizer specifications", limit=limit)


def answer(user_message: str, context: dict | None = None, **_: dict) -> Tuple[str, List[Document]]:
    """
    Answer user questions about Storeganizer using RAG and heuristics.

    Args:
        user_message: User's question
        context: Optional context dict with current session state

    Returns:
        Tuple of (answer_text, relevant_documents)
    """
    msg = (user_message or "").lower()
    ctx = context or {}

    # Help/capabilities query
    if any(k in msg for k in HELP_KEYWORDS) or msg.strip() in {"help", "help?"}:
        return CAPABILITY_TEXT, []

    # Planogram summary
    if "summarize planogram" in msg and ctx:
        preset = ctx.get("preset", "this configuration")
        bay_count = ctx.get("bay_count")
        overweight = ctx.get("overweight_count", 0)
        reject_count = ctx.get("reject_count", 0)
        sku_count = ctx.get("sku_count")

        parts = [f"Preset: {preset}."]
        if sku_count is not None:
            parts.append(f"SKUs planned: {sku_count}.")
        if bay_count is not None:
            parts.append(f"Bay estimate: {bay_count}.")
        parts.append(f"Overweight columns: {overweight}.")
        if reject_count:
            parts.append(f"Rejected rows: {reject_count}.")

        summary = " ".join(parts)
        return f"Lena: {summary} Ask me where to trim weight or increase bays.", []

    # Weight/overweight guidance
    if "overweight" in msg or "too heavy" in msg or "weight limit" in msg:
        col_limit = ctx.get("max_weight_per_column", config.DEFAULT_COLUMN_WEIGHT_LIMIT)
        tip = (
            f"Keep columns under ~{col_limit} kg; reduce units/column for heavy SKUs "
            f"and spread weight across bays. Per-pocket guideline: ~{config.DEFAULT_POCKET_WEIGHT_LIMIT} kg each."
        )
        return (
            f"Lena: To avoid overweight columns, 1) reduce units per column for heavy SKUs, "
            f"2) split heavy SKUs across multiple columns/bays, 3) keep heavy items in lower rows, "
            f"4) raise bay count if every column is near limit. {tip}",
            [],
        )

    # Storeganizer sizing FAQ
    if ("storeganizer" in msg or "pocket" in msg or "sizing" in msg) and any(
        kw in msg for kw in ["dimension", "size", "weight", "spec"]
    ):
        return f"Lena: {STOREGANIZER_SPECS['sizing']}", []

    # Inventory format question
    if "format" in msg or "column" in msg or ("csv" in msg or "excel" in msg):
        req_cols = ", ".join(config.REQUIRED_COLUMNS)
        return (
            f"Lena: Required columns for inventory upload: {req_cols}. "
            f"I can accept various aliases (e.g., 'article' for 'sku_code', 'forecast' for 'weekly_demand').",
            [],
        )

    # Reference materials query
    if "what guide" in msg or "which guide" in msg or "references" in msg or "knowledge" in msg:
        sources = list_ref_sources()
        if sources:
            return (
                "Lena: I can pull from these local references:\n- " + "\n- ".join(sources),
                [],
            )
        else:
            return "Lena: No reference materials loaded yet.", []

    # Bay count heuristic
    bay_match = re.search(r"(\d+)\s*(sku|item)", msg)
    units_match = re.search(r"(\d+)\s*unit", msg)
    if ("how many bay" in msg or "bays" in msg or "bays do i need" in msg) and (bay_match or ctx):
        sku_count = int(bay_match.group(1)) if bay_match else ctx.get("sku_count")
        cols = ctx.get("columns_per_bay", config.DEFAULT_COLUMNS_PER_BAY)
        rows = ctx.get("rows_per_column", config.DEFAULT_ROWS_PER_COLUMN)
        cells_per_bay = cols * rows if cols and rows else (config.DEFAULT_COLUMNS_PER_BAY * config.DEFAULT_ROWS_PER_COLUMN)
        bays_needed = math.ceil(sku_count / cells_per_bay) if sku_count else None

        per_bay_note = f"Assuming {cols} columns x {rows} rows ≈ {cells_per_bay} pockets/bay."
        capacity_note = ""
        if units_match:
            units_per_column = int(units_match.group(1))
            capacity_note = f" With {units_per_column} units/column, each bay stages ~{units_per_column * cols} units."

        if bays_needed:
            return (
                f"Lena: For ~{sku_count} SKUs, estimate ~{bays_needed} bay(s). "
                f"{per_bay_note} {capacity_note}",
                [],
            )
        else:
            return (
                f"Lena: Bay estimate needs SKU count. {per_bay_note} "
                "Tell me the SKU count or upload a file to refine.",
                [],
            )

    # General RAG query
    docs = query(user_message, limit=3)
    if not docs:
        return (
            "Lena: I didn't find a relevant snippet in the Storeganizer buying guide. "
            "Try a specific ask, e.g., 'weight limit per pocket' or 'required CSV columns'.",
            [],
        )

    # Return RAG results
    lines = ["Lena: pulling from Storeganizer references:"]
    for d in docs[:3]:
        preview = d.content.strip().replace("\n", " ")
        if len(preview) > 180:
            preview = preview[:180] + "..."
        lines.append(f"- {d.source}: {preview}")

    return "\n".join(lines), docs
