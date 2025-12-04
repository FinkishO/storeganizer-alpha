"""Ingest reference materials from ./ref into the local RAG store.

Usage:
    python -m ingest_ref

Creates/updates rag_store.db with chunks from Markdown, PDF, and Excel files.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from pypdf import PdfReader

from .rag_store import Document, add_documents, get_connection


REF_DIR = Path(__file__).parent.parent / "ref"
DB_PATH = Path(__file__).parent.parent / "rag_store.db"


def chunk_text(text: str, max_chars: int = 1500, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks for better recall."""
    words = text.split()
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for word in words:
        current.append(word)
        current_len += len(word) + 1
        if current_len >= max_chars:
            chunks.append(" ".join(current))
            # overlap words
            overlap_words = current[-overlap // 5 :] if overlap > 0 else []
            current = list(overlap_words)
            current_len = sum(len(w) + 1 for w in current)
    if current:
        chunks.append(" ".join(current))
    return chunks


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n\n".join(pages)


def read_excel_as_text(path: Path) -> str:
    xl = pd.ExcelFile(path)
    sections = []
    for name in xl.sheet_names:
        df = xl.parse(name)
        if df.empty:
            sections.append(f"Sheet: {name}\n<empty>")
        else:
            preview = df.head(50)
            sections.append(f"Sheet: {name}\n{preview.to_csv(index=False)}")
    return "\n\n".join(sections)

def iter_documents() -> Iterable[Document]:
    for path in REF_DIR.rglob("*"):
        if path.is_dir():
            continue
        ext = path.suffix.lower()
        if ext in {".md", ".txt", ".py", ".json"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for i, chunk in enumerate(chunk_text(text)):
                yield Document(
                    source=f"{path.relative_to(REF_DIR)}#chunk{i}",
                    content=chunk,
                    metadata={"path": str(path), "type": ext.lstrip(".")},
                )
        elif ext == ".pdf":
            text = read_pdf(path)
            for i, chunk in enumerate(chunk_text(text)):
                yield Document(
                    source=f"{path.relative_to(REF_DIR)}#chunk{i}",
                    content=chunk,
                    metadata={"path": str(path), "type": "pdf"},
                )
        elif ext in {".xlsx", ".xls"}:
            if ext == ".xls":
                # Skip legacy xls if xlrd not present
                continue
            text = read_excel_as_text(path)
            for i, chunk in enumerate(chunk_text(text)):
                yield Document(
                    source=f"{path.relative_to(REF_DIR)}#chunk{i}",
                    content=chunk,
                    metadata={"path": str(path), "type": "excel"},
                )


def ingest():
    conn = get_connection(DB_PATH)
    docs = list(iter_documents())
    add_documents(conn, docs)
    print(f"Ingested {len(docs)} chunks into {DB_PATH}")


if __name__ == "__main__":
    ingest()
