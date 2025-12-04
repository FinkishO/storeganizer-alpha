"""Lightweight SQLite FTS5 RAG store for local knowledge snippets.

This is intentionally minimal: documents are stored as text chunks with optional
metadata (JSON). Querying uses FTS5 MATCH for fast keyword retrieval.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, List, Optional


DEFAULT_DB_PATH = Path(__file__).parent / "rag_store.db"


@dataclass
class Document:
    source: str
    content: str
    metadata: Optional[dict] = None


def get_connection(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS documents
        USING fts5(source, content, metadata);
        """
    )
    return conn


def add_document(conn: sqlite3.Connection, doc: Document) -> None:
    metadata_json = json.dumps(doc.metadata or {})
    conn.execute(
        "INSERT INTO documents (source, content, metadata) VALUES (?, ?, ?)",
        (doc.source, doc.content, metadata_json),
    )
    conn.commit()


def add_documents(conn: sqlite3.Connection, docs: Iterable[Document]) -> None:
    rows = []
    for doc in docs:
        rows.append((doc.source, doc.content, json.dumps(doc.metadata or {})))
    conn.executemany("INSERT INTO documents (source, content, metadata) VALUES (?, ?, ?)", rows)
    conn.commit()


def reset_store(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS documents;")
    conn.commit()
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS documents
        USING fts5(source, content, metadata);
        """
    )
    conn.commit()


def search(conn: sqlite3.Connection, query: str, limit: int = 5) -> List[Document]:
    """Search documents using BM25 ranking for relevance.

    Uses SQLite FTS5's built-in BM25 algorithm to rank results by relevance
    rather than just keyword matching. This dramatically improves search quality
    for large knowledge bases.
    """
    def _safe_match(q: str) -> str:
        """Build safe FTS5 MATCH query from user input."""
        tokens = re.findall(r"[A-Za-z0-9_]+", q.lower())
        if not tokens:
            tokens = ["planogram"]
        # Use first 8 tokens with OR for broader matching
        tokens = tokens[:8]
        return " OR ".join(tokens)

    safe_query = _safe_match(query)

    # Use bm25() ranking function - lower scores = better matches
    # ORDER BY bm25(documents) returns most relevant results first
    cur = conn.execute(
        """
        SELECT source, content, metadata, bm25(documents) as rank
        FROM documents
        WHERE documents MATCH ?
        ORDER BY rank
        LIMIT ?
        """,
        (safe_query, limit),
    )
    results: List[Document] = []
    for source, content, metadata_json, _rank in cur.fetchall():
        try:
            meta = json.loads(metadata_json) if metadata_json else {}
        except json.JSONDecodeError:
            meta = {}
        results.append(Document(source=source, content=content, metadata=meta))
    return results


if __name__ == "__main__":
    conn = get_connection()
    add_document(conn, Document(source="example", content="Hello world", metadata={"tag": "demo"}))
    print(search(conn, "hello"))
