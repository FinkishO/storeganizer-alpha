"""
Article Library - Remember article dimensions across uploads
"""

import sqlite3
from pathlib import Path
from typing import Optional, Dict
import pandas as pd


class ArticleLibrary:
    def __init__(self, db_path: str = ".tmp/article_library.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create articles table if it doesn't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                article_number TEXT PRIMARY KEY,
                article_name TEXT,
                width_mm REAL,
                depth_mm REAL,
                height_mm REAL,
                weight_kg REAL,
                last_seen TEXT,
                source_file TEXT
            )
        """)

        conn.commit()
        conn.close()

    def import_from_excel(self, excel_path: str, sheet_name: str = 'rpt'):
        """
        Import articles from Excel (like cp_input1.xlsx)

        Maps IKEA format:
        - Article Number → article_number
        - CP Width (mm) → width_mm
        - CP Length (mm) → depth_mm
        - CP Height (mm) → height_mm
        - CP Weight (kg) → weight_kg
        """
        print(f"📚 Importing articles from {excel_path}...")

        # Read Excel with header detection (skip metadata rows)
        df = pd.read_excel(excel_path, sheet_name=sheet_name, header=1)

        # Find columns (handle newlines and variations)
        article_col = next((c for c in df.columns if 'Article Number' in str(c)), None)
        name_col = next((c for c in df.columns if 'Article Name' in str(c)), None)
        width_col = next((c for c in df.columns if 'CP Width' in str(c)), None)
        length_col = next((c for c in df.columns if 'CP Length' in str(c)), None)
        height_col = next((c for c in df.columns if 'CP Height' in str(c)), None)
        weight_col = next((c for c in df.columns if 'CP Weight' in str(c)), None)

        if not article_col:
            raise ValueError("Could not find 'Article Number' column")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        imported = 0
        for _, row in df.iterrows():
            article_number = row.get(article_col)
            if pd.isna(article_number):
                continue

            cursor.execute("""
                INSERT OR REPLACE INTO articles
                (article_number, article_name, width_mm, depth_mm, height_mm, weight_kg, last_seen, source_file)
                VALUES (?, ?, ?, ?, ?, ?, datetime('now'), ?)
            """, (
                str(int(article_number)) if isinstance(article_number, float) else str(article_number),
                row.get(name_col) if name_col else None,
                row.get(width_col) if width_col else None,
                row.get(length_col) if length_col else None,
                row.get(height_col) if height_col else None,
                row.get(weight_col) if weight_col else None,
                str(excel_path)
            ))
            imported += 1

        conn.commit()
        conn.close()

        print(f"✅ Imported {imported} articles to library")
        return imported

    def lookup(self, article_number: str) -> Optional[Dict]:
        """Look up article dimensions from library"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT article_number, article_name, width_mm, depth_mm, height_mm, weight_kg, last_seen, source_file
            FROM articles
            WHERE article_number = ?
        """, (str(article_number),))

        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                "article_number": row[0],
                "article_name": row[1],
                "width_mm": row[2],
                "depth_mm": row[3],
                "height_mm": row[4],
                "weight_kg": row[5],
                "last_seen": row[6],
                "source_file": row[7],
                "from_library": True
            }
        return None

    def enrich_dataframe(self, df: pd.DataFrame, article_col: str) -> tuple[pd.DataFrame, dict]:
        """
        Enrich dataframe with dimensions from library

        Returns:
            (enriched_df, stats)

        Stats:
            - total: Total articles in uploaded file
            - found_in_library: Articles with dimensions auto-filled
            - missing_from_library: Articles NOT in library (need manual input)
            - already_complete: Articles that already had dimensions
        """
        stats = {
            "total": len(df),
            "found_in_library": 0,
            "missing_from_library": 0,
            "already_complete": 0,
            "enriched_fields": []
        }

        # Track which fields we enriched
        dimension_cols = ["width_mm", "depth_mm", "height_mm", "weight_kg"]

        for idx, row in df.iterrows():
            article_number = row.get(article_col)
            if pd.isna(article_number):
                continue

            # Check if article already has dimensions
            has_dims = all(
                col in df.columns and pd.notna(row.get(col)) and row.get(col) > 0
                for col in dimension_cols
            )

            if has_dims:
                stats["already_complete"] += 1
                continue

            # Look up in library
            library_data = self.lookup(str(article_number))

            if library_data:
                # Auto-fill missing dimensions
                for field in dimension_cols:
                    if field not in df.columns or pd.isna(row.get(field)) or row.get(field) == 0:
                        df.at[idx, field] = library_data[field]
                        if field not in stats["enriched_fields"]:
                            stats["enriched_fields"].append(field)

                # Add metadata
                df.at[idx, "_from_library"] = True
                df.at[idx, "_library_source"] = library_data["source_file"]

                stats["found_in_library"] += 1
            else:
                stats["missing_from_library"] += 1

        return df, stats

    def get_stats(self) -> dict:
        """Get library statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM articles")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM articles WHERE width_mm IS NOT NULL AND width_mm > 0")
        with_dimensions = cursor.fetchone()[0]

        conn.close()

        return {
            "total_articles": total,
            "with_dimensions": with_dimensions,
            "completeness": (with_dimensions / total * 100) if total > 0 else 0
        }


# Helper function for quick import
def build_library_from_ikea(excel_path: str = "ref/cp_input1.xlsx"):
    """Quick helper to build library from IKEA master file"""
    library = ArticleLibrary()
    library.import_from_excel(excel_path, sheet_name='rpt')
    stats = library.get_stats()
    print(f"\n📊 Library Stats:")
    print(f"   Total articles: {stats['total_articles']:,}")
    print(f"   With dimensions: {stats['with_dimensions']:,}")
    print(f"   Completeness: {stats['completeness']:.1f}%")
    return library


if __name__ == "__main__":
    # Build library from IKEA master file
    build_library_from_ikea()
