"""
Data ingestion and harmonization for Storeganizer.

Handles:
- CSV/Excel file upload with smart sheet/header detection
- Multi-row header Excel files (IKEA format)
- Column name harmonization (aliases)
- Type coercion and validation
- Optional column defaults
"""

from typing import Union, IO, List
import pandas as pd

from config import storeganizer as config


def _find_best_sheet(xl: pd.ExcelFile, keywords: List[str]) -> str:
    """
    Auto-detect best sheet by scoring keyword matches.

    Scans first 8 rows of each sheet, counts keyword occurrences.
    Returns sheet with highest score.
    """
    best_sheet = xl.sheet_names[0]
    best_score = -1

    for name in xl.sheet_names:
        try:
            preview = xl.parse(name, nrows=8, header=None, dtype=str)
        except Exception:
            continue

        score = 0
        for _, row in preview.iterrows():
            row_lower = [str(x).lower() for x in row.tolist()]
            score = max(score, sum(1 for cell in row_lower if any(k in cell for k in keywords)))

        if score > best_score:
            best_score = score
            best_sheet = name

    return best_sheet


def _load_table_with_best_header(xl: pd.ExcelFile, sheet_name: str) -> pd.DataFrame:
    """
    Auto-detect header row by scoring keyword density.

    Scans first 15 rows, finds row with most data-related keywords.
    Common in IKEA exports where rows 0-1 are metadata, row 2 is headers.
    """
    raw = xl.parse(sheet_name, header=None, dtype=str)

    # Keywords that typically appear in column headers (incl. EMM format)
    target_keys = [
        "article", "sku", "description", "name", "width", "length", "height",
        "depth", "weight", "fcst", "forecast", "demand", "stock", "weeks",
        "pa", "hfb", "price", "kg", "mm", "cm",
        "selected solution", "multipack", "speedcell", "consumer pack",  # EMM format
    ]

    best_idx, best_score = 0, -1
    for i in range(min(15, len(raw))):
        row_lower = [str(x).lower() for x in raw.iloc[i].tolist()]
        score = sum(1 for cell in row_lower if any(k in cell for k in target_keys))
        if score > best_score:
            best_score = score
            best_idx = i

    # Parse with detected header row
    df = xl.parse(sheet_name, header=best_idx)

    # Clean column names (remove newlines, extra spaces)
    df.columns = [str(c).replace("\n", " ").replace("\r", " ").strip() for c in df.columns]

    return df


def normalize_column_name(col: str) -> str:
    """
    Normalize column name for comparison.

    Args:
        col: Column name

    Returns:
        Lowercase, stripped version with underscores/hyphens treated as spaces.
    """
    collapsed = col.replace("_", " ").replace("-", " ").strip().lower()
    return " ".join(collapsed.split())


def harmonize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Harmonize column names using configured aliases.

    Maps various input column names to standardized names.
    Handles both generic formats and IKEA-specific formats.

    Only maps the FIRST occurrence of each standard column to avoid duplicates.

    Args:
        df: Input DataFrame with varied column names

    Returns:
        DataFrame with standardized column names
    """
    df = df.copy()
    rename_map = {}

    # Normalize existing columns once for quick lookup (preserve original names)
    normalized_existing = {normalize_column_name(c): c for c in df.columns}
    used_source_cols = set()

    # Walk aliases in the order they are declared for each standard column
    for std_col, aliases in config.COLUMN_ALIASES.items():
        for alias in aliases:
            norm_alias = normalize_column_name(alias)
            if norm_alias in normalized_existing:
                source_col = normalized_existing[norm_alias]
                if source_col not in used_source_cols:
                    rename_map[source_col] = std_col
                    used_source_cols.add(source_col)
                break

    return df.rename(columns=rename_map)


def add_optional_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add optional columns with default values if missing.
    Also fills required numeric fields with safe defaults when configured.

    Args:
        df: DataFrame potentially missing optional columns

    Returns:
        DataFrame with optional columns added
    """
    df = df.copy()

    # Fill required numeric fields when a sensible default exists
    if hasattr(config, "REQUIRED_DEFAULTS"):
        for col, default_value in config.REQUIRED_DEFAULTS.items():
            if col not in df.columns:
                df[col] = default_value

    for col, default_value in config.OPTIONAL_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default_value

    return df


def validate_required_columns(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Check if all required columns are present.

    Args:
        df: DataFrame to validate

    Returns:
        Tuple of (is_valid, list_of_missing_columns)
    """
    missing = [col for col in config.REQUIRED_COLUMNS if col not in df.columns]
    return len(missing) == 0, missing


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce columns to expected data types.

    - sku_code, description: string, stripped
    - Numeric columns: float, NaN -> 0

    Args:
        df: DataFrame with harmonized columns

    Returns:
        DataFrame with coerced types
    """
    df = df.copy()

    # String columns
    if "sku_code" in df.columns:
        df["sku_code"] = df["sku_code"].astype(str).str.strip()
    if "description" in df.columns:
        df["description"] = df["description"].astype(str).str.strip()

    # Numeric columns
    numeric_cols = [
        "width_mm", "depth_mm", "height_mm",
        "weight_kg", "weekly_demand", "stock_weeks"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


def load_inventory_file(
    file_like: Union[str, IO],
    detect_format: bool = True,
) -> pd.DataFrame:
    """
    Load inventory file (CSV or Excel) and prepare for processing.

    Smart detection for:
    - CSV vs Excel format
    - Best sheet (if multi-sheet Excel)
    - Header row location (scans first 15 rows)
    - Column name variations (IKEA, generic warehouse formats)

    Full pipeline:
    1. Read file (auto-detect format, sheet, header row)
    2. Harmonize column names (handle aliases)
    3. Add optional columns with defaults
    4. Coerce types
    5. Validate required columns

    Args:
        file_like: File path or file-like object
        detect_format: If True, auto-detect CSV vs Excel

    Returns:
        Processed DataFrame ready for eligibility filtering

    Raises:
        ValueError: If file cannot be read or missing required columns
    """
    # Step 1: Read file with smart detection
    df = None

    # Try CSV first (simpler, faster)
    try:
        df = pd.read_csv(file_like)
    except Exception:
        pass

    # If CSV failed, try Excel with smart detection
    if df is None:
        try:
            if hasattr(file_like, "seek"):
                file_like.seek(0)

            xl = pd.ExcelFile(file_like)

            # Smart sheet detection - include EMM format keywords
            keywords = [
                "article", "sku", "cp width", "cp length", "planning fcst",
                "description", "demand", "stock",
                "selected solution", "multipack", "speedcell",  # EMM format
            ]
            sheet_name = _find_best_sheet(xl, keywords)

            # Smart header detection
            df = _load_table_with_best_header(xl, sheet_name)

        except Exception as exc:
            raise ValueError(f"Could not read file as CSV or Excel: {exc}")

    if df is None or df.empty:
        raise ValueError("File is empty or could not be parsed")

    # Step 2-5: Process pipeline
    df = harmonize_columns(df)
    df = add_optional_columns(df)
    df = coerce_types(df)

    # Validate
    is_valid, missing = validate_required_columns(df)
    if not is_valid:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def get_column_status(df: pd.DataFrame) -> dict:
    """
    Get status of required and optional columns.

    Useful for UI feedback showing which columns are present/missing.

    Args:
        df: DataFrame to check

    Returns:
        Dict with column status information:
        {
            "required_present": [...],
            "required_missing": [...],
            "optional_present": [...],
            "optional_missing": [...],
        }
    """
    present_cols = set(df.columns)
    required_set = set(config.REQUIRED_COLUMNS)
    optional_set = set(config.OPTIONAL_COLUMNS.keys()) if hasattr(config, 'OPTIONAL_COLUMNS') else set()

    return {
        "required_present": sorted(list(required_set & present_cols)),
        "required_missing": sorted(list(required_set - present_cols)),
        "optional_present": sorted(list(optional_set & present_cols)),
        "optional_missing": sorted(list(optional_set - present_cols)),
    }


def detect_priority_whitelist(file_like, main_df: pd.DataFrame = None) -> dict:
    """
    Detect if file has a priority/whitelist indicator.

    Checks for:
    1. Priority sheets: requested_in_cell, priority, whitelist, country_request, etc.
    2. Priority columns: country request, hybrid, requested, priority, in_cell, etc.

    Returns:
        {
            "detected": bool,
            "source": "sheet" | "column" | None,
            "source_name": str (sheet or column name),
            "article_col": str (column containing article numbers),
            "article_numbers": list[str] (the whitelist),
            "count": int,
            "description": str (human-readable description),
        }
    """
    result = {
        "detected": False,
        "source": None,
        "source_name": None,
        "article_col": None,
        "article_numbers": [],
        "count": 0,
        "description": None,
    }

    # === CHECK FOR PRIORITY SHEETS ===
    priority_sheet_keywords = [
        "requested", "in_cell", "incell", "priority", "whitelist",
        "country_request", "country request", "selected", "target"
    ]

    try:
        if hasattr(file_like, "seek"):
            file_like.seek(0)
        xl = pd.ExcelFile(file_like)

        for sheet_name in xl.sheet_names:
            sheet_lower = sheet_name.lower().replace(" ", "_")
            if any(kw in sheet_lower for kw in priority_sheet_keywords):
                # Found a priority sheet - extract article numbers
                sheet_df = xl.parse(sheet_name)

                # Find article number column (prefer specific matches over generic)
                article_col = None
                # Priority order: article number > sku > item > product > pa
                priority_patterns = [
                    ["article number", "article_number", "articlenumber"],
                    ["sku_code", "sku code", "skucode", "sku"],
                    ["item number", "item_number", "itemnumber", "item code"],
                    ["product number", "product_number", "product code"],
                ]
                for patterns in priority_patterns:
                    for col in sheet_df.columns:
                        col_lower = str(col).lower().strip()
                        if any(p in col_lower for p in patterns):
                            if "name" not in col_lower:
                                article_col = col
                                break
                    if article_col:
                        break

                if article_col and len(sheet_df) > 0:
                    articles = sheet_df[article_col].dropna().astype(str).str.strip().tolist()
                    articles = [a for a in articles if a and a.lower() not in ["nan", "none", ""]]

                    if articles:
                        result["detected"] = True
                        result["source"] = "sheet"
                        result["source_name"] = sheet_name
                        result["article_col"] = article_col
                        result["article_numbers"] = articles
                        result["count"] = len(articles)
                        result["description"] = f"{len(articles)} articles from '{sheet_name}' sheet"
                        return result
    except Exception:
        pass  # Not an Excel file or can't read sheets

    # === CHECK FOR PRIORITY COLUMNS IN MAIN DATA ===
    if main_df is not None and len(main_df) > 0:
        # Helper to find article column
        def find_article_col(df):
            priority_patterns = [
                ["article number", "article_number", "articlenumber"],
                ["sku_code", "sku code", "skucode", "sku"],
                ["pa"],  # IKEA PA column (last resort)
            ]
            for patterns in priority_patterns:
                for col in df.columns:
                    col_lower = str(col).lower().strip()
                    if any(p in col_lower for p in patterns):
                        if "name" not in col_lower:
                            return col
            return None

        # === EMM FORMAT: "Selected Solution" = "Multipack" (highest priority) ===
        for col in main_df.columns:
            col_lower = str(col).lower().replace("_", " ")
            if "selected solution" in col_lower:
                col_vals = main_df[col].astype(str).str.strip()
                # EMM logic: "Multipack" = Storeganizer-suitable articles
                multipack_mask = col_vals.str.lower() == "multipack"

                if multipack_mask.any():
                    article_col = find_article_col(main_df)
                    if article_col:
                        articles = main_df.loc[multipack_mask, article_col].dropna().astype(str).str.strip().tolist()
                        articles = [a for a in articles if a and a.lower() not in ["nan", "none", ""]]

                        if articles:
                            result["detected"] = True
                            result["source"] = "column"
                            result["source_name"] = col
                            result["article_col"] = article_col
                            result["article_numbers"] = articles
                            result["count"] = len(articles)
                            result["description"] = f"{len(articles)} Multipack articles (EMM selection)"
                            return result

        # === BOOLEAN PRIORITY COLUMNS (Y/Yes/1/True values) ===
        priority_col_keywords = [
            "country request", "country_request", "countryrequest",
            "hybrid", "requested", "in_cell", "incell", "in cell",
            "priority", "selected", "target", "whitelist"
        ]

        for col in main_df.columns:
            col_lower = str(col).lower().replace("_", " ")

            for kw in priority_col_keywords:
                if kw in col_lower:
                    # Found a priority column - check for Y/1/True values
                    col_vals = main_df[col].astype(str).str.strip().str.upper()

                    # Check for boolean-like values
                    positive_mask = col_vals.isin(["Y", "YES", "1", "TRUE", "X", "REQUESTED"])

                    if positive_mask.any():
                        article_col = find_article_col(main_df)

                        if article_col:
                            articles = main_df.loc[positive_mask, article_col].dropna().astype(str).str.strip().tolist()
                            articles = [a for a in articles if a and a.lower() not in ["nan", "none", ""]]

                            if articles:
                                result["detected"] = True
                                result["source"] = "column"
                                result["source_name"] = col
                                result["article_col"] = article_col
                                result["article_numbers"] = articles
                                result["count"] = len(articles)
                                result["description"] = f"{len(articles)} articles where '{col}' = Y"
                                return result

    return result
