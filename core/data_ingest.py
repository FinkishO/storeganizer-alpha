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

    # Keywords that typically appear in column headers
    target_keys = [
        "article", "sku", "description", "name", "width", "length", "height",
        "depth", "weight", "fcst", "forecast", "demand", "stock", "weeks",
        "pa", "hfb", "price", "kg", "mm", "cm"
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

            # Smart sheet detection
            keywords = [
                "article", "sku", "cp width", "cp length", "planning fcst",
                "description", "demand", "stock"
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
