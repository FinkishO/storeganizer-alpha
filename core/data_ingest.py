"""
Data ingestion and harmonization for Storeganizer.

Handles:
- CSV/Excel file upload
- Column name harmonization (aliases)
- Type coercion and validation
- Optional column defaults
"""

from typing import Union, IO
import pandas as pd

from config import storeganizer as config


def normalize_column_name(col: str) -> str:
    """
    Normalize column name for comparison.

    Args:
        col: Column name

    Returns:
        Lowercase, stripped, underscored version
    """
    return col.lower().strip().replace(" ", "_")


def harmonize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Harmonize column names using configured aliases.

    Maps various input column names to standardized names.
    For example: "article", "SKU", "item_code" all map to "sku_code"

    Args:
        df: Input DataFrame with varied column names

    Returns:
        DataFrame with standardized column names
    """
    df = df.copy()
    rename_map = {}

    for std_col, aliases in config.COLUMN_ALIASES.items():
        for alias in aliases:
            for original_col in df.columns:
                if normalize_column_name(original_col) == normalize_column_name(alias):
                    rename_map[original_col] = std_col
                    break

    return df.rename(columns=rename_map)


def add_optional_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add optional columns with default values if missing.

    Args:
        df: DataFrame potentially missing optional columns

    Returns:
        DataFrame with optional columns added
    """
    df = df.copy()

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

    Full pipeline:
    1. Read file (auto-detect CSV vs Excel)
    2. Harmonize column names
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
    # Read file
    if detect_format:
        try:
            df = pd.read_csv(file_like)
        except Exception:
            try:
                if hasattr(file_like, "seek"):
                    file_like.seek(0)
                df = pd.read_excel(file_like)
            except Exception as exc:
                raise ValueError(f"Could not read file as CSV or Excel: {exc}")
    else:
        # Try both formats
        try:
            df = pd.read_csv(file_like)
        except Exception:
            if hasattr(file_like, "seek"):
                file_like.seek(0)
            df = pd.read_excel(file_like)

    # Process
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
    optional_set = set(config.OPTIONAL_COLUMNS.keys())

    return {
        "required_present": sorted(list(required_set & present_cols)),
        "required_missing": sorted(list(required_set - present_cols)),
        "optional_present": sorted(list(optional_set & present_cols)),
        "optional_missing": sorted(list(optional_set - present_cols)),
    }
