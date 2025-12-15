"""
Storeganizer-specific configuration.

All Storeganizer product specifications, dimensions, and business rules.
"""

# ===========================
# PRODUCT SPECIFICATIONS
# ===========================

# Default pocket/cell dimensions (mm) - align with Medium preset
# FROM JAN'S STOREGANIZER SPECS: Medium = 300 x 450 x 300 mm (W×D×H)
DEFAULT_POCKET_WIDTH = 300
DEFAULT_POCKET_DEPTH = 450
DEFAULT_POCKET_HEIGHT = 300

# Weight limits (kg)
DEFAULT_POCKET_WEIGHT_LIMIT = 20.0
DEFAULT_COLUMN_WEIGHT_LIMIT = 100.0  # Column overweight flag threshold

# Structure configuration (Medium preset)
# TERMINOLOGY from buying guide:
#   - "rows deep" = depth rows front-to-back (M=3, L=2, XS=4)
#   - "columns per row" = columns in each depth row [6,5,4] for Medium
#   - "pockets per column" = vertical stacking (M=6, L=4, XS=13)
DEFAULT_COLUMNS_PER_BAY = 6   # Medium front row columns
DEFAULT_ROWS_DEEP = 3         # Medium = 3 rows front-to-back
DEFAULT_POCKETS_PER_COLUMN = 6  # Medium = 6 pockets stacked vertically
DEFAULT_ROWS_PER_COLUMN = 6   # DEPRECATED: use DEFAULT_POCKETS_PER_COLUMN
DEFAULT_UNITS_PER_COLUMN = 30

# ===========================
# STANDARD CONFIGURATIONS
# ===========================

STANDARD_CONFIGS = {
    # FROM OFFICIAL STOREGANIZER BUYING GUIDE (storeganizer_buying_guide.pdf page 2)
    # Max weight per Pocket: 20kg (ALL sizes)
    # Max weight per column: 100kg
    "xs": {
        "name": "Extra Small",
        "description": "Compact pockets for small items",
        "pocket_width": 260,   # From Jan's Storeganizer specs
        "pocket_depth": 300,   # From Jan's Storeganizer specs
        "pocket_height": 150,  # From Jan's Storeganizer specs
        "pocket_weight_limit": 20.0,  # Same for all sizes
        "pockets_per_column": 13,  # Vertical stacking
        "rows_deep": 4,  # 4 rows front-to-back
        "columns_per_row_pattern": [9, 7, 7, 7],  # Columns in each depth row (front→back)
        "columns_per_bay": 9,  # Front row only (for backward compat)
        "total_columns": 30,  # sum([9,7,7,7]) = 30
        "rows_per_column": 13,  # DEPRECATED: use pockets_per_column
        "cells_per_bay": 390,  # 30 columns × 13 pockets = 390
        "typical_bays": "Small installations",
        "image": "ref/sg_xs.png",
        "price_per_bay_eur": None,
    },
    "small": {
        "name": "Small",
        "description": "Standard small item storage",
        "pocket_width": 300,   # From buying guide
        "pocket_depth": 300,   # From buying guide
        "pocket_height": 300,  # From buying guide
        "pocket_weight_limit": 20.0,
        "pockets_per_column": 6,  # Vertical stacking
        "rows_deep": 3,  # 3 rows front-to-back
        "columns_per_row_pattern": [9, 7, 7],  # Columns in each depth row (front→back)
        "columns_per_bay": 9,  # Front row only (for backward compat)
        "total_columns": 23,  # sum([9,7,7]) = 23
        "rows_per_column": 6,  # DEPRECATED: use pockets_per_column
        "cells_per_bay": 138,  # 23 columns × 6 pockets = 138
        "typical_bays": "4-8 bays",
        "image": "ref/sg_s.png",
        "price_per_bay_eur": None,
    },
    "medium": {
        "name": "Medium",
        "description": "Most popular - balanced capacity",
        "pocket_width": 300,   # From Jan's Storeganizer specs
        "pocket_depth": 450,   # From Jan's Storeganizer specs
        "pocket_height": 300,  # From Jan's Storeganizer specs
        "pocket_weight_limit": 20.0,
        "pockets_per_column": 6,  # Vertical stacking
        "rows_deep": 3,  # 3 rows front-to-back
        "columns_per_row_pattern": [6, 5, 4],  # Columns in each depth row (front→back)
        "columns_per_bay": 6,  # Front row only (for backward compat)
        "total_columns": 15,  # sum([6,5,4]) = 15
        "rows_per_column": 6,  # DEPRECATED: use pockets_per_column
        "cells_per_bay": 90,  # 15 columns × 6 pockets = 90
        "typical_bays": "6-12 bays",
        "image": "ref/sg_m.png",
        "price_per_bay_eur": None,
        "recommended": True,
    },
    "large": {
        "name": "Large",
        "description": "Larger items, deeper pockets",
        "pocket_width": 500,   # From Jan's Storeganizer specs
        "pocket_depth": 450,   # From Jan's Storeganizer specs
        "pocket_height": 450,  # From Jan's Storeganizer specs
        "pocket_weight_limit": 20.0,  # Same 20kg limit (NOT 25!)
        "pockets_per_column": 4,  # Vertical stacking
        "rows_deep": 2,  # 2 rows front-to-back
        "columns_per_row_pattern": [6, 4],  # Columns in each depth row (front→back)
        "columns_per_bay": 6,  # Front row only (for backward compat)
        "total_columns": 10,  # sum([6,4]) = 10
        "rows_per_column": 4,  # DEPRECATED: use pockets_per_column
        "cells_per_bay": 40,  # 10 columns × 4 pockets = 40
        "typical_bays": "8-20 bays",
        "image": "ref/sg_l.png",
        "price_per_bay_eur": None,
    },
}

DEFAULT_CONFIG_SIZE = "medium"

# ===========================
# ELIGIBILITY RULES
# ===========================

# Dimension tolerances
ALLOW_SQUEEZE_PACKAGING = False  # Allow 10% width overage for soft goods
SQUEEZE_WIDTH_MULTIPLIER = 1.10

# Fragile item handling
FRAGILE_KEYWORDS = ["glass", "porcelain", "fragile", "ceramic", "china"]
DEFAULT_REMOVE_FRAGILE = False

# Velocity band thresholds (percentiles)
VELOCITY_BAND_A_PERCENTILE = 80  # Top 20% of demand
VELOCITY_BAND_B_PERCENTILE = 50  # Top 50% of demand

# ===========================
# BRANDING & UI
# ===========================

SOLUTION_NAME = "Storeganizer"
PRODUCT_DESCRIPTION = "High-density pocket storage system with flexible configurations"

LENA_PERSONA = (
    "Lena is a Storeganizer logistics specialist who helps with sizing, "
    "eligibility checking, bay/column calculations, and warehouse planning. "
    "Ask her about Storeganizer specifications, SKU fit rules, or how to prep your inventory files."
)

LENA_KNOWLEDGE_AREAS = [
    "Storeganizer buying guide (pocket sizing, weight limits, configurations)",
    "Inventory CSV/Excel format requirements",
    "Eligibility filtering (dimensions, weight, fragile items)",
    "Bay and column capacity calculations",
]

SUGGESTED_PROMPTS = [
    "What are Storeganizer pocket dimensions?",
    "Check if my SKUs fit Storeganizer sizing",
    "How many bays for 500 SKUs at 30 units/column?",
    "What are the weight limits per pocket/column?",
]

# ===========================
# DATA REQUIREMENTS
# ===========================

REQUIRED_COLUMNS = [
    "sku_code",
    "description",
    "width_mm",
    "depth_mm",
    "height_mm",
    "weight_kg",
    "weekly_demand",
    # stock_weeks is optional (defaults to 4.0)
]

OPTIONAL_COLUMNS = {
    "stock_weeks": "Weeks of stock coverage (defaults to 4 if not provided)"
}

OPTIONAL_DEFAULTS = {
    "stock_weeks": 4.0  # Default to 4 weeks if not in file
}

# Column aliases for flexible CSV parsing
# Supports: generic warehouse formats, IKEA exports, Storeganizer/Speedcell formats
COLUMN_ALIASES = {
    "sku_code": [
        "article number", "article_number", "article_no", "article", "sku",
        "sku_code", "item_code", "product_code", "artikelnummer", "artnr", "pa"
    ],
    "description": [
        "article name", "description", "item description", "sku description",
        "desc", "product_name", "product description", "name", "item_name"
    ],
    "width_mm": [
        "cp width (mm)", "cp width", "width_mm", "width (mm)", "width",
        "w (mm)", "w", "ul width (mm)", "ul width", "breedte", "largeur"
    ],
    "depth_mm": [
        "cp length (mm)", "cp length", "depth_mm", "depth (mm)", "depth",
        "length_mm", "length (mm)", "length", "cp depth (mm)", "cp depth",
        "ul length (mm)", "ul length", "d (mm)", "l (mm)", "d", "l",
        "diepte", "profondeur"
    ],
    "height_mm": [
        "cp height (mm)", "cp height", "height_mm", "height (mm)", "height",
        "h (mm)", "h", "ul height (mm)", "ul height", "hoogte", "hauteur"
    ],
    "weight_kg": [
        "cp weight (kg)", "cp weight", "weight_kg", "weight (kg)", "weight",
        "kg", "ul weight (kg)", "ul weight", "net weight (kg)", "gewicht", "poids",
        "wt"
    ],
    "weekly_demand": [
        "planning fcst", "total store ff fcst", "store ff fcst", "fcst (total)",
        "weekly_demand", "weekly demand", "forecast", "fcst", "demand",
        "demand per week", "expected weekly sales", "ews", "weekly_fcst",
        "verkoop per week", "fus"
    ],
    "stock_weeks": [
        "stock_weeks", "stockweeks", "stock weeks", "weeks_of_stock", "weeks of stock",
        "weeks_of_supply", "stock weeks", "weeks", "wos", "sos", "coverage",
        "voorraad weken", "semaines de stock", "total wis"
    ],
}

# Defaults for required numeric columns when the source file omits them entirely
REQUIRED_DEFAULTS = {
    "width_mm": 0.0,
    "depth_mm": 0.0,
    "height_mm": 0.0,
    "weight_kg": 0.0,
    "weekly_demand": 1.0,
}

# ===========================
# REFERENCE MATERIALS
# ===========================

REFERENCE_FILES = [
    "storeganizer_buying_guide.pdf",
    # Add more as provided by Dimitri
]

# Dimitri's visualization platform URLs (for future 3D integration)
VISUALIZATION_URLS = {
    "large": "https://offer.storeganizer.com/visualisations/02171ec5-72ab-453c-b7f5-79f95f87fd59",
    "medium": "https://offer.storeganizer.com/visualisations/668f6dcd-b76a-41b5-b07e-4d3fec63e243",
    "small": "https://offer.storeganizer.com/visualisations/ffcafcad-31f9-45d9-83e8-c98483111b9c",
    "xs": "https://offer.storeganizer.com/visualisations/a40e1a1f-3270-4e35-9282-4b01c4668d57",
}

# ===========================
# BUSINESS RULES
# ===========================

# DEPRECATED: Old EWS-based filtering (wrong approach - doesn't account for pocket capacity)
DEFAULT_FORECAST_THRESHOLD = 4.0  # DEPRECATED - kept for backward compat only

# Stockweeks-based eligibility (CORRECT approach)
# Stockweeks = ASSQ / EWS = how many weeks of stock fit in one pocket
# - If stockweeks < min → item sells too fast (constant replenishment needed)
# - If stockweeks > max → item is too slow (wasting pocket space)
MIN_STOCKWEEKS = 1.0          # Minimum weeks coverage (IKEA default)
MAX_STOCKWEEKS = 26.0         # Maximum weeks (6 months, for 3PL use cases)
USE_STOCKWEEKS_FILTER = True  # Set False to use old EWS filter

# User-configurable filter options (defaults for UI)
DEFAULT_REMOVE_FRAGILE = False      # Whether to exclude fragile items
DEFAULT_ALLOW_ROTATION = True       # Allow L/W swap in ASSQ calculation
DEFAULT_ALLOW_FLIP = True           # Allow height swap in ASSQ calculation
DEFAULT_AIR_BUFFER_PCT = 0.25       # 25% air buffer (Jan's rule)
DEFAULT_MAX_POCKETS_PER_ARTICLE = 2 # Prefer 1 Large over 2 Medium

# Golden zone placement (ergonomic row placement)
GOLDEN_ZONE_ROWS = [2, 3]  # Middle rows for high-velocity items

# ===========================
# FUTURE INTEGRATION HOOKS
# ===========================

# 3D Visualization integration settings (placeholder)
ENABLE_3D_VIEWER = False  # Set to True when integration is ready
VISUALIZATION_API_ENDPOINT = None  # To be determined with Dimitri

# Customization options API (for future feature)
CUSTOMIZATION_API_ENDPOINT = None

# Internal UI modernization (discussed with Dimitri)
LEGACY_UI_INTEGRATION = False
