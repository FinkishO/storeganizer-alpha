"""
Storeganizer-specific configuration.

All Storeganizer product specifications, dimensions, and business rules.
"""

# ===========================
# PRODUCT SPECIFICATIONS
# ===========================

# Default pocket/cell dimensions (mm)
DEFAULT_POCKET_WIDTH = 450
DEFAULT_POCKET_DEPTH = 500
DEFAULT_POCKET_HEIGHT = 450

# Weight limits (kg)
DEFAULT_POCKET_WEIGHT_LIMIT = 20.0
DEFAULT_COLUMN_WEIGHT_LIMIT = 100.0  # For largest configuration

# Structure configuration
DEFAULT_COLUMNS_PER_BAY = 8
DEFAULT_ROWS_PER_COLUMN = 5
DEFAULT_UNITS_PER_COLUMN = 30

# ===========================
# STANDARD CONFIGURATIONS
# ===========================

STANDARD_CONFIGS = {
    "xs": {
        "name": "Extra Small",
        "description": "Compact setup for smaller operations",
        "pocket_width": 300,
        "pocket_depth": 400,
        "pocket_height": 350,
        "pocket_weight_limit": 15.0,
        "columns_per_bay": 10,
        "rows_per_column": 6,
        "cells_per_bay": 60,  # 10 columns × 6 rows
        "typical_bays": "2-4 bays",
        "image": "ref/sg_xs.png",
        "price_per_bay_eur": None,  # Placeholder for future
    },
    "small": {
        "name": "Small",
        "description": "Standard configuration for growing businesses",
        "pocket_width": 400,
        "pocket_depth": 450,
        "pocket_height": 400,
        "pocket_weight_limit": 18.0,
        "columns_per_bay": 9,
        "rows_per_column": 5,
        "cells_per_bay": 45,  # 9 columns × 5 rows
        "typical_bays": "4-8 bays",
        "image": "ref/sg_s.png",
        "price_per_bay_eur": None,
    },
    "medium": {
        "name": "Medium",
        "description": "Most popular - balanced capacity and flexibility",
        "pocket_width": 450,
        "pocket_depth": 500,
        "pocket_height": 450,
        "pocket_weight_limit": 20.0,
        "columns_per_bay": 8,
        "rows_per_column": 5,
        "cells_per_bay": 40,  # 8 columns × 5 rows
        "typical_bays": "6-12 bays",
        "image": "ref/sg_m.png",
        "price_per_bay_eur": None,
        "recommended": True,  # Flag for UI
    },
    "large": {
        "name": "Large",
        "description": "High-capacity for warehouses with extensive inventory",
        "pocket_width": 500,
        "pocket_depth": 550,
        "pocket_height": 500,
        "pocket_weight_limit": 25.0,
        "columns_per_bay": 7,
        "rows_per_column": 4,
        "cells_per_bay": 28,  # 7 columns × 4 rows
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
        "sku", "article", "article_number", "article number", "item_code",
        "product_code", "sku_code", "artikelnummer", "pa"
    ],
    "description": [
        "desc", "description", "name", "product_name", "article_name",
        "article name", "product description", "item_name"
    ],
    "width_mm": [
        "width", "w", "width_mm", "width (mm)", "cp width", "cp width (mm)",
        "width mm", "breedte", "largeur"
    ],
    "depth_mm": [
        "depth", "d", "length", "l", "depth_mm", "length_mm", "depth (mm)",
        "length (mm)", "cp length", "cp length (mm)", "cp depth", "cp depth (mm)",
        "diepte", "profondeur"
    ],
    "height_mm": [
        "height", "h", "height_mm", "height (mm)", "cp height", "cp height (mm)",
        "hoogte", "hauteur"
    ],
    "weight_kg": [
        "weight", "wt", "weight_kg", "weight (kg)", "cp weight", "cp weight (kg)",
        "weight kg", "gewicht", "poids"
    ],
    "weekly_demand": [
        "demand", "forecast", "fcst", "ews", "weekly_fcst", "weekly_demand",
        "weekly demand", "store ff fcst", "total store ff fcst", "fcst (total)",
        "planning fcst", "demand per week", "verkoop per week"
    ],
    "stock_weeks": [
        "weeks", "wos", "weeks_of_stock", "weeks of stock", "coverage", "stock_weeks",
        "stock weeks", "voorraad weken", "semaines de stock"
    ],
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

# Default forecast threshold (weekly demand ceiling for filtering)
DEFAULT_FORECAST_THRESHOLD = 4.0

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
