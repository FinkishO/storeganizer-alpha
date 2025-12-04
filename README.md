# Storeganizer Planning Tool (Alpha)

AI-powered warehouse planning tool specifically designed for Storeganizer high-density pocket storage systems.

Developed for partnership with Storeganizer (Dimitri Saerens) - December 2025.

---

## Overview

This tool automates the warehouse planning process for Storeganizer installations:

1. **Upload** - Import inventory data (CSV/Excel)
2. **Filter** - Apply eligibility rules (dimensions, weight, velocity)
3. **Optimize** - Automatically allocate SKUs to bays/columns/rows
4. **Visualize** - See planogram with weight distribution
5. **Export** - Download implementation-ready allocation files

### Key Features

- ✅ **Intelligent SKU filtering** based on Storeganizer pocket specifications
- ✅ **Automatic family grouping** to keep related products together
- ✅ **Weight distribution** with overweight column flagging
- ✅ **Velocity-based placement** (A/B/C band prioritization)
- ✅ **Lena AI assistant** for sizing questions and planning guidance
- 🚧 **3D visualization integration** (planned for Belgium workshop)

---

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Tool

```bash
streamlit run app.py
```

Open browser to `http://localhost:8501`

---

## Architecture

This tool uses a modular architecture designed for easy customization and future scaling:

```
storeganizer-alpha/
├── core/                    # Core business logic
│   ├── eligibility.py       # SKU filtering (dimensions, weight, velocity)
│   ├── allocation.py        # Bay/column/row mapping algorithm
│   └── data_ingest.py      # CSV/Excel parsing and harmonization
│
├── rag/                     # RAG-powered chat assistant
│   ├── rag_service.py       # Lena conversation logic
│   ├── rag_store.py         # Vector database
│   └── ingest_ref.py        # Reference material ingestion
│
├── visualization/           # Planogram rendering
│   ├── planogram_2d.py      # Current 2D Plotly charts
│   └── viewer_3d.py         # 3D integration (placeholder)
│
├── config/
│   └── storeganizer.py      # All Storeganizer-specific settings
│
├── ref/                     # Reference materials
│   └── storeganizer_buying_guide.pdf
│
└── app.py                   # Streamlit UI
```

### Why This Structure?

**Modularity**: Core logic separated from UI - easy to test and modify
**Configurability**: All Storeganizer specs in one config file
**Scalability**: Easy to create Speedcell/VASS versions by swapping config
**Professional**: Clean architecture for potential handover to Storeganizer

---

## Configuration

All Storeganizer-specific settings are in `config/storeganizer.py`:

- Pocket dimensions (450mm W x 500mm D x 450mm H)
- Weight limits (20kg per pocket, 100kg per column)
- Default bay/column structure (8 columns x 5 rows)
- Eligibility rules and thresholds
- Reference materials and visualization URLs

To customize for a different Storeganizer configuration, edit this file.

---

## Core Modules

### Eligibility Filtering (`core/eligibility.py`)

Determines which SKUs fit in Storeganizer pockets:

```python
from core.eligibility import apply_all_filters

filtered_df, rejected_count, reasons = apply_all_filters(
    df,
    max_width=450,
    max_height=450,
    velocity_band="A",  # A, B, C, or "All"
    remove_fragile=True,
)
```

### Allocation (`core/allocation.py`)

Maps SKUs to specific bay/column/row locations:

```python
from core.allocation import compute_planning_metrics, build_layout

# Calculate planning metrics
df = compute_planning_metrics(df, units_per_column=30, max_weight_per_column_kg=100)

# Generate layout
df, blocks, columns_summary = build_layout(
    df,
    bays=5,
    columns_per_bay=8,
    rows_per_column=5,
    units_per_column=30,
    max_weight_per_column_kg=100,
)
```

### Data Ingestion (`core/data_ingest.py`)

Handles flexible CSV/Excel uploads with column alias support:

```python
from core.data_ingest import load_inventory_file

df = load_inventory_file("inventory.csv")
# Automatically maps "article" → "sku_code", "forecast" → "weekly_demand", etc.
```

---

## 3D Visualization Integration (Planned)

Dimitri provided visualization URLs for different Storeganizer configurations:

- Large: https://offer.storeganizer.com/visualisations/02171ec5-...
- Medium: https://offer.storeganizer.com/visualisations/668f6dcd-...
- Small: https://offer.storeganizer.com/visualisations/ffcafcad-...
- XS: https://offer.storeganizer.com/visualisations/a40e1a1f-...

**Next step**: During Belgium workshop, determine how to:
1. Pass planogram data to visualization platform (URL params or API)
2. Integrate customization configurator
3. Link to their internal systems

Placeholder implementation in `visualization/viewer_3d.py`.

---

## Development Roadmap

### Alpha (Current - Pre-Belgium)
- ✅ Core eligibility and allocation logic
- ✅ Lena RAG assistant (Storeganizer-focused)
- ✅ 2D planogram visualization
- ✅ Modular architecture

### Beta (Post-Belgium Workshop)
- [ ] 3D visualization integration
- [ ] Refined eligibility rules based on Dimitri's input
- [ ] Customization configurator integration
- [ ] Real Storeganizer test data validation

### V1 (Production)
- [ ] Multi-warehouse support
- [ ] Saved configurations ("user library")
- [ ] Walking distance optimization
- [ ] Internal UI integration (if applicable)
- [ ] Deployment to production environment

---

## Partnership Context

**Meeting Date**: December 2025
**Contact**: Dimitri Saerens (Storeganizer, 20 years logistics experience)
**Workshop**: Planned in Belgium next week
**Business Model**: TBD - options include:
- Storeganizer purchases tool outright
- License per deployment
- Co-developed product integration
- SaaS model with Storeganizer white-label

**Current Planning Service Cost**: ~€5000 per client project
**Value Proposition**: Automate planning process, add as client-facing service

---

## Technical Notes

**Dependencies**: See `requirements.txt`
- Streamlit (UI framework)
- Pandas (data processing)
- Plotly (2D visualization)
- ChromaDB (RAG vector store)

**Data Format**: Expects inventory files with:
- Required: sku_code, description, width_mm, depth_mm, height_mm, weight_kg, weekly_demand, stock_weeks
- Optional: Various aliases accepted (e.g., "article", "forecast", "wos")

**Performance**: Tested up to 500 SKUs, handles larger datasets with degraded UI responsiveness (solvable with optimization)

---

## Future Extensions

### For Speedcell/VASS Versions
1. Copy this folder → `speedcell-alpha`
2. Replace `config/storeganizer.py` → `config/speedcell.py`
3. Update `ref/` folder with Speedcell materials
4. Adjust `app.py` imports to use new config
5. Core logic remains identical

### For Enterprise Multi-Client
- Add client-specific config management
- Per-warehouse saved presets
- API layer for integration with other systems
- Audit logging for regulatory compliance

---

## Contact

**Developer**: Fred Olsen
**Project**: Storeganizer Planning Tool Alpha
**Status**: Pre-Belgium workshop demo
**Date**: December 2025

For questions during development, reach out via agreed channels.

---

## License

Proprietary - Partnership agreement pending with Storeganizer.
