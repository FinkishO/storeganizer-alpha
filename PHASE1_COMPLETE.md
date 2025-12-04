# Phase 1 Complete: Storeganizer Alpha Architecture

**Status**: ✅ Foundation built, ready for coding agent refactoring
**Time**: ~45 minutes
**Next**: Spawn coding agent for app.py refactoring

---

## What's Been Built

### 📁 Clean Modular Architecture

```
storeganizer-alpha/
├── core/                           ✅ Core business logic extracted
│   ├── eligibility.py              # SKU filtering (dimensions, weight, velocity)
│   ├── allocation.py               # Bay/column/row mapping algorithm
│   └── data_ingest.py             # CSV/Excel parsing with column aliases
│
├── rag/                            ✅ Lena chat system (Storeganizer-only)
│   ├── rag_service.py              # Conversation logic
│   ├── rag_store.py                # Vector database
│   └── ingest_ref.py               # Reference ingestion
│
├── visualization/                  ✅ Rendering modules
│   ├── planogram_2d.py             # Current Plotly charts
│   └── viewer_3d.py                # 3D integration placeholder
│
├── config/                         ✅ Configuration system
│   └── storeganizer.py             # All Storeganizer specs, rules, defaults
│
├── ref/                            ✅ Reference materials
│   ├── storeganizer_buying_guide.pdf
│   └── storeganizer_soroksar*.jpg
│
├── source/                         ✅ Assets
│   └── lena2_img.png
│
├── README.md                       ✅ Comprehensive documentation
├── .gitignore                      ✅ Standard Python ignore
├── requirements.txt                ✅ Dependencies copied
└── REFACTORING_INSTRUCTIONS.md     ✅ Coding agent instructions
```

---

## Progress Summary

### ✅ Completed (Phase 1)

1. **Folder structure** - Modular architecture designed for scaling
2. **Core modules** - Eligibility, allocation, and data ingestion logic extracted and cleaned
3. **Config system** - All Storeganizer specs centralized in `config/storeganizer.py`
4. **RAG system** - Lena chat focused on Storeganizer only
5. **Visualization** - 2D copied, 3D placeholder with integration notes
6. **Documentation** - README, refactoring instructions, .gitignore

### 🔄 Next (Phase 2 - Coding Agent)

**Main task**: Refactor original `app.py` (3000 lines) → new `app.py` (~1500-2000 lines)

**Key changes**:
- Remove all multi-solution code (Speedcell, VASS, Hyllevagn, Tornado)
- Replace inline logic with core module calls
- Use config system instead of hardcoded values
- Simplify wizard to Storeganizer-only flow
- Maintain all working functionality

**Estimated time**: 3-4 hours (coding agent work)

---

## Architecture Benefits

### Why This Structure Matters

**For Alpha (next 24h)**:
- Clean, professional code Dimitri can review
- Easy to test individual components
- Lena knows only Storeganizer (no confusing competitor references)

**For Future Refactoring**:
- **Speedcell version**: Copy folder, swap `config/storeganizer.py` → `config/speedcell.py`
- **Multi-client**: Each client gets their own config with custom rules
- **3D integration**: Just plug into `visualization/viewer_3d.py` after Belgium workshop
- **Handover**: If Storeganizer buys this, modular architecture shows technical maturity

---

## Config System Explained

All Storeganizer-specific values are in `config/storeganizer.py`:

```python
# Product specs
DEFAULT_POCKET_WIDTH = 450
DEFAULT_POCKET_DEPTH = 500
DEFAULT_POCKET_HEIGHT = 450
DEFAULT_POCKET_WEIGHT_LIMIT = 20.0
DEFAULT_COLUMN_WEIGHT_LIMIT = 100.0

# Structure
DEFAULT_COLUMNS_PER_BAY = 8
DEFAULT_ROWS_PER_COLUMN = 5

# Business rules
FRAGILE_KEYWORDS = ["glass", "porcelain", "fragile", "ceramic", "china"]
VELOCITY_BAND_A_PERCENTILE = 80
VELOCITY_BAND_B_PERCENTILE = 50

# UI branding
SOLUTION_NAME = "Storeganizer"
LENA_PERSONA = "Lena is a Storeganizer logistics specialist..."

# 3D visualization URLs (from Dimitri)
VISUALIZATION_URLS = {
    "large": "https://offer.storeganizer.com/visualisations/02171ec5-...",
    "medium": "https://offer.storeganizer.com/visualisations/668f6dcd-...",
    ...
}
```

**To customize**: Edit this one file. Everything else updates automatically.

---

## Core Modules Overview

### `core/eligibility.py`

Filters SKUs based on Storeganizer specifications.

**Functions**:
- `apply_dimension_filter()` - Width/depth/height checking
- `apply_weight_filter()` - Weight limits
- `apply_velocity_filter()` - A/B/C band filtering
- `apply_forecast_filter()` - Demand threshold
- `apply_fragile_filter()` - Remove fragile items
- `apply_all_filters()` - Combined filtering with rejection tracking

**Usage**:
```python
filtered_df, rejected_count, reasons = apply_all_filters(
    df,
    max_width=450,
    max_height=450,
    velocity_band="A",
)
```

### `core/allocation.py`

Maps SKUs to specific bay/column/row locations.

**Functions**:
- `compute_planning_metrics()` - Calculate units_required, velocity_band, overweight flags
- `build_layout()` - Generate bay/column/row allocations
- `calculate_bay_requirements()` - Estimate bay count from SKU count

**Key algorithm**:
1. Group SKUs by family (first 2 words of description)
2. Sort by family, then by demand (keeps related products together)
3. Allocate to columns sequentially, respecting row capacity
4. Track weight per column, flag overweight conditions

### `core/data_ingest.py`

Handles flexible CSV/Excel uploads.

**Functions**:
- `load_inventory_file()` - Full pipeline (read → harmonize → validate)
- `harmonize_columns()` - Map aliases ("article" → "sku_code", etc.)
- `coerce_types()` - Ensure correct data types
- `validate_required_columns()` - Check for missing columns

**Supported aliases**:
- sku_code: "sku", "article", "article_number", "item_code"
- weekly_demand: "demand", "forecast", "fcst", "ews"
- stock_weeks: "weeks", "wos", "weeks_of_stock"
- (etc.)

---

## 3D Visualization Plan

**Current**: Placeholder in `visualization/viewer_3d.py`

**Dimitri's URLs provided**:
- Large config: https://offer.storeganizer.com/visualisations/02171ec5-...
- Medium: ...668f6dcd-...
- Small: ...ffcafcad-...
- XS: ...a40e1a1f-...

**Workshop goals** (Belgium, next week):
1. Understand URL parameter structure to pass planogram data
2. Explore API integration vs iframe embed
3. Discuss customization configurator integration
4. Determine internal UI modernization scope

**Implementation approach**:
- Alpha: Show placeholder with links to Dimitri's visualizations
- Beta: Integrate based on workshop findings (iframe or API)
- V1: Full 3D interaction with SKU-level detail

---

## Next Steps for Fred

### 1. Spawn Coding Agent

Use the prompt below (or customize as needed).

### 2. Review Agent's Work

After agent completes:
- Check that app launches: `cd /home/flinux/storeganizer-alpha && streamlit run app.py`
- Verify no Speedcell/VASS/Hyllevagn references remain
- Test Lena chat
- Test file upload with sample CSV
- Check planogram visualization

### 3. Fix Any Issues

Agent might miss edge cases - review and fix together.

### 4. Test with Real Data

Use sample Storeganizer SKU data (or create test data matching their specs).

### 5. Git & GitHub

```bash
cd /home/flinux/storeganizer-alpha
git init
git add .
git commit -m "Initial commit: Storeganizer Planning Tool Alpha"
gh repo create storeganizer-alpha --private --source=. --push
```

### 6. Send to Dimitri

- Share GitHub repo (or zip file)
- Include README and setup instructions
- Mention 3D integration planned for Belgium workshop

---

## Coding Agent Prompt (Ready to Use)

Copy this prompt when spawning the coding agent:

```
You are a Python/Streamlit coding specialist. Your task is to refactor a large Streamlit application to use a clean modular architecture.

CONTEXT:
I'm building a warehouse planning tool for Storeganizer (high-density storage systems). I've extracted the core business logic into separate modules and created a config system. Now I need you to refactor the main Streamlit app to use this new architecture.

TASK:
Refactor /home/flinux/99.project_nov/hd_storage_planner/app.py → /home/flinux/storeganizer-alpha/app.py

DETAILED INSTRUCTIONS:
Read /home/flinux/storeganizer-alpha/REFACTORING_INSTRUCTIONS.md for complete step-by-step guidance.

KEY REQUIREMENTS:
1. Remove ALL references to Speedcell, VASS, Hyllevagn, Tornado (competitor products)
2. Replace inline logic with calls to core modules (eligibility, allocation, data_ingest)
3. Use config.storeganizer for all Storeganizer-specific values (dimensions, weights, defaults)
4. Simplify wizard to Storeganizer-only flow (no multi-solution branching)
5. Maintain all working functionality (Lena chat, file upload, filtering, planning, visualization)

MODULES TO USE:
- from core import eligibility, allocation, data_ingest
- from rag import rag_service
- from visualization import planogram_2d
- from config import storeganizer as config

TARGET:
Clean, professional Streamlit app (~1500-2000 lines, down from 3000) that works exclusively for Storeganizer.

TESTING:
After refactoring, verify the app launches and all features work.

START BY:
1. Reading REFACTORING_INSTRUCTIONS.md thoroughly
2. Reading the original app.py to understand current structure
3. Creating new app.py using the modular architecture
4. Testing that it works

Begin now.
```

---

## Time Estimate

**Phase 1 (Complete)**: 45 minutes
**Phase 2 (Coding agent)**: 3-4 hours
**Phase 3 (Testing & fixes)**: 1-2 hours
**Phase 4 (Git & deployment)**: 30 minutes

**Total to demo-ready**: 6-8 hours

With your obsessive gremlin mode activated, you could have this done in 12 hours and be sending Dimitri the link tomorrow morning.

---

## Success Indicators

You'll know Phase 2 is done when:

✅ `streamlit run app.py` launches without errors
✅ Lena chat answers Storeganizer questions
✅ File upload works with CSV/Excel
✅ Filtering removes SKUs correctly
✅ Planogram generates with bay/column/row allocations
✅ No competitor references in UI or console
✅ Code is clean and uses core modules

---

## Questions for Dimitri (Belgium Workshop)

Based on this architecture, prepare these questions:

1. **3D Integration**: What's the best way to pass planogram data to your visualization platform?
2. **Customization**: Can we integrate your configurator into the planning workflow?
3. **Data Exchange**: What format do you prefer for exporting allocation data?
4. **Deployment**: Should this run on your servers, my servers, or cloud?
5. **Future Features**: What other internal tools could benefit from AI integration?

---

## Fred's Decision Point

**Option A: Spawn agent now** (recommended)
- I've done the hard architectural work
- Agent does mechanical refactoring
- You review and fix edge cases
- Demo ready tonight/tomorrow

**Option B: You code it yourself**
- More control over details
- Longer time investment
- Still demo-ready within 24h given your speed

**Option C: We pair-program it together**
- I guide, you code
- Slower but you learn the codebase deeply
- Demo-ready tomorrow

What's the call, mate?

---

**Phase 1 Status**: ✅ COMPLETE
**Next Step**: Spawn coding agent with prompt above
**ETA to Demo**: 6-8 hours from now

You've got solid foundations. Time to build the house.
