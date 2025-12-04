# App.py Refactoring Instructions for Coding Agent

## Mission

Refactor `/home/flinux/99.project_nov/hd_storage_planner/app.py` into `/home/flinux/storeganizer-alpha/app.py` to use the new modular architecture.

**Goal**: Clean, Storeganizer-only Streamlit app that uses the core modules instead of inline logic.

---

## What's Been Done (Phase 1)

✅ Folder structure created
✅ Core modules extracted:
   - `core/eligibility.py` - SKU filtering logic
   - `core/allocation.py` - Bay/column/row mapping
   - `core/data_ingest.py` - CSV/Excel parsing
✅ Config system created:
   - `config/storeganizer.py` - All Storeganizer-specific settings
✅ RAG system copied and modified:
   - `rag/rag_service.py` - Lena chat (Storeganizer-only)
   - `rag/rag_store.py` - Vector database
✅ Visualization prepared:
   - `visualization/planogram_2d.py` - 2D charts (copied from original)
   - `visualization/viewer_3d.py` - 3D placeholder
✅ Documentation:
   - `README.md` - Comprehensive project documentation
   - `.gitignore` - Standard Python ignore file

---

## Your Task (Phase 2)

Refactor the original `app.py` (3000 lines) to:

1. **Use core modules** instead of inline logic
2. **Remove all multi-solution code** (Speedcell, VASS, Hyllevagn, Tornado references)
3. **Use config system** instead of hardcoded values
4. **Simplify UI** - Storeganizer-only wizard
5. **Keep Lena chat** using `rag.rag_service`
6. **Maintain all working functionality** from original

---

## Detailed Refactoring Steps

### 1. Import Structure

**Replace:**
```python
import planner
import layout_engine
import rag_service
# etc.
```

**With:**
```python
from core import eligibility, allocation, data_ingest
from rag import rag_service
from visualization import planogram_2d
from config import storeganizer as config
```

### 2. Remove Multi-Solution Logic

**Find and remove:**
- `SOLUTION_GUIDE_LINKS` dict with all four solutions
- Solution dropdown/selectbox UI elements
- `solution_type` session state variable
- Conditional logic like `if solution_type == "Speedcell"`
- Dynamic prompts that switch based on solution type
- All references to Speedcell, VASS, Hyllevagn, Tornado

**Keep only:**
- Storeganizer references
- Single-solution workflow

### 3. Replace Hardcoded Values with Config

**Find:**
```python
DEFAULT_POCKET_WIDTH = 450
cell_width = st.number_input(...)
```

**Replace with:**
```python
from config import storeganizer as config

DEFAULT_POCKET_WIDTH = config.DEFAULT_POCKET_WIDTH
cell_width = st.number_input(..., value=config.DEFAULT_POCKET_WIDTH)
```

**Apply to:**
- All dimension defaults
- Weight limits
- Columns/rows defaults
- Lena persona text
- Suggested prompts

### 4. Replace Inline Functions with Core Modules

#### Eligibility Filtering

**Original (lines 842-887):**
```python
def apply_inventory_filters(df: pd.DataFrame):
    # 45 lines of filtering logic
    ...
```

**Refactored:**
```python
from core.eligibility import apply_all_filters

# In wizard Step 4 or wherever filtering happens:
filtered_df, rejected_count, rejection_reasons = apply_all_filters(
    df,
    max_width=st.session_state.get("elig_max_w", config.DEFAULT_POCKET_WIDTH),
    max_depth=st.session_state.get("elig_max_d", config.DEFAULT_POCKET_DEPTH),
    max_height=st.session_state.get("elig_max_h", config.DEFAULT_POCKET_HEIGHT),
    max_weight_kg=st.session_state.get("elig_max_weight", config.DEFAULT_POCKET_WEIGHT_LIMIT),
    velocity_band=st.session_state.get("velocity_band_filter", "All"),
    max_weekly_demand=st.session_state.get("forecast_threshold", config.DEFAULT_FORECAST_THRESHOLD),
    allow_squeeze=st.session_state.get("allow_extra_width", False),
    remove_fragile=st.session_state.get("remove_fragile", False),
)
```

#### Data Ingestion

**Original:**
```python
def parse_flexible_table(file_like):
    # CSV parsing, column harmonization, type coercion
    ...
    df = harmonize_columns(df, aliases)
    ...
```

**Refactored:**
```python
from core.data_ingest import load_inventory_file

# In file upload section:
try:
    df = load_inventory_file(uploaded_file)
    st.success(f"Loaded {len(df)} SKUs")
except ValueError as e:
    st.error(f"File error: {e}")
```

#### Planning Metrics

**Original:**
```python
import planner

df = planner.compute_planning_metrics(df, units_per_column, max_weight_per_column_kg)
```

**Refactored:**
```python
from core.allocation import compute_planning_metrics

df = compute_planning_metrics(df, units_per_column, max_weight_per_column_kg)
```

#### Layout Generation

**Original:**
```python
import layout_engine

df, blocks, columns_summary = layout_engine.build_layout(
    df, bays, columns_per_bay, rows_per_column, units_per_column, max_weight_per_column_kg
)
```

**Refactored:**
```python
from core.allocation import build_layout

df, blocks, columns_summary = build_layout(
    df, bays, columns_per_bay, rows_per_column, units_per_column, max_weight_per_column_kg
)
```

### 5. Update Lena Chat Integration

**Original:**
```python
import rag_service

answer, docs = rag_service.answer(user_message, context)
```

**Refactored:**
```python
from rag import rag_service

answer, docs = rag_service.answer(user_message, context)
```

**Also update:**
- `LENA_PERSONA` → use `config.LENA_PERSONA`
- `LENA_KNOWLEDGE` → use `config.LENA_KNOWLEDGE_AREAS`
- `LENA_SUGGESTED` → use `config.SUGGESTED_PROMPTS`

### 6. Simplify Wizard Flow

The original has a 6-step wizard with multi-solution branching. Simplify to Storeganizer-only:

**Step 1: Introduction**
- Remove solution selector dropdown
- Show Storeganizer branding only
- Lena chat with Storeganizer context

**Step 2: Solution Configuration** (rename to "Pocket Configuration")
- Remove solution type switching
- Show only Storeganizer specs (width, depth, height, weight)
- Load defaults from config

**Step 3: Upload Inventory**
- Use `core.data_ingest.load_inventory_file()`
- Show column status
- Keep CSV/Excel auto-detection

**Step 4: Filter & Refine**
- Use `core.eligibility.apply_all_filters()`
- Show rejection summary using `eligibility.get_rejection_summary()`

**Step 5: Plan & Optimize**
- Use `core.allocation.compute_planning_metrics()`
- Use `core.allocation.build_layout()`

**Step 6: Review & Export**
- Use `visualization.planogram_2d` (already copied)
- Add placeholder for 3D viewer (optional):
  ```python
  from visualization.viewer_3d import embed_3d_viewer
  if st.checkbox("Preview 3D (coming soon)"):
      st.components.v1.html(embed_3d_viewer({}, height=650)
  ```

### 7. Remove Consumer Package (CP) Mode

The original has special CP format handling for IKEA-specific data. Unless Storeganizer needs this:

**Remove:**
- `cp_format` session state variable
- `parse_cp_excel()` function
- CP-specific UI elements
- CP-specific prompts

**Or simplify:**
- If Dimitri might want CP support, keep the logic but simplify it

---

## Files to Reference

### Original Source
`/home/flinux/99.project_nov/hd_storage_planner/app.py` - Full original implementation

### New Modules to Use
- `/home/flinux/storeganizer-alpha/core/eligibility.py`
- `/home/flinux/storeganizer-alpha/core/allocation.py`
- `/home/flinux/storeganizer-alpha/core/data_ingest.py`
- `/home/flinux/storeganizer-alpha/config/storeganizer.py`
- `/home/flinux/storeganizer-alpha/rag/rag_service.py`

### Supporting Files
- `/home/flinux/storeganizer-alpha/visualization/planogram_2d.py` (already copied)
- `/home/flinux/storeganizer-alpha/visualization/viewer_3d.py` (placeholder)

---

## Testing Checklist

After refactoring, verify:

- [ ] App launches without errors: `streamlit run app.py`
- [ ] Lena chat works and knows Storeganizer specs
- [ ] File upload accepts CSV and Excel
- [ ] Column harmonization works (aliases map correctly)
- [ ] Eligibility filters remove SKUs correctly
- [ ] Planning metrics calculate (units_required, velocity_band, etc.)
- [ ] Layout generation creates bay/column/row allocations
- [ ] Planogram visualization renders
- [ ] Weight flagging works (overweight columns highlighted)
- [ ] Export downloads CSV files
- [ ] No references to Speedcell/VASS/Hyllevagn in UI or console logs

---

## Output Deliverable

**Primary**: `/home/flinux/storeganizer-alpha/app.py` - Fully refactored Streamlit app

**Structure** (approximate):
- Lines 1-50: Imports and config loading
- Lines 51-200: Helper functions (using core modules)
- Lines 201-500: Lena chat sidebar
- Lines 501-2000: Wizard steps 1-6 (simplified, Storeganizer-only)
- Lines 2001-2200: Main app logic and session state init

**Target**: ~1500-2000 lines (down from 3000)

---

## Common Pitfalls to Avoid

1. **Don't copy-paste inline logic** - Use the core modules
2. **Don't leave competitor references** - Remove all Speedcell/VASS/etc.
3. **Don't hardcode dimensions** - Use config everywhere
4. **Don't break Lena chat** - Ensure rag_service.answer() works
5. **Don't forget type coercion** - data_ingest handles this, don't duplicate

---

## If You Get Stuck

**Module not found errors**: Check import paths and __init__.py files
**Config not loading**: Verify `from config import storeganizer as config`
**Lena not answering**: Check rag/ folder has rag_store.py and ingest_ref.py
**Visualization broken**: Ensure planogram_2d.py copied correctly

---

## Success Criteria

✅ App runs without errors
✅ All Speedcell/VASS/Hyllevagn references removed
✅ Uses core modules (not inline logic)
✅ Uses config for all Storeganizer specs
✅ Lena chat works
✅ File upload, filtering, planning, visualization all functional
✅ Code is clean, readable, and well-structured
✅ Ready for testing with sample Storeganizer data

---

## Final Note

This refactoring sets up the architecture for:
- Easy creation of Speedcell/VASS versions (just swap config)
- Professional code quality for potential handover to Storeganizer
- 3D visualization integration (post-Belgium workshop)
- Future enterprise features (multi-client, saved presets, API layer)

Take your time, test thoroughly, and ensure all functionality from the original is preserved.

Good luck, coding agent. You've got this.
