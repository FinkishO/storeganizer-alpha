# Codex Task: Redesign Step 2 - Configuration Selection

## Context

The Storeganizer Planning Tool currently has Step 2 showing raw number inputs for pocket dimensions. This is confusing for users who don't know Storeganizer specs. We need to redesign it with visual configuration cards showing 4 standard sizes (XS, Small, Medium, Large) with images, capacity info, and price placeholders.

## Your Task

Implement a user-friendly configuration selection UI in Step 2 that guides customers through choosing the right Storeganizer configuration.

## Detailed Instructions

Read `/home/flinux/storeganizer-alpha/STEP2_REDESIGN_INSTRUCTIONS.md` for complete implementation details.

**Summary of changes**:

1. **Update config file** (`config/storeganizer.py`):
   - Add `STANDARD_CONFIGS` dict with 4 configurations (XS, Small, Medium, Large)
   - Each config includes: dimensions, weight limits, columns/rows, cells per bay, image path, pricing placeholder

2. **Rewrite Step 2** (`app.py` function `render_step_configuration()`):
   - Section 1: 4 clickable configuration cards with images (sg_xs/s/m/l.png)
   - Section 2: Bay count selector
   - Section 3: Dynamic capacity summary (cells, SKUs, price estimate)
   - Section 4: Advanced expander for custom dimensions (optional)

3. **Update session state** (`init_session_state()`):
   - Add: `selected_config_size` (default "medium")
   - Add: `num_bays` (default 5)
   - Add: `show_custom_config` (default False)

## Expected User Flow

1. User lands on Step 2 → Sees 4 config cards in a grid
2. Medium config is pre-selected (highlighted, marked "Recommended")
3. User clicks "Large" card → Card highlights, session state updates with Large dimensions
4. User changes bay count from 5 to 10 → Capacity summary recalculates instantly
5. Capacity summary shows: "60 cells/bay × 10 bays = 600 total cells, ~480 SKUs (80% util)"
6. User clicks "Next" → Proceeds to Step 3 with Large config + 10 bays

## Visual Design Requirements

**Configuration Cards**:
- 4 columns grid layout (st.columns(4))
- Each card shows:
  - Config name as button (st.button with use_container_width=True)
  - Description text (st.caption)
  - 3D image preview (st.image from sg_xs/s/m/l.png)
  - Cells per bay metric (st.metric)
  - Typical bay range caption
  - "✨ Recommended" badge for Medium (st.success)
- Selected card: type="primary", non-selected: type="secondary"

**Capacity Summary**:
- 3-column metrics (st.columns(3))
- Metric 1: Total Cells = bays × cells_per_bay
- Metric 2: Estimated SKUs = total_cells × 0.8
- Metric 3: Price = "TBD" (placeholder for future)
- Info box below with current selection summary

**Advanced Options**:
- Collapsed by default (st.expander)
- Contains existing manual input UI (keep current number inputs)
- If user changes anything here, set selected_config_size to "custom"

## Files to Modify

1. `/home/flinux/storeganizer-alpha/config/storeganizer.py`
   - Add after line 22 (after DEFAULT_UNITS_PER_COLUMN)
   - Insert STANDARD_CONFIGS dict (see STEP2_REDESIGN_INSTRUCTIONS.md for exact structure)

2. `/home/flinux/storeganizer-alpha/app.py`
   - Update `init_session_state()` around line 50-77
   - Replace `render_step_configuration()` function (lines 245-320)
   - Keep navigation buttons at bottom

## Configuration Specs (Use These)

```python
STANDARD_CONFIGS = {
    "xs": {
        "name": "Extra Small",
        "description": "Compact setup for smaller operations",
        "pocket_width": 300,
        "pocket_depth": 400,
        "pocket_height": 350,
        "pocket_weight_limit": 15.0,
        "columns_per_bay": 6,
        "rows_per_column": 4,
        "cells_per_bay": 24,
        "typical_bays": "2-4 bays",
        "image": "ref/sg_xs.png",
        "price_per_bay_eur": None,
    },
    "small": {
        "name": "Small",
        "description": "Standard configuration for growing businesses",
        "pocket_width": 400,
        "pocket_depth": 450,
        "pocket_height": 400,
        "pocket_weight_limit": 18.0,
        "columns_per_bay": 7,
        "rows_per_column": 5,
        "cells_per_bay": 35,
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
        "cells_per_bay": 40,
        "typical_bays": "6-12 bays",
        "image": "ref/sg_m.png",
        "price_per_bay_eur": None,
        "recommended": True,
    },
    "large": {
        "name": "Large",
        "description": "High-capacity for warehouses with extensive inventory",
        "pocket_width": 500,
        "pocket_depth": 550,
        "pocket_height": 500,
        "pocket_weight_limit": 25.0,
        "columns_per_bay": 10,
        "rows_per_column": 6,
        "cells_per_bay": 60,
        "typical_bays": "8-20 bays",
        "image": "ref/sg_l.png",
        "price_per_bay_eur": None,
    },
}

DEFAULT_CONFIG_SIZE = "medium"
```

## Testing Checklist

After implementation, verify:
- [ ] Step 2 shows 4 configuration cards with images
- [ ] Medium card has "✨ Recommended" badge
- [ ] Clicking a card highlights it and updates session state
- [ ] Clicking a card applies config dimensions to pocket_width/depth/height/etc
- [ ] Bay count input updates capacity summary in real-time
- [ ] Capacity calculations correct: cells = bays × cells_per_bay
- [ ] Advanced expander shows custom inputs (collapsed by default)
- [ ] Navigation buttons work (Back to Step 1, Next to Step 3)
- [ ] No errors in console when clicking cards or changing bay count

## Important Notes

- **Don't break existing functionality** - Steps 3-6 should still work after this change
- **Use st.rerun()** after config selection to update UI immediately
- **Keep type consistency** - All number_input min/max/value must be same type (float or int)
- **Image paths** - Images are at `ref/sg_xs.png`, `ref/sg_s.png`, `ref/sg_m.png`, `ref/sg_l.png`
- **Session state persistence** - Selected config should persist when navigating back/forward

## Example Card Rendering Code

```python
# In render_step_configuration():
st.markdown("### Standard Configurations")
cols = st.columns(4)

for idx, (config_key, config_data) in enumerate(config.STANDARD_CONFIGS.items()):
    with cols[idx]:
        is_selected = (st.session_state["selected_config_size"] == config_key)

        # Button for selection
        if st.button(
            config_data["name"],
            key=f"config_{config_key}",
            use_container_width=True,
            type="primary" if is_selected else "secondary",
        ):
            # Apply this configuration
            st.session_state["selected_config_size"] = config_key
            st.session_state["pocket_width"] = config_data["pocket_width"]
            st.session_state["pocket_depth"] = config_data["pocket_depth"]
            st.session_state["pocket_height"] = config_data["pocket_height"]
            st.session_state["pocket_weight_limit"] = config_data["pocket_weight_limit"]
            st.session_state["columns_per_bay"] = config_data["columns_per_bay"]
            st.session_state["rows_per_column"] = config_data["rows_per_column"]
            st.rerun()

        # Card content
        st.caption(config_data["description"])
        st.image(config_data["image"], use_container_width=True)
        st.metric("Cells/Bay", config_data["cells_per_bay"])
        st.caption(f"Typical: {config_data['typical_bays']}")

        if config_data.get("recommended"):
            st.success("✨ Recommended")
```

## Questions?

If anything is unclear, refer to:
- `/home/flinux/storeganizer-alpha/STEP2_REDESIGN_INSTRUCTIONS.md` - Full detailed spec
- `/home/flinux/storeganizer-alpha/config/storeganizer.py` - Current config structure
- `/home/flinux/storeganizer-alpha/app.py` lines 245-320 - Current Step 2 implementation

---

**Start by reading STEP2_REDESIGN_INSTRUCTIONS.md, then implement the changes. Test thoroughly before marking complete.**
