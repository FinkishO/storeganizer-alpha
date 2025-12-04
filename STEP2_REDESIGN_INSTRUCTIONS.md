# Step 2 Pocket Configuration Redesign

## Current Problem

Step 2 currently shows raw number inputs for pocket dimensions (width, depth, height, weight). This requires users to know Storeganizer specifications and doesn't provide guidance on which configuration to choose.

## Desired Outcome

Redesign Step 2 to guide users through configuration selection with:
1. **Visual configuration cards** - 4 standard Storeganizer sizes (XS, Small, Medium, Large) as clickable cards
2. **Product imagery** - Show 3D visualization for each size
3. **Capacity guidance** - Display cells/SKUs per bay for each config
4. **Price estimation placeholder** - Space for future pricing info
5. **Custom configuration option** - Advanced users can still manually input dimensions

## Configuration Data to Add

First, update `config/storeganizer.py` with standard configurations:

```python
# Add after line 22 (after DEFAULT_UNITS_PER_COLUMN = 30)

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
        "columns_per_bay": 6,
        "rows_per_column": 4,
        "cells_per_bay": 24,  # 6 columns × 4 rows
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
        "columns_per_bay": 7,
        "rows_per_column": 5,
        "cells_per_bay": 35,  # 7 columns × 5 rows
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
        "columns_per_bay": 10,
        "rows_per_column": 6,
        "cells_per_bay": 60,  # 10 columns × 6 rows
        "typical_bays": "8-20 bays",
        "image": "ref/sg_l.png",
        "price_per_bay_eur": None,
    },
}

# Default to Medium configuration
DEFAULT_CONFIG_SIZE = "medium"
```

## UI Redesign for Step 2

Replace the current `render_step_configuration()` function in `app.py` with this new implementation:

### New Structure

```
Step 2: Choose Your Configuration
├── Section 1: Standard Configurations (Card Grid)
│   ├── Card: XS
│   ├── Card: Small
│   ├── Card: Medium (recommended badge)
│   └── Card: Large
│
├── Section 2: Bay Count Input
│   └── Number input: "How many bays do you need?" (1-50)
│
├── Section 3: Capacity Summary (Dynamic)
│   ├── "Total cells available: {bays × cells_per_bay}"
│   ├── "Estimated SKUs: {cells × 0.8} (assuming 80% utilization)"
│   └── "Price estimate: Coming soon"
│
└── Section 4: Advanced Options (Collapsible)
    └── Custom configuration (manual number inputs like current)
```

### Implementation Details

**Session state additions:**
```python
# In init_session_state():
st.session_state.setdefault("selected_config_size", config.DEFAULT_CONFIG_SIZE)
st.session_state.setdefault("num_bays", 5)
st.session_state.setdefault("show_custom_config", False)
```

**Configuration card rendering:**
```python
def render_config_card(config_key: str, config_data: dict, is_selected: bool):
    """
    Render a clickable configuration card.

    Args:
        config_key: "xs", "small", "medium", "large"
        config_data: Dict from STANDARD_CONFIGS
        is_selected: Whether this config is currently selected

    Returns:
        True if card was clicked (user wants to select this config)
    """
    # Use st.columns and st.button with custom styling
    # Show:
    # - Config name (e.g., "Medium")
    # - Description
    # - Cells per bay
    # - Typical bay range
    # - Image preview (use st.image with config_data["image"])
    # - "RECOMMENDED" badge if config_data.get("recommended")
    # - Visual indication if is_selected (border/highlight)

    # Return True if user clicks this card
```

**Full render_step_configuration() pseudocode:**
```python
def render_step_configuration():
    st.subheader("Step 2 — Choose Your Configuration")
    st.caption("Select a standard Storeganizer configuration or customize your own")

    # ===== SECTION 1: Configuration Cards =====
    st.markdown("### Standard Configurations")

    # Create 4-column grid for config cards
    cols = st.columns(4)

    for idx, (config_key, config_data) in enumerate(config.STANDARD_CONFIGS.items()):
        with cols[idx]:
            is_selected = (st.session_state["selected_config_size"] == config_key)

            # Render card with button
            # If clicked, update session state and apply config values
            if st.button(
                f"{config_data['name']}",
                key=f"config_{config_key}",
                use_container_width=True,
                type="primary" if is_selected else "secondary",
            ):
                # User selected this config
                st.session_state["selected_config_size"] = config_key

                # Apply config values to session state
                st.session_state["pocket_width"] = config_data["pocket_width"]
                st.session_state["pocket_depth"] = config_data["pocket_depth"]
                st.session_state["pocket_height"] = config_data["pocket_height"]
                st.session_state["pocket_weight_limit"] = config_data["pocket_weight_limit"]
                st.session_state["columns_per_bay"] = config_data["columns_per_bay"]
                st.session_state["rows_per_column"] = config_data["rows_per_column"]

                st.rerun()

            # Show config details
            st.caption(config_data["description"])
            st.image(config_data["image"], use_container_width=True)
            st.metric("Cells/Bay", config_data["cells_per_bay"])
            st.caption(f"Typical: {config_data['typical_bays']}")

            if config_data.get("recommended"):
                st.success("✨ Recommended")

    st.markdown("---")

    # ===== SECTION 2: Bay Count =====
    st.markdown("### How Many Bays?")

    col1, col2 = st.columns([2, 1])
    with col1:
        num_bays = st.number_input(
            "Number of bays",
            min_value=1,
            max_value=50,
            value=st.session_state["num_bays"],
            key="num_bays",
            help="Total number of Storeganizer bays in your warehouse"
        )

    # ===== SECTION 3: Capacity Summary =====
    st.markdown("### Your Configuration Summary")

    selected_config = config.STANDARD_CONFIGS[st.session_state["selected_config_size"]]
    cells_per_bay = selected_config["cells_per_bay"]
    total_cells = num_bays * cells_per_bay
    estimated_skus = int(total_cells * 0.8)  # Assume 80% utilization

    metric_cols = st.columns(3)
    with metric_cols[0]:
        st.metric("Total Cells", total_cells)
    with metric_cols[1]:
        st.metric("Estimated SKUs", estimated_skus, help="Assuming 80% utilization")
    with metric_cols[2]:
        if selected_config["price_per_bay_eur"]:
            total_price = num_bays * selected_config["price_per_bay_eur"]
            st.metric("Estimated Price", f"€{total_price:,.0f}")
        else:
            st.metric("Estimated Price", "TBD", help="Contact Storeganizer for pricing")

    st.info(f"💡 Configuration: **{selected_config['name']}** × {num_bays} bays")

    st.markdown("---")

    # ===== SECTION 4: Advanced/Custom Config (Collapsible) =====
    with st.expander("⚙️ Advanced: Custom Configuration", expanded=False):
        st.caption("Override standard configurations with custom pocket dimensions")

        # Keep existing manual input UI here
        cols = st.columns(3)
        with cols[0]:
            st.number_input(
                "Pocket width (mm)",
                min_value=100.0,
                max_value=2000.0,
                value=float(st.session_state["pocket_width"]),
                step=5.0,
                key="pocket_width_custom",
            )
            # ... etc for all dimensions

        # If user changes anything in custom inputs,
        # update selected_config_size to "custom"

    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Welcome", use_container_width=True):
            st.session_state["wizard_step"] = 1
            st.rerun()
    with col2:
        if st.button("Next: Upload Inventory →", type="primary", use_container_width=True):
            st.session_state["wizard_step"] = 3
            st.rerun()
```

## Expected Behavior

1. **On initial load**: Medium configuration is pre-selected (visual highlight)
2. **User clicks "Small" card**:
   - Card highlights
   - Session state updates to Small config values
   - Capacity summary recalculates
   - User sees immediate feedback
3. **User changes bay count**: Capacity summary updates in real-time
4. **User opens Advanced expander**: Can manually override any dimension
5. **User clicks Next**: Proceeds to Step 3 with selected configuration

## Visual Design Notes

**Card styling**:
- Selected card: Blue border, primary button color
- Non-selected: Gray border, secondary button
- Recommended card: Green "✨ Recommended" badge
- Image: 3D visualization preview (from sg_xs/s/m/l.png)

**Capacity summary**:
- Use st.metric() for clean number display
- Show delta/help text for context
- Info box with current selection summary

**Responsive layout**:
- 4 columns on desktop
- Should stack on mobile (Streamlit handles automatically)

## Files to Modify

1. **`config/storeganizer.py`**: Add `STANDARD_CONFIGS` dict (lines 23-80)
2. **`app.py`**: Replace `render_step_configuration()` function (lines 245-310)
3. **`app.py`**: Update `init_session_state()` to include new keys (around line 50)

## Testing Checklist

After implementation:
- [ ] All 4 config cards render with images
- [ ] Clicking a card updates selection visually
- [ ] Clicking a card applies config values to session state
- [ ] Bay count input updates capacity summary in real-time
- [ ] Capacity calculations are correct (cells_per_bay × num_bays)
- [ ] Advanced expander shows custom inputs
- [ ] Navigation buttons work (back to Step 1, forward to Step 3)
- [ ] Selected config persists when navigating back/forward

## Future Enhancements (Post-Alpha)

- [ ] Pull actual pricing from Storeganizer API/database
- [ ] Add "Add to Cart" or "Request Quote" button
- [ ] Show cost per SKU metric
- [ ] Compare configurations side-by-side
- [ ] Allow users to save/export configuration choice

---

**Codex: Implement the above redesign. Focus on visual clarity, user guidance, and maintaining existing functionality. The goal is to make Step 2 self-explanatory for Dimitri's customers who may not know Storeganizer specs.**
