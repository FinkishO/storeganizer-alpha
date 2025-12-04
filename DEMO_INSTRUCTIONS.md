# Storeganizer Planning Tool - Demo Instructions for Dimitri

**Demo URL**: [To be provided by Fred after Streamlit Cloud deployment]

---

## What This Tool Does

Automatically plans Storeganizer layouts from raw inventory data:
1. Upload your SKU list (CSV/Excel)
2. Select Storeganizer configuration (XS, Small, Medium, Large)
3. Apply eligibility filters (dimensions, weight, velocity, demand)
4. Generate bay-level planogram
5. Download planning files for implementation

---

## Quick Demo Workflow (5 minutes)

### 1. Choose Configuration
- Click through the 4 standard configs (XS, Small, Medium, Large)
- Each shows 3D image, cells per bay, typical deployment size
- Try adjusting "Number of bays" slider
- Watch capacity summary update in real-time

**Try**: Select "Medium" (recommended), set 5 bays

### 2. Upload Inventory
- Click "Browse files" and upload `sample_inventory.csv` (included in repo)
- Tool shows:
  - 80 SKUs loaded
  - Required vs. optional columns detected
  - Preview of first 10 rows
  - ASSQ (auto stacking quantity) calculated for each SKU

**What to notice**: Tool handles flexible column names automatically

### 3. Filter SKUs
- Adjust filters:
  - **Velocity band**: A (fast), B (medium), C (slow), or All
  - **Forecast threshold**: Max weekly demand (default 4.0)
  - **Remove fragile**: Toggle to exclude glass, porcelain, ceramic items
  - **Allow squeeze**: +10% width tolerance for soft packaging
- Click "Apply filters"
- Review rejection summary

**Expected result**: ~59 SKUs eligible, ~21 rejected (16 dimensions, 5 demand)

### 4. Generate Plan
- Click "Run planning"
- Tool calculates:
  - Columns required per SKU
  - Bay allocations
  - Weight per column
  - Overweight flags

**What to notice**: Planning table shows each SKU's bay/column assignment, velocity band, weight loading

### 5. Review & Export
- View 2D planogram (color by SKU or velocity)
- Download 3 CSV files:
  1. **Planning table**: SKU-level details (columns required, weight, velocity)
  2. **Columns summary**: Weight per column, overweight flags
  3. **Planogram blocks**: Exact bay/column/row placement for each SKU

**Real-world use**: Give these files to warehouse team for physical layout

---

## Ask Lena (AI Assistant)

Chat sidebar on the left - try these:

- "What are Storeganizer pocket dimensions?"
- "How many bays do I need for 500 SKUs?"
- "What are the weight limits?"
- "Check if my SKUs fit Storeganizer sizing"

Lena knows Storeganizer specs, eligibility rules, and planning calculations.

---

## Key Features to Highlight

### 1. Visual Configuration Selection
- No manual spec entry - just click a card
- 3D images show actual product
- Capacity calculator updates live

### 2. Intelligent Filtering
- Automatic rejection reasons (dimensions, weight, demand)
- See exactly why SKUs don't fit
- Helps refine inventory selection

### 3. ASSQ (Auto-Stacking Quantity)
- Tool calculates how many units fit per pocket
- Considers multibox/case quantities if provided
- Flags SKUs needing manual review

### 4. Weight Management
- Calculates weight per column
- Flags overweight columns (exceeding 100kg)
- Helps prevent safety issues

### 5. Velocity-Based Planning
- A/B/C velocity bands calculated automatically
- Can filter by velocity (e.g., only slow-movers)
- Planogram colors by velocity for visual planning

---

## Sample Data Explained

`sample_inventory.csv` contains 80 SKUs:
- **75 slow-movers** (≤ 4.0 units/week demand) - Storeganizer candidates
- **5 fast-movers** (> 4.0 demand) - should be rejected (too fast for organized storage)
- **Mix of sizes**: Some fit Medium config (450×500×450mm), some too large
- **Mix of weights**: Some exceed 20kg pocket limit or 100kg column limit
- **5 fragile items**: Can be filtered out with "Remove fragile" toggle

This diversity demonstrates all filtering scenarios you'd encounter with real inventory.

---

## Business Value

### Current State (Manual)
- Client sends inventory spreadsheet
- You manually check dimensions, calculate capacity
- Trial-and-error to determine bay count
- Planning service: ~€5,000 per project

### With This Tool (Automated)
- Client uploads inventory
- Tool filters and plans in seconds
- Clear rejection reasons help client refine list
- Downloadable layout files ready for warehouse

**Potential uses:**
1. **Sales tool**: Show prospects feasibility instantly during meetings
2. **Planning service**: Automate tedious manual calculations
3. **Client self-service**: Let clients pre-qualify their inventory
4. **Upsell opportunity**: "You have 150 rejected SKUs - consider XL configuration"

---

## Next Steps / Feedback Wanted

1. **Configuration accuracy**: Do the 4 standard configs match your product specs?
2. **Eligibility rules**: Are dimension/weight filters correct? Any missing criteria?
3. **Planning logic**: Does bay/column allocation make sense for your process?
4. **Export format**: Do the CSV downloads have the info you need for implementation?
5. **3D visualization**: Would iframe integration of your 3D viewer add value?
6. **Additional features**: What would make this production-ready for you?

---

## Questions to Consider

- Could this replace manual planning work?
- Would clients use this themselves, or Storeganizer team only?
- What pricing model makes sense? (per-use, per-client, unlimited license)
- Should we build similar tools for Speedcell and Tornado-VASS?
- How does this fit into your current sales/implementation workflow?

---

**Let's discuss in Belgium! Looking forward to hearing your thoughts.**

— Fred
