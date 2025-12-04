# Storeganizer Alpha - Deployment Instructions

## Option 1: Streamlit Cloud (Recommended for Demo)

**Best for sharing with Dimitri - gives him a clickable URL**

### Steps (Fred to complete):

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select:
   - **Repository**: `FinkishO/storeganizer-alpha`
   - **Branch**: `master`
   - **Main file path**: `app.py`
5. Click "Deploy"
6. Wait ~5 minutes for deployment
7. Copy the URL (will be something like: `https://storeganizer-alpha.streamlit.app`)
8. Send URL to Dimitri

**Note**: RAG database will need to be re-initialized on first deploy. If Lena chat doesn't work immediately, run the ingest script once via terminal (see Option 2 below for RAG setup).

---

## Option 2: Local Setup Instructions (for Dimitri)

If Streamlit Cloud deployment has issues, Dimitri can run locally:

### Prerequisites

- Python 3.10+ installed
- Git installed

### Setup Steps

```bash
# Clone the repository
git clone https://github.com/FinkishO/storeganizer-alpha.git
cd storeganizer-alpha

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Initialize RAG database (for Lena chat)
python rag/ingest_ref.py

# Run the app
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

---

## Testing the Demo

### Workflow to demonstrate:

1. **Step 1**: Welcome page → Click "Start planning"
2. **Step 2**: Choose configuration
   - Try clicking different config cards (XS, Small, Medium, Large)
   - Medium is marked as "Recommended"
   - Adjust "Number of bays" (try 5-10)
   - Note the capacity summary updates dynamically
   - Click "Next"
3. **Step 3**: Upload inventory
   - Upload the included `sample_inventory.csv` (80 SKUs)
   - Review the preview
   - Click "Next: Filter & refine"
4. **Step 4**: Filter & refine
   - Try different velocity bands (A, B, C, All)
   - Toggle "Remove fragile items" to see effect
   - Adjust forecast threshold
   - Click "Apply filters"
   - Review rejection summary (should show ~59 eligible SKUs)
   - Click "Next: Plan & optimize"
5. **Step 5**: Plan & optimize
   - Click "Run planning"
   - Review planning metrics (SKUs planned, bays, overweight columns)
   - Check planning preview table
   - Click "Next: Review & export"
6. **Step 6**: Review & export
   - View the 2D planogram visualization
   - Try "Color by velocity" option
   - Download exports:
     - Planning table CSV
     - Columns summary CSV
     - Planogram blocks CSV
7. **Lena Chat** (sidebar)
   - Try asking: "What are Storeganizer pocket dimensions?"
   - Try: "How many bays for 500 SKUs?"
   - Lena has knowledge of Storeganizer specs, eligibility rules, and planning calculations

---

## Sample Inventory Details

The included `sample_inventory.csv` contains:
- 80 SKUs total
- 75 slow-movers (demand ≤ 4.0 units/week) - Storeganizer candidates
- 5 fast-movers (demand > 4.0) - should be rejected by forecast filter
- Mix of sizes (some fit Medium config, some too large)
- Mix of weights (some exceed 20kg pocket limit)
- 5 fragile items (Glass, Ceramic, Porcelain, China) - can be filtered out

**Default Medium configuration**:
- Pocket: 450mm W × 500mm D × 450mm H
- Weight limit: 20kg per pocket, 100kg per column
- Structure: 8 columns × 5 rows per bay = 40 cells/bay

**Expected results with default settings**:
- ~59 SKUs pass eligibility (73% pass rate)
- ~16 rejected for dimensions (too large)
- ~5 rejected for forecast threshold (too much demand)
- With 5 bays × 40 cells = 200 total cells
- Estimated capacity: ~160 SKUs at 80% utilization

---

## Known Limitations (for V2)

- 3D visualization is placeholder only (future Dimitri integration)
- Pricing in config cards is "TBD" (placeholder for real pricing)
- RAG knowledge base includes Storeganizer specs only (no competitor data)
- No user authentication (single-user demo tool)

---

## Troubleshooting

**If Lena chat doesn't respond:**
```bash
# Re-initialize RAG database
python rag/ingest_ref.py
```

**If app won't start:**
```bash
# Check Python version
python3 --version  # Should be 3.10+

# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

**If Streamlit Cloud deployment fails:**
- Check that all files pushed to GitHub
- Verify `requirements.txt` includes all dependencies
- Check deployment logs in Streamlit Cloud dashboard
- May need to add `.streamlit/config.toml` if custom settings needed

---

## Repository Structure

```
storeganizer-alpha/
├── app.py                 # Main Streamlit application
├── config/
│   └── storeganizer.py    # Product specs, business rules
├── core/
│   ├── allocation.py      # Bay/column planning logic
│   ├── eligibility.py     # SKU filtering rules
│   └── data_ingest.py     # CSV/Excel parsing
├── rag/
│   ├── rag_service.py     # Lena chat interface
│   ├── rag_store.py       # Vector database
│   └── ingest_ref.py      # Knowledge base ingestion
├── visualization/
│   ├── planogram_2d.py    # 2D planogram rendering
│   └── viewer_3d.py       # 3D viewer placeholder
├── ref/                   # Reference materials (PDFs, images)
├── source/                # UI assets (Lena avatar)
├── sample_inventory.csv   # Demo data
└── requirements.txt       # Python dependencies
```

---

## Contact

For questions or issues, contact Fred Olsen (finke.olsen@gmail.com)
