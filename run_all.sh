#!/usr/bin/env bash
# =============================================================================
# run_all.sh — Chicago Crime Analysis — Full Pipeline (Anaconda)
#
# BEFORE RUNNING:
#   1. Place this file inside your chicago-crime-analysis/ folder
#   2. Run:
#        chmod +x run_all.sh
#        ./run_all.sh
#
# The script will automatically download Crimes.csv (~1.7 GB) and all four
# GeoJSON boundary files if they are not already present.
# =============================================================================

set -e

GREEN='\033[0;32m'; BLUE='\033[0;34m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log()  { echo -e "${BLUE}[INFO]${NC}  $1"; }
ok()   { echo -e "${GREEN}[OK]${NC}    $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $1"; }
fail() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

echo ""
echo "============================================================"
echo "   Chicago Crime Analysis — Full Pipeline (Anaconda)"
echo "============================================================"
echo ""

# ── Confirm we are in the right folder ───────────────────────────────────────
if [ ! -f "script_01_cleaning.py" ]; then
    fail "Cannot find script_01_cleaning.py in the current folder.
Please run this script from inside the chicago-crime-analysis/ folder:
  cd chicago-crime-analysis
  ./run_all.sh"
fi

# ── Check wget is available ───────────────────────────────────────────────────
if ! command -v wget &> /dev/null; then
    fail "wget not found. Install it and re-run:
  Linux  :  sudo apt-get install wget
  Mac    :  brew install wget
  Windows:  place wget.exe on your PATH (or use WSL)"
fi

# ── Check conda ───────────────────────────────────────────────────────────────
if ! command -v conda &> /dev/null; then
    fail "conda not found. Open Anaconda Prompt or add conda to your PATH."
fi
ok "conda found: $(conda --version)"

# ── Activate conda base so we can use conda commands ─────────────────────────
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# =============================================================================
# STEP 0 — Ensure data files are present (download if missing)
# =============================================================================
echo ""
echo "============================================================"
log "STEP 0/6 — Checking data files ..."
echo "============================================================"

mkdir -p data boundaries

# ── 0a: Crimes.csv ───────────────────────────────────────────────────────────
CRIMES_URL="https://data.cityofchicago.org/api/views/ijzp-q8t2/rows.csv?accessType=DOWNLOAD"

if [ -f "data/Crimes.csv" ]; then
    ok "data/Crimes.csv already present ($(du -sh data/Crimes.csv | cut -f1))"
else
    echo ""
    warn "data/Crimes.csv not found."
    echo "  Source : Chicago Data Portal (City of Chicago Open Data)"
    echo "  Size   : ~1.7 GB"
    echo "  URL    : $CRIMES_URL"
    echo ""
    read -rp "  Download now? This may take several minutes. [Y/n] " answer
    if [[ "$answer" =~ ^[Nn]$ ]]; then
        fail "Crimes.csv is required. Re-run once you have placed it in data/Crimes.csv"
    fi
    echo ""
    log "  Downloading Crimes.csv ..."
    wget --progress=bar:force \
         --retry-connrefused \
         --tries=3 \
         --timeout=60 \
         -O "data/Crimes.csv" \
         "$CRIMES_URL" || fail "Download failed. Check your internet connection and try again."
    ok "data/Crimes.csv downloaded ($(du -sh data/Crimes.csv | cut -f1))"
fi

# ── 0b: GeoJSON boundary files ───────────────────────────────────────────────
echo ""
log "Checking GeoJSON boundary files ..."

declare -A BOUNDARIES=(
    ["Beat_Boundary.geojson"]="https://data.cityofchicago.org/api/geospatial/aerh-rz74?method=export&type=GeoJSON"
    ["District_Boundary.geojson"]="https://data.cityofchicago.org/api/geospatial/fthy-xz3r?method=export&type=GeoJSON"
    ["Ward_Boundary.geojson"]="https://data.cityofchicago.org/api/geospatial/sp34-6z76?method=export&type=GeoJSON"
    ["Comm_Boundary.geojson"]="https://data.cityofchicago.org/api/geospatial/cauq-8yn6?method=export&type=GeoJSON"
)

BOUNDARY_MISSING=0
for FNAME in "${!BOUNDARIES[@]}"; do
    if [ ! -f "boundaries/$FNAME" ]; then
        BOUNDARY_MISSING=$((BOUNDARY_MISSING + 1))
    fi
done

if [ "$BOUNDARY_MISSING" -eq 0 ]; then
    ok "All 4 GeoJSON boundary files present."
else
    warn "$BOUNDARY_MISSING boundary file(s) missing — downloading now ..."
    echo ""

    # Iterate in a fixed order for readable output
    for FNAME in "Beat_Boundary.geojson" "District_Boundary.geojson" "Ward_Boundary.geojson" "Comm_Boundary.geojson"; do
        FPATH="boundaries/$FNAME"
        if [ -f "$FPATH" ]; then
            ok "  $FNAME already present — skipping."
        else
            log "  Downloading $FNAME ..."
            wget --progress=bar:force \
                 --retry-connrefused \
                 --tries=3 \
                 --timeout=60 \
                 -O "$FPATH" \
                 "${BOUNDARIES[$FNAME]}" \
            && ok "  $FNAME downloaded." \
            || {
                warn "  Could not download $FNAME automatically."
                echo "  Manual download URL:"
                echo "    ${BOUNDARIES[$FNAME]}"
                echo "  Save as: boundaries/$FNAME"
                rm -f "$FPATH"   # remove any partial download
            }
        fi
    done
fi

# ── 0c: Final check — abort early if any required file is still missing ───────
echo ""
MISSING_FILES=0
for FNAME in "Beat_Boundary.geojson" "District_Boundary.geojson" "Ward_Boundary.geojson" "Comm_Boundary.geojson"; do
    if [ ! -f "boundaries/$FNAME" ]; then
        warn "Still missing: boundaries/$FNAME"
        MISSING_FILES=$((MISSING_FILES + 1))
    fi
done
if [ ! -f "data/Crimes.csv" ]; then
    warn "Still missing: data/Crimes.csv"
    MISSING_FILES=$((MISSING_FILES + 1))
fi
if [ "$MISSING_FILES" -gt 0 ]; then
    fail "$MISSING_FILES required file(s) could not be obtained. Fix the issues above and re-run."
fi
ok "All required data files are present — proceeding with pipeline."

# =============================================================================
# STEP 1 — Create conda environment (skipped if already exists)
# =============================================================================
echo ""
log "STEP 1/6 — Conda environment ..."

if conda env list | grep -q "^chicago-crime "; then
    warn "Environment 'chicago-crime' already exists — skipping creation."
else
    log "  Creating environment with Python 3.10 ..."
    conda create -n chicago-crime python=3.10 -y
    ok "Environment created."
fi

conda activate chicago-crime
ok "Activated: $(python --version)"

# =============================================================================
# STEP 2 — Install packages (skipped if already installed)
# =============================================================================
echo ""
log "STEP 2/6 — Installing packages ..."

conda config --add channels conda-forge --quiet 2>/dev/null || true
conda config --set channel_priority strict --quiet 2>/dev/null || true

# Check if all key packages are already importable
if python -c "import pandas, geopandas, statsmodels, sklearn, xgboost, pmdarima, folium" 2>/dev/null; then
    warn "All packages already installed — skipping."
else
    log "  Installing core data science stack ..."
    conda install -c conda-forge pandas numpy matplotlib seaborn scipy jupyterlab ipykernel -y --quiet

    log "  Installing geospatial stack ..."
    conda install -c conda-forge geopandas shapely rtree pyproj fiona folium -y --quiet

    log "  Installing ML stack ..."
    conda install -c conda-forge statsmodels scikit-learn xgboost imbalanced-learn pmdarima missingno -y --quiet

    log "  Installing utilities ..."
    conda install -c conda-forge holidays tqdm python-dateutil -y --quiet

    pip install pyarrow --quiet
    ok "All packages installed."
fi

# =============================================================================
# STEP 3 — Register Jupyter kernel
# =============================================================================
echo ""
log "STEP 3/6 — Registering Jupyter kernel ..."
python -m ipykernel install --user --name chicago-crime \
       --display-name "Chicago Crime (Python 3.10)" 2>/dev/null || true
ok "Kernel ready."

# =============================================================================
# STEP 4 — Prepare output folders
# =============================================================================
echo ""
log "STEP 4/6 — Preparing output folders ..."
mkdir -p data figures reports/figures boundaries
ok "Folders ready."

# =============================================================================
# STEP 5 — Run the three Python scripts directly
#           (no nbconvert — no conversion errors)
# =============================================================================
echo ""
echo "============================================================"
log "STEP 5a/6 — Script 1: Data Cleaning & Feature Engineering"
echo "  Expected : 3–8 minutes"
echo "  Output   : data/Crimes_Cleaned.csv, data/train.csv, data/test.csv"
echo "============================================================"

# Skip cleaning if all three output files are already present
if [ -f "data/Crimes_Cleaned.csv" ] && [ -f "data/train.csv" ] && [ -f "data/test.csv" ]; then
    warn "Cleaned data files already present — skipping Script 1."
    echo "  Found:"
    echo "    data/Crimes_Cleaned.csv  ($(du -sh data/Crimes_Cleaned.csv | cut -f1))"
    echo "    data/train.csv           ($(du -sh data/train.csv           | cut -f1))"
    echo "    data/test.csv            ($(du -sh data/test.csv            | cut -f1))"
    echo ""
    echo "  To force re-generation, delete any of those files and re-run."
else
    # Run cleaning — report which file(s) triggered it
    if [ ! -f "data/Crimes_Cleaned.csv" ]; then warn "data/Crimes_Cleaned.csv not found — running Script 1."; fi
    if [ ! -f "data/train.csv" ];          then warn "data/train.csv not found — running Script 1.";          fi
    if [ ! -f "data/test.csv" ];           then warn "data/test.csv not found — running Script 1.";           fi
    echo ""

    python script_01_cleaning.py

    if [ ! -f "data/Crimes_Cleaned.csv" ]; then
        fail "Script 1 did not produce data/Crimes_Cleaned.csv — check errors above."
    fi
    ok "Script 1 complete — $(du -sh data/Crimes_Cleaned.csv | cut -f1) saved."
fi

echo ""
echo "============================================================"
log "STEP 5b/6 — Script 2: Descriptive Analysis & Visualisations"
echo "  Expected : 5–15 minutes"
echo "  Output   : figures/*.png  +  figures/*.html"
echo "============================================================"

python script_02_descriptive.py

ok "Script 2 complete — $(ls figures/*.png 2>/dev/null | wc -l) charts saved."

echo ""
echo "============================================================"
log "STEP 5c/6 — Script 3: Machine Learning & Predictions"
echo "  Expected : 10–30 minutes"
echo "  Output   : figures/03_*.png  +  printed model results"
echo "============================================================"

python script_03_predictive.py

ok "Script 3 complete."

# =============================================================================
# STEP 6 — Summary
# =============================================================================
echo ""
echo "============================================================"
echo -e "${GREEN}  ALL STEPS COMPLETE!${NC}"
echo "============================================================"
echo ""

echo "  DATA FILES:"
for f in data/Crimes.csv data/Crimes_Cleaned.csv data/train.csv data/test.csv; do
    [ -f "$f" ] && echo "    ✅  $f  ($(du -sh $f | cut -f1))" || echo "    ❌  $f  (missing)"
done

echo ""
echo "  BOUNDARY FILES:"
for f in boundaries/Beat_Boundary.geojson boundaries/District_Boundary.geojson \
          boundaries/Ward_Boundary.geojson boundaries/Comm_Boundary.geojson; do
    [ -f "$f" ] && echo "    ✅  $f" || echo "    ❌  $f  (missing)"
done

echo ""
echo "  FIGURES ($(ls figures/*.png 2>/dev/null | wc -l) PNG files):"
for f in figures/*.png; do
    [ -f "$f" ] && echo "    ✅  $f"
done

echo ""
echo "  INTERACTIVE MAPS:"
for f in figures/*.html; do
    [ -f "$f" ] && echo "    ✅  $f"
done

echo ""
echo "------------------------------------------------------------"
echo "  Open interactive maps:"
echo "    Linux  :  xdg-open figures/02_choropleth_community.html"
echo "    Mac    :  open figures/02_choropleth_community.html"
echo "    Windows:  start figures/02_choropleth_community.html"
echo ""
echo "  Launch Jupyter Lab for interactive exploration:"
echo "    conda activate chicago-crime && jupyter lab"
echo "============================================================"
