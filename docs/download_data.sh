#!/usr/bin/env bash
# =============================================================================
# download_data.sh — Download the Chicago Crimes dataset and boundary files
# =============================================================================
# Usage: bash docs/download_data.sh
# Run from the project root directory.
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"
BOUNDARIES_DIR="$PROJECT_ROOT/boundaries"
PROCESSED_DIR="$DATA_DIR/processed"

echo "=== Chicago Crime Analysis — Data Download Script ==="
echo "Project root: $PROJECT_ROOT"

# ── Create directories ───────────────────────────────────────────────────────
mkdir -p "$DATA_DIR" "$PROCESSED_DIR" "$BOUNDARIES_DIR"

# ── Download Crimes.csv ──────────────────────────────────────────────────────
CRIMES_URL="https://data.cityofchicago.org/api/views/ijzp-q8t2/rows.csv?accessType=DOWNLOAD"
CRIMES_FILE="$DATA_DIR/Crimes.csv"

if [[ -f "$CRIMES_FILE" ]]; then
    SIZE=$(du -sh "$CRIMES_FILE" | cut -f1)
    echo "[SKIP] Crimes.csv already exists ($SIZE). Delete it to re-download."
else
    echo "[DOWNLOAD] Fetching Crimes.csv (~1.7 GB — this may take several minutes)..."
    wget --progress=bar:force \
         --retry-connrefused \
         --tries=3 \
         -O "$CRIMES_FILE" \
         "$CRIMES_URL"
    echo "[OK] Crimes.csv downloaded → $CRIMES_FILE"
fi

# ── Download GeoJSON boundary files ─────────────────────────────────────────
declare -A GEOJSONS=(
    ["Beat_Boundary.geojson"]="https://data.cityofchicago.org/api/geospatial/aerh-rz74?method=export&type=GeoJSON"
    ["Ward_Boundary.geojson"]="https://data.cityofchicago.org/api/geospatial/sp34-6z76?method=export&type=GeoJSON"
    ["Comm_Boundary.geojson"]="https://data.cityofchicago.org/api/geospatial/cauq-8yn6?method=export&type=GeoJSON"
)

for FNAME in "${!GEOJSONS[@]}"; do
    FPATH="$BOUNDARIES_DIR/$FNAME"
    if [[ -f "$FPATH" ]]; then
        echo "[SKIP] $FNAME already exists."
    else
        echo "[DOWNLOAD] Fetching $FNAME..."
        wget --progress=bar:force \
             --retry-connrefused \
             --tries=3 \
             -O "$FPATH" \
             "${GEOJSONS[$FNAME]}" || echo "[WARN] Could not download $FNAME — you may need to download it manually."
        echo "[OK] $FNAME → $FPATH"
    fi
done

echo ""
echo "=== Download complete. ==="
echo "Next step: run the notebooks in order, starting with:"
echo "  jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_acquisition.ipynb"
