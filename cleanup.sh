#!/usr/bin/env bash
# =============================================================================
# cleanup.sh — Chicago Crime Analysis — Reset Project to Fresh State
#
# Removes all generated files so you can run the pipeline from scratch.
# Your original source files (scripts, notebooks, boundaries, src/) are
# never touched.
#
# Run from inside your chicago-crime-analysis/ folder:
#     chmod +x cleanup.sh
#     ./cleanup.sh
#
# To skip the confirmation prompt (e.g. in automated runs):
#     ./cleanup.sh --yes
# =============================================================================

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'

log()  { echo -e "${BLUE}[INFO]${NC}  $1"; }
ok()   { echo -e "${GREEN}[OK]${NC}    $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $1"; }
gone() { echo -e "${RED}[DEL]${NC}   $1"; }

echo ""
echo -e "${BOLD}============================================================${NC}"
echo -e "${BOLD}   Chicago Crime Analysis — Project Cleanup${NC}"
echo -e "${BOLD}============================================================${NC}"
echo ""

# ── Confirm we are in the right folder ───────────────────────────────────────
if [ ! -f "run_all.sh" ] && [ ! -f "script_01_cleaning.py" ]; then
    echo -e "${RED}[ERROR]${NC} This doesn't look like the project folder."
    echo "        Please cd into chicago-crime-analysis/ first."
    exit 1
fi

# ── Preview what will be deleted ─────────────────────────────────────────────
echo -e "${YELLOW}The following will be permanently deleted:${NC}"
echo ""
echo "  DATA FILES"
echo "    data/Crimes_Cleaned.csv"
echo "    data/train.csv"
echo "    data/test.csv"
echo ""
echo "  FIGURES"
echo "    figures/*.png  (all chart images)"
echo "    figures/*.html (interactive maps)"
echo ""
echo "  REPORTS"
echo "    reports/report_embedded.html"
echo "    reports/report_embedded.pdf"
echo "    reports/chicago_crime_heatmap.html"
echo ""
echo "  TEMPORARY FILES"
echo "    .nb_scripts/   (nbconvert temp folder)"
echo "    **/__pycache__/ (Python bytecode cache)"
echo "    **/*.pyc        (compiled Python files)"
echo "    **/*.pyo        (optimised Python files)"
echo ""
echo -e "${GREEN}The following will NOT be touched:${NC}"
echo "    data/Crimes.csv           (your original dataset)"
echo "    boundaries/*.geojson      (boundary files)"
echo "    src/                      (source code)"
echo "    notebooks/                (Jupyter notebooks)"
echo "    scripts (run_all.sh, script_0*.py, etc.)"
echo "    reports/descriptive_analysis.md"
echo "    reports/codebase_walkthrough.docx"
echo "    reports/technical_report.md"
echo "    reports/data_dictionary.md"
echo ""

# ── Confirmation prompt (skip with --yes flag) ────────────────────────────────
if [[ "$1" != "--yes" ]]; then
    echo -e "${YELLOW}${BOLD}Are you sure you want to delete all generated files? [y/N]${NC} "
    read -r answer
    if [[ "$answer" != "y" && "$answer" != "Y" ]]; then
        echo ""
        echo "  Cancelled — nothing was deleted."
        echo ""
        exit 0
    fi
fi

echo ""
echo -e "${BOLD}============================================================${NC}"
echo -e "${BOLD}  Deleting...${NC}"
echo -e "${BOLD}============================================================${NC}"

DELETED=0

# ── Helper: delete a file if it exists ───────────────────────────────────────
del_file() {
    if [ -f "$1" ]; then
        rm -f "$1"
        gone "$1"
        DELETED=$((DELETED + 1))
    fi
}

# ── Helper: delete a folder if it exists ─────────────────────────────────────
del_dir() {
    if [ -d "$1" ]; then
        rm -rf "$1"
        gone "$1/"
        DELETED=$((DELETED + 1))
    fi
}

# ── Data files ────────────────────────────────────────────────────────────────
echo ""
log "Removing generated CSV files ..."
del_file "data/Crimes_Cleaned.csv"
del_file "data/train.csv"
del_file "data/test.csv"

# ── Figures — PNG charts ──────────────────────────────────────────────────────
echo ""
log "Removing generated figures ..."
if [ -d "figures" ]; then
    PNG_COUNT=$(find figures -name "*.png" 2>/dev/null | wc -l)
    HTML_COUNT=$(find figures -name "*.html" 2>/dev/null | wc -l)
    if [ "$PNG_COUNT" -gt 0 ] || [ "$HTML_COUNT" -gt 0 ]; then
        find figures -name "*.png"  -delete
        find figures -name "*.html" -delete
        gone "figures/*.png  ($PNG_COUNT files)"
        gone "figures/*.html ($HTML_COUNT files)"
        DELETED=$((DELETED + PNG_COUNT + HTML_COUNT))
    else
        warn "figures/ exists but no PNG/HTML files found"
    fi
else
    warn "figures/ folder does not exist — nothing to remove"
fi

# ── Reports — generated HTML and PDF ─────────────────────────────────────────
echo ""
log "Removing generated report files ..."
del_file "reports/report_embedded.html"
del_file "reports/report_embedded.pdf"
del_file "reports/chicago_crime_heatmap.html"

# ── Temporary nbconvert folder ────────────────────────────────────────────────
echo ""
log "Removing temporary files ..."
del_dir ".nb_scripts"

# ── Python cache files ────────────────────────────────────────────────────────
echo ""
log "Removing Python cache files ..."
CACHE_COUNT=0

# __pycache__ folders
while IFS= read -r -d '' dir; do
    rm -rf "$dir"
    gone "$dir"
    CACHE_COUNT=$((CACHE_COUNT + 1))
done < <(find . -type d -name "__pycache__" \
         -not -path "./venv/*" \
         -not -path "./.git/*" \
         -print0 2>/dev/null)

# .pyc files not inside __pycache__
while IFS= read -r -d '' file; do
    rm -f "$file"
    gone "$file"
    CACHE_COUNT=$((CACHE_COUNT + 1))
done < <(find . -name "*.pyc" -o -name "*.pyo" \
         -not -path "./venv/*" \
         -not -path "./.git/*" \
         -print0 2>/dev/null)

if [ "$CACHE_COUNT" -eq 0 ]; then
    warn "No Python cache files found"
else
    DELETED=$((DELETED + CACHE_COUNT))
fi

# ── Jupyter checkpoints ───────────────────────────────────────────────────────
echo ""
log "Removing Jupyter checkpoint folders ..."
CHECKPOINT_COUNT=0
while IFS= read -r -d '' dir; do
    rm -rf "$dir"
    gone "$dir"
    CHECKPOINT_COUNT=$((CHECKPOINT_COUNT + 1))
done < <(find . -type d -name ".ipynb_checkpoints" \
         -not -path "./venv/*" \
         -print0 2>/dev/null)

if [ "$CHECKPOINT_COUNT" -eq 0 ]; then
    warn "No Jupyter checkpoints found"
else
    DELETED=$((DELETED + CHECKPOINT_COUNT))
fi

# ── Verify Crimes.csv is still intact ────────────────────────────────────────
echo ""
log "Verifying original dataset is intact ..."
if [ -f "data/Crimes.csv" ]; then
    SIZE=$(du -sh "data/Crimes.csv" | cut -f1)
    ok "data/Crimes.csv is safe ($SIZE)"
else
    warn "data/Crimes.csv not found — you will need to provide it before re-running"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}============================================================${NC}"
echo -e "${GREEN}${BOLD}  CLEANUP COMPLETE${NC}"
echo -e "${BOLD}============================================================${NC}"
echo ""
echo -e "  Deleted ${BOLD}${DELETED}${NC} files/folders"
echo ""
echo "  To run the project from scratch:"
echo ""
echo "    conda activate chicago-crime"
echo "    python script_01_cleaning.py"
echo "    python script_02_descriptive.py"
echo "    python script_03_predictive.py"
echo "    python generate_heatmap.py"
echo "    python build_report.py"
echo ""
echo "  Or run everything in one go:"
echo ""
echo "    ./run_all.sh"
echo ""
echo -e "${BOLD}============================================================${NC}"
