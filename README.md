# Chicago Crime Analysis — A Comprehensive Data Science Project

> **A full-pipeline data science project** covering data ingestion, cleaning, descriptive analytics, geospatial visualisation, time-series modelling (ARIMA/SARIMA), and machine-learning prediction — applied to the Chicago Police Department's publicly available crime dataset (~8.4 million records, 2001–2024).

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Quick-Start (Ubuntu/Linux)](#quick-start-ubuntulinux)
4. [Dataset](#dataset)
5. [Part 1 – Data Cleaning & Processing](#part-1--data-cleaning--processing)
6. [Part 2 – Descriptive Analysis](#part-2--descriptive-analysis)
7. [Part 3 – Predictive Analysis](#part-3--predictive-analysis)
8. [Figures & Visualisations](#figures--visualisations)
9. [Reports & Outputs](#reports--outputs)
10. [Statistical Pitfalls & Bias Disclaimers](#statistical-pitfalls--bias-disclaimers)
11. [Dependencies](#dependencies)
12. [Authors & Licence](#authors--licence)

---

## Project Overview

This repository presents an end-to-end analysis of crime in Chicago, Illinois, using open data published by the City of Chicago Data Portal. The project is structured into three main analytical stages:

| Stage | Script | Notebook | Focus |
|-------|--------|----------|-------|
| **Part 1** | `script_01_cleaning.py` | `notebooks/notebook_01_cleaning.ipynb` | Data ingestion, quality audit, missing-value imputation, feature engineering |
| **Part 2** | `script_02_descriptive.py` | `notebooks/notebook_02_descriptive.ipynb` | 30+ descriptive analyses — temporal, geospatial, statistical, advanced visualisations |
| **Part 3** | `script_03_predictive.py` | `notebooks/notebook_03_predictive.ipynb` | 11 ML tasks — arrest prediction, crime-type classification, volume forecasting, hot-spot detection |

Key headline results:
- **51% crime reduction** from the 2001 peak (482,857) to 2024 (236,450)
- **25.2% overall arrest rate** — proactive crimes approach 100%, reactive property crimes fall below 15%
- **78% ML accuracy** predicting arrests (Random Forest, AUC 0.82)
- **R² = 0.78** for daily crime-count forecasting with XGBoost lag features

---

## Repository Structure

```
chicago-crime-analysis/
│
├── README.md                            ← This file
├── run_all.sh                           ← Run all three scripts end-to-end
├── cleanup.sh                           ← Reset generated files (data, figures, reports)
├── requirements.txt                     ← Python dependencies
├── environment.yml                      ← Conda environment definition
├── LICENSE                              ← MIT Licence
│
├── script_01_cleaning.py                ← Part 1: Data cleaning & feature engineering
├── script_02_descriptive.py            ← Part 2: Descriptive analysis & visualisation
├── script_03_predictive.py             ← Part 3: Predictive modelling (11 ML tasks)
├── generate_heatmap.py                  ← Interactive Folium heatmap (10 layers)
│
├── notebooks/
│   ├── notebook_01_cleaning.ipynb       ← Part 1 as Jupyter notebook
│   ├── notebook_02_descriptive.ipynb    ← Part 2 as Jupyter notebook
│   └── notebook_03_predictive.ipynb     ← Part 3 as Jupyter notebook
│
├── src/                                 ← Shared utility modules
│   ├── __init__.py
│   ├── data_loader.py                   ← Data ingestion helpers
│   ├── data_cleaner.py                  ← Cleaning & imputation logic
│   ├── data_utils.py                    ← General data manipulation helpers
│   ├── feature_engineering.py          ← Feature creation & transformation
│   ├── geo_utils.py                     ← Geospatial helper functions
│   ├── time_series.py                   ← ARIMA / SARIMA helpers
│   ├── time_series_utils.py             ← Additional forecasting utilities
│   ├── ml_models.py                     ← Model training & evaluation
│   ├── ml_utils.py                      ← Cross-validation, SMOTE, metrics
│   └── visualizer.py                    ← Reusable plotting helpers
│
├── data/                                ← Raw and processed datasets (git-ignored)
│   ├── Crimes.csv                       ← Source data from Chicago Data Portal (~1.7 GB)
│   └── Crimes_Cleaned.csv              ← Output of script_01_cleaning.py
│
├── boundaries/                          ← GeoJSON boundary files for Chicago
│   ├── Beat_Boundary.geojson
│   ├── Comm_Boundary.geojson
│   ├── Ward_Boundary.geojson
│   └── District_Boundary.geojson
│
├── figures/                             ← Pipeline-generated figures (git-ignored)
│   ├── 01_missing_values.png            ← Missing value heatmap
│   ├── 01_location_groups.png           ← Location description clustering
│   ├── 02_annual_volume.png             ← Annual crime volume 2001–2024
│   ├── 02_yoy_change.png                ← Year-over-year % change
│   ├── 02_moving_avg.png                ← 12-month rolling average
│   ├── 02_hour.png                      ← Crimes by hour of day
│   ├── 02_dow.png                       ← Crimes by day of week
│   ├── 02_hour_dow_heatmap.png          ← Hour × day heatmap
│   ├── 02_monthly_seasonality.png       ← Monthly seasonality
│   ├── 02_monthly_ts.png                ← Monthly time series
│   ├── 02_top_types.png                 ← Top 15 crime types by volume
│   ├── 02_arrest_rate.png               ← Arrest rate by crime type
│   ├── 02_domestic.png                  ← Domestic crime breakdown
│   ├── 02_hotspots.png                  ← Top crime hotspot blocks
│   ├── 02_district_commarea_ward.png    ← Crime by district / area / ward
│   ├── 02_correlation.png               ← Feature correlation matrix
│   ├── 02_stl_decomp.png                ← STL decomposition
│   ├── 02_acf_pacf.png                  ← ACF & PACF plots
│   ├── 02_sarima_forecast.png           ← SARIMA 12-month forecast
│   ├── 02_choropleth_community.html     ← Interactive community-area choropleth
│   ├── 02_heatmap.html                  ← Interactive Folium density heatmap
│   ├── 03_task1_importance.png          ← ML feature importance (arrest prediction)
│   ├── 03_task1_confusion.png           ← Confusion matrix (arrest prediction)
│   ├── 03_task5_monthly_forecast.png    ← Monthly crime forecast
│   ├── 03_task9_weekend.png             ← Weekend vs weekday model
│   ├── 03_task10_quarterly.png          ← Quarterly count forecast
│   └── 03_task11_xgb_forecast.png       ← XGBoost daily count forecast
│
├── reports/
│   ├── chicago_crime_heatmap.html       ← Self-contained interactive map (10 layers)
│   ├── technical_report.md              ← Full written technical report
│   ├── descriptive_analysis.md          ← Descriptive analysis narrative
│   ├── data_dictionary.md               ← Column-by-column data dictionary
│   └── figures/                         ← Publication-quality report figures
│       ├── B1_annual_trend.png          ← Annual trend (report style)
│       ├── B2_boxplot_season.png        ← Seasonal boxplot
│       ├── B3_crime_types.png           ← Crime type breakdown
│       ├── 01_heatmap.png               ← Hour × day heatmap (report style)
│       ├── 02_violin.png                ← Violin plot — hourly distribution by type
│       ├── 03_correlation.png           ← Correlation matrix (report style)
│       ├── 04_scatter.png               ← Arrest rate vs volume scatter
│       ├── 05_hexbin.png                ← Geospatial hexbin density
│       ├── 06_bubble.png                ← District bubble chart
│       ├── 07_parallel_coords.png       ← Parallel coordinates — crime profiles
│       ├── 08_3d.png                    ← 3D surface — hour × month × volume
│       ├── 09_network.png               ← Crime type spatial co-occurrence wheel
│       ├── 10_sunburst.png              ← Sunburst — type × location hierarchy
│       ├── 11_choropleth.png            ← Police district choropleth map (Voronoi)
│       ├── 12_geo_heatmap.png           ← Annotated geospatial heatmap
│       ├── 13_slopegraph.png            ← Slopegraph — district rank change
│       └── 14_streamgraph.png           ← Streamgraph — crime type mix over time
│
└── docs/
    ├── methodology.md                   ← Step-by-step methodology notes
    └── download_data.sh                 ← Manual data download script
```

---

## Quick-Start (Ubuntu/Linux)

> **Tested on Ubuntu 22.04 LTS with Python 3.10**

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/chicago-crime-analysis.git
cd chicago-crime-analysis
```

### 2. Set up the environment

```bash
# Option A — Conda (recommended)
conda env create -f environment.yml
conda activate chicago-crime

# Option B — pip
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Download the dataset

```bash
bash docs/download_data.sh
# or manually:
wget "https://data.cityofchicago.org/api/views/ijzp-q8t2/rows.csv?accessType=DOWNLOAD" \
     -O data/Crimes.csv
```

### 4a. Run all scripts end-to-end

```bash
bash run_all.sh
```

`run_all.sh` executes all three scripts in sequence and generates all figures, reports, and the interactive heatmap.

### 4b. Run scripts individually

```bash
python script_01_cleaning.py      # ~2 min — outputs data/Crimes_Cleaned.csv
python script_02_descriptive.py   # ~5 min — outputs figures/ and reports/figures/
python script_03_predictive.py    # ~15 min — outputs figures/ and model results
python generate_heatmap.py        # ~1 min — outputs reports/chicago_crime_heatmap.html
```

### 5. (Optional) Open as Jupyter notebooks

```bash
jupyter lab
```

Open notebooks in order: `notebook_01_cleaning.ipynb` → `notebook_02_descriptive.ipynb` → `notebook_03_predictive.ipynb`

### 6. Reset generated files

```bash
bash cleanup.sh
# Preview deletions first (no files removed until confirmed):
bash cleanup.sh --yes   # skip confirmation prompt
```

---

## Dataset

| Property | Value |
|----------|-------|
| Source | [City of Chicago Data Portal](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2) |
| Size (approx.) | ~1.7 GB, ~8.4 million rows |
| Date range | 2001 – 2024 (refreshed daily) |
| Licence | City of Chicago Open Data Licence |

---

## Part 1 – Data Cleaning & Processing

**Script:** `script_01_cleaning.py` | **Notebook:** `notebooks/notebook_01_cleaning.ipynb`

1. Schema audit — column types, cardinality, expected vs. actual value ranges
2. Mixed-type detection in the `Location` column
3. Missing-value quantification with `missingno` visualisations
4. Geospatial imputation using shapely point-in-polygon lookups
5. Duplicate detection on `Case Number` and soft-duplicate analysis
6. Date/time feature engineering (`Hour`, `DayOfWeek`, `Month`, `Year`, `IsWeekend`, `IsHoliday`)
7. Category consolidation of `Location Description` (~300 values → ~15 buckets)
8. Chronological 80/20 train/test split
9. Export to `data/Crimes_Cleaned.csv`

---

## Part 2 – Descriptive Analysis

**Script:** `script_02_descriptive.py` | **Notebook:** `notebooks/notebook_02_descriptive.ipynb`

30+ analyses across five themes:

| Theme | Key outputs |
|-------|-------------|
| **Temporal trends** | Annual volume, YoY change, 12-month moving average, STL decomposition, SARIMA(1,1,1)(1,1,1)[12] forecast |
| **Time-of-day & seasonality** | Hour/day heatmap, monthly seasonality, peak identification |
| **Crime type breakdown** | Top 15 types, arrest rates, domestic flag, parallel coordinates profiles |
| **Geospatial analysis** | District choropleth (Voronoi-tessellated), community area hotspots, geo heatmap with hotspot annotations |
| **Advanced visualisations** | Correlation wheel, sunburst hierarchy, streamgraph, slopegraph, 3D surface, hexbin density |

Notable figures:
- **`09_network.png`** — Spatial co-occurrence wheel showing all 28 crime-type pairs coloured by Pearson r. Battery & Assault cluster together (r = +0.86); Theft & Battery are spatially separated (r = −0.64).
- **`11_choropleth.png`** — Police district choropleth using Voronoi tessellation of CPD district centroids, clipped to Chicago's actual city outline. West and South Side districts dominate in crime volume.

---

## Part 3 – Predictive Analysis

**Script:** `script_03_predictive.py` | **Notebook:** `notebooks/notebook_03_predictive.ipynb`

11 ML prediction tasks:

| Task | Target | Best Model | Accuracy | AUC |
|------|--------|-----------|----------|-----|
| 1 | Arrest prediction | Random Forest | 78% | 0.82 |
| 2 | Arrest by location type | XGBoost | 75.1% | 0.78 |
| 3 | Domestic crime flag | Logistic Regression | 81.2% | 0.86 |
| 4 | Crime type (top 10) | XGBoost | 64.3% | 0.61 |
| 5 | Monthly crime forecast | SARIMA + Prophet | — | — |
| 6 | Hot-beat classification | Random Forest | 81.4% | 0.87 |
| 7 | Night-time crime flag | Logistic Regression | 72.3% | 0.76 |
| 8 | Crime severity regression | Ridge Regression | — | R²=0.21 |
| 9 | Weekend vs. weekday | XGBoost | 68.5% | 0.72 |
| 10 | Quarterly count forecast | XGBoost | — | R²=0.65 |
| 11 | Daily count (lag features) | XGBoost | — | R²=0.78 |

All classification metrics from stratified k-fold cross-validation. SMOTE applied inside each fold to prevent data leakage.

---

## Figures & Visualisations

All figures are auto-generated by the scripts and saved into two locations:

- **`figures/`** — pipeline figures (intermediate and exploratory)
- **`reports/figures/`** — 17 publication-quality figures used in the executive report and presentation

The interactive heatmap (`reports/chicago_crime_heatmap.html`) is a self-contained single HTML file with 10 toggle-able layers including crime density KDE, community area choropleth, district boundaries, beat boundaries, arrest rate map, night-crime overlay, and top-10 hotspot pins. Opens in any browser with no internet connection required.

---

## Reports & Outputs

| File | Description |
|------|-------------|
| `reports/technical_report.md` | Full written technical report |
| `reports/descriptive_analysis.md` | Descriptive analysis narrative with embedded findings |
| `reports/data_dictionary.md` | Column-by-column data dictionary |
| `reports/chicago_crime_heatmap.html` | Interactive 10-layer Folium map |
| `chicago_crime_analysis.pptx` | Executive presentation (26 slides) |

---

## Statistical Pitfalls & Bias Disclaimers

- **Reporting Bias**: Data reflects *reported* crimes only; under-reporting varies across communities and crime types.
- **Truncation Bias**: Recent months have fewer finalised records; exclude the last 1–3 months from trend analyses.
- **Spatial Autocorrelation**: Neighbouring areas are not independent; standard regression may produce inflated significance estimates.
- **ARIMA Stationarity**: ADF test applied before model selection; forecasts beyond 12 months carry substantial uncertainty.
- **Class Imbalance**: Arrest and crime-type classifiers use stratified k-fold and SMOTE oversampling; F1 and AUC-ROC reported alongside accuracy.
- **Confounding Variables**: Weather–crime correlation does not establish causation; correlated variables (day length, school calendar, economy) are omitted confounders.
- **Temporal Data Leakage**: All ML splits respect chronological ordering to prevent future information leaking into training.
- **Voronoi District Boundaries**: The choropleth map uses Voronoi tessellation of CPD district centroids as a boundary approximation. For precise district-level analysis use the official `District_Boundary.geojson` file.

---

## Dependencies

See `requirements.txt` and `environment.yml`. Key packages:

| Category | Packages |
|----------|---------|
| Data | `pandas`, `numpy`, `pyarrow` |
| Visualisation | `matplotlib`, `seaborn`, `plotly` |
| Geospatial | `folium`, `geopandas`, `shapely`, `scipy` |
| Time series | `statsmodels`, `prophet` |
| Machine learning | `scikit-learn`, `xgboost`, `imbalanced-learn` |
| Utilities | `missingno`, `jupyterlab`, `tqdm` |

---

## Authors & Licence

Code released under the **MIT Licence**. Dataset published by the City of Chicago under its [Open Data Licence](https://www.chicago.gov/city/en/narr/foia/data_disclaimer.html).
