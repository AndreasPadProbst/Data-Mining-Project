# Chicago Crime Analysis — A Comprehensive Data Science Project

> **A full-pipeline data science project** covering data ingestion, cleaning, descriptive analytics, geospatial visualisation, time-series modelling (ARIMA/SARIMA), and machine-learning prediction — applied to the Chicago Police Department's publicly available crime dataset (~7 million records, 2001–present).

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Quick-Start (Ubuntu/Linux)](#quick-start-ubuntulinux)
4. [Dataset](#dataset)
5. [Part 1 – Data Cleaning & Processing](#part-1--data-cleaning--processing)
6. [Part 2 – Descriptive Analysis](#part-2--descriptive-analysis)
7. [Part 3 – Predictive Analysis](#part-3--predictive-analysis)
8. [Statistical Pitfalls & Bias Disclaimers](#statistical-pitfalls--bias-disclaimers)
9. [Dependencies](#dependencies)
10. [Authors & Licence](#authors--licence)

---

## Project Overview

This repository presents an end-to-end analysis of crime in Chicago, Illinois, using open data published by the City of Chicago Data Portal. The project is structured into three main analytical stages:

| Stage | Focus |
|-------|-------|
| **Part 1** | Data ingestion, quality audit, missing-value imputation, and feature engineering |
| **Part 2** | 30+ descriptive analyses — time-series, seasonal decomposition, ARIMA/SARIMA forecasting, geospatial choropleth and heat maps, correlation analysis |
| **Part 3** | Machine-learning models predicting arrest likelihood, crime type, hot-spot beats, daily crime volume, and more (10+ tasks) |

All notebooks are self-contained, heavily commented, and reproducible on a fresh Ubuntu 22.04 machine with the single `setup.sh` script provided.

---

## Repository Structure

```
chicago-crime-analysis/
│
├── README.md                          ← This file
├── setup.sh                           ← One-command environment setup + data download
├── requirements.txt                   ← Python dependencies
├── LICENSE                            ← MIT Licence
│
├── data/                              ← Raw and processed datasets (git-ignored for size)
│   └── .gitkeep
│
├── boundaries/                        ← GeoJSON boundary files for Chicago
│   ├── Beat_Boundary.geojson
│   ├── Comm_Boundary.geojson
│   ├── Ward_Boundary.geojson
│   └── District_Boundary.geojson
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb         ← Part 1: Cleaning & feature engineering
│   ├── 02_descriptive_analysis.ipynb  ← Part 2: 30+ descriptive analyses
│   └── 03_predictive_analysis.ipynb   ← Part 3: ML prediction tasks
│
├── src/
│   ├── __init__.py
│   ├── data_utils.py                  ← Reusable data loading / cleaning helpers
│   ├── geo_utils.py                   ← Geospatial helper functions
│   ├── time_series_utils.py           ← ARIMA / forecasting helpers
│   └── ml_utils.py                    ← Model training / evaluation helpers
│
├── figures/                           ← Auto-generated plots (git-ignored)
│   └── .gitkeep
│
├── reports/
│   ├── technical_report.md            ← Full written technical report
│   └── data_dictionary.md             ← Column-by-column data dictionary
│
└── docs/
    └── methodology.md                 ← Step-by-step methodology notes
```

---

## Quick-Start (Ubuntu/Linux)

> **Tested on Ubuntu 22.04 LTS with Python 3.10**

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/chicago-crime-analysis.git
cd chicago-crime-analysis
```

### 2. Run the automated setup

```bash
chmod +x setup.sh
./setup.sh
```

`setup.sh` will:
- Install system dependencies (`libspatialindex-dev`, `libgeos-dev`)
- Create and activate a Python virtual environment (`venv/`)
- Install all Python packages from `requirements.txt`
- Download the Crimes.csv dataset (~1.7 GB) into `data/`
- Download GeoJSON boundary files

### 3. Launch Jupyter

```bash
source venv/bin/activate
jupyter lab
```

Open notebooks in order:
1. `notebooks/01_data_cleaning.ipynb`
2. `notebooks/02_descriptive_analysis.ipynb`
3. `notebooks/03_predictive_analysis.ipynb`

### 4. (Optional) Run headlessly as Python scripts

```bash
source venv/bin/activate
jupyter nbconvert --to script notebooks/01_data_cleaning.ipynb --output src/run_01
python src/run_01.py
```

---

## Dataset

| Property | Value |
|----------|-------|
| Source | [City of Chicago Data Portal](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2) |
| Download URL | See `setup.sh` |
| Size (approx.) | ~1.7 GB, ~7.7 million rows |
| Date range | 2001 – present (refreshed daily) |
| Licence | City of Chicago Open Data Licence |

Download manually if `setup.sh` is not used:

```bash
wget "https://data.cityofchicago.org/api/views/ijzp-q8t2/rows.csv?accessType=DOWNLOAD" \
     -O data/Crimes.csv
```

---

## Part 1 – Data Cleaning & Processing

Notebook: `notebooks/01_data_cleaning.ipynb`

Key steps:
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

Notebook: `notebooks/02_descriptive_analysis.ipynb`

30+ analyses including annual trends, seasonal decomposition, ARIMA/SARIMA forecasting, geospatial choropleth maps, heat maps, correlation analysis, and crime-type breakdowns.

---

## Part 3 – Predictive Analysis

Notebook: `notebooks/03_predictive_analysis.ipynb`

10+ ML prediction tasks covering arrest prediction, crime-type classification, daily/monthly crime volume regression, hot-spot beat identification, and weather-driven crime prediction.

---

## Statistical Pitfalls & Bias Disclaimers

- **Reporting Bias**: Data reflects *reported* crimes only; under-reporting varies across communities and crime types.
- **Truncation Bias**: Recent months have fewer finalised records; exclude the last 1–3 months from trend analyses.
- **Spatial Autocorrelation**: Neighbouring areas are not independent; standard regression may produce inflated significance estimates.
- **ARIMA Stationarity**: ADF test applied before model selection; forecasts beyond 12 months carry substantial uncertainty.
- **Class Imbalance**: Arrest and crime-type classifiers use stratified k-fold and SMOTE oversampling; F1 and AUC-ROC reported alongside accuracy.
- **Confounding Variables**: Weather–crime correlation does not establish causation; correlated variables (day length, school calendar, economy) are omitted confounders.
- **Temporal Data Leakage**: All ML splits respect chronological ordering to prevent future information leaking into training.

---

## Dependencies

See `requirements.txt`. Key packages: `pandas`, `numpy`, `matplotlib`, `seaborn`, `folium`, `geopandas`, `shapely`, `statsmodels`, `scikit-learn`, `xgboost`, `imbalanced-learn`, `missingno`, `jupyterlab`.

---

## Authors & Licence

Code released under the **MIT Licence**. Dataset published by the City of Chicago under its [Open Data Licence](https://www.chicago.gov/city/en/narr/foia/data_disclaimer.html).
