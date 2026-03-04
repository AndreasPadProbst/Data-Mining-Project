# Methodology — Step-by-Step Guide

This document explains how the project code operates and describes the data transformations at each step. It is intended as a companion to the Jupyter notebooks for readers who want to understand the design decisions in depth.

---

## Step 1: Environment Setup (`setup.sh`)

`setup.sh` automates the following:

1. **System packages** — `apt-get` installs `libspatialindex-dev` (required for the Rtree spatial index used by geopandas) and `libgeos-dev` (GEOS geometry engine, a shapely dependency).
2. **Virtual environment** — `python3 -m venv venv` creates an isolated Python environment. This prevents dependency conflicts with any system-wide packages.
3. **Python packages** — `pip install -r requirements.txt` installs all pinned dependencies. Pinned versions ensure reproducibility across machines and over time.
4. **Dataset download** — `wget` downloads Crimes.csv from the City of Chicago Data Portal. The file is ~1.7 GB and may take 5–20 minutes depending on connection speed.
5. **GeoJSON boundaries** — Four boundary files are downloaded from the Data Portal's GeoJSON API: beat, community area, ward, and district polygons. These are used for both visualisation and spatial imputation.

---

## Step 2: Data Loading (`src/data_utils.py → load_crime_data`)

- Reads Crimes.csv with `pandas.read_csv(..., low_memory=False)`.
- The `low_memory=False` flag forces pandas to read the entire file before inferring dtypes, which suppresses the DtypeWarning on the mixed-type Location column at the cost of higher RAM usage (~8 GB for the full file).
- Column presence is validated against `REQUIRED_COLUMNS`.
- Returns a raw DataFrame with original column names.

---

## Step 3: Quality Audit (`audit_quality`)

- Iterates all columns and computes: dtype, missing count, missing %, unique count, most frequent value.
- Returns a summary DataFrame that is displayed in the notebook before any cleaning is applied. This provides a baseline view of the data's quality state.

---

## Step 4: Data Cleaning (`clean_crime_data`)

Transformations applied in sequence:

| Order | Step | Rationale |
|-------|------|-----------|
| 1 | Drop NaN Location rows | Float-type Location entries are confirmed NaN rows — they carry no geographic information |
| 2 | Rename Community Area → Community_Area | Python attribute access requires no spaces |
| 3 | Drop missing coordinates | Rows without Lat/Lon cannot be used in spatial analyses |
| 4 | Deduplicate on Case Number | Each CPD case number should appear once in the dataset |
| 5 | Parse Date to datetime | Required for time-series operations and feature extraction |
| 6 | Cast Arrest/Domestic to bool | These are binary flags; object dtype wastes memory |
| 7 | Cast spatial IDs to Int64 | Int64 (nullable integer) handles remaining NaN values in Ward/Beat/District |
| 8 | Strip whitespace | Removes leading/trailing spaces that cause spurious category splits |

---

## Step 5: Feature Engineering (`engineer_features`)

All new columns are derived deterministically from existing columns:

- **Temporal**: `dt.hour`, `dt.dayofweek`, `dt.month`, `dt.year`, `dt.quarter` — direct pandas datetime accessors.
- **IsWeekend**: `DayOfWeek.isin([5, 6])` — Saturday = 5, Sunday = 6 in Python's ISO weekday convention.
- **IsHoliday**: Uses the `holidays.UnitedStates()` object to perform a set-membership lookup for each date. This covers 10 federal holidays per year.
- **Season**: A deterministic mapping from month number to meteorological season.
- **LocationGrouped**: Dictionary-based string mapping using `.map()`. Unmapped values fall into the "Other" bucket.

---

## Step 6: Geospatial Imputation (`src/geo_utils.py → impute_spatial_columns`)

For rows with valid coordinates but missing Ward/Beat/Community Area:

1. Load GeoJSON boundary files as GeoDataFrames (geopandas).
2. Project to EPSG:4326 (WGS84) if not already.
3. For each row needing imputation, create a shapely `Point(longitude, latitude)`.
4. Test the point against each polygon in the boundary GeoDataFrame using `geometry.contains(point)`.
5. Assign the polygon's area identifier to the missing column.

This is a computationally expensive O(N × P) operation where N = rows to impute and P = number of polygons. For the full dataset, running on only the rows with missing values (< 2% of records) is essential for performance.

---

## Step 7: Train/Test Split (`train_test_split_temporal`)

- Sort by `Date` ascending.
- Take the first 80% of rows as training data, the remaining 20% as test.
- No random shuffling is ever applied — this would constitute temporal data leakage.
- The cutoff date naturally falls around 2018–2019 depending on the dataset snapshot.

---

## Step 8: Time-Series Analysis (`src/time_series_utils.py`)

### Aggregation
`df.set_index("Date").resample("M").size()` — groups all records by month-end date and counts them. This produces a regular monthly time-series suitable for ARIMA modelling.

### Stationarity
The ADF test H₀ is "unit root exists" (non-stationary). We reject H₀ at p < 0.05. If the series is non-stationary, we apply first-order differencing (`diff(1)`) and re-test.

### Model Selection
`pmdarima.auto_arima` performs stepwise search over ARIMA(p, d, q)(P, D, Q)[12] orders, using AIC as the selection criterion. Stepwise search starts from a simple model and adds terms greedily, which is much faster than exhaustive grid search.

### Cross-Validation
Expanding-window CV: in fold k, train on data up to month M_k, evaluate on the next 12 months. This mirrors the real-world scenario where a model is retrained each time new data arrives.

---

## Step 9: Machine Learning (`src/ml_utils.py`)

### Feature Preparation
Numeric columns are passed through directly. Categorical columns are label-encoded with `sklearn.preprocessing.LabelEncoder`. Label encoding is appropriate for tree-based models; for linear models, one-hot encoding should be used instead (via `pd.get_dummies`).

### Class Imbalance Handling
SMOTE (Synthetic Minority Over-sampling Technique) generates synthetic minority-class samples by interpolating between existing minority samples in feature space. It is applied inside each training fold only, never to the combined dataset or the test fold.

### Evaluation
- **Classifiers**: Stratified k-fold ensures each fold has the same class distribution as the full dataset. Metrics: accuracy, F1-macro, ROC-AUC.
- **Regressors**: Standard k-fold. Metrics: MAE, RMSE, R².
- All metrics are reported as mean ± std across folds.

---

## Running Order

Always run notebooks in this order:
1. `01_data_cleaning.ipynb` — produces `data/Crimes_Cleaned.csv`
2. `02_descriptive_analysis.ipynb` — reads `data/Crimes_Cleaned.csv`, saves plots to `figures/`
3. `03_predictive_analysis.ipynb` — reads `data/Crimes_Cleaned.csv`, saves model outputs to `figures/`

If `data/Crimes_Cleaned.csv` does not exist, notebooks 2 and 3 will fail at the data-loading cell.
