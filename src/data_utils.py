"""
data_utils.py
=============
Reusable helpers for loading, cleaning, and feature-engineering the
Chicago crime dataset.

All public functions follow a consistent convention:
  - Accept a DataFrame and return a (modified) DataFrame.
  - Never mutate the input in place; always work on a copy.
  - Raise informative ValueError / KeyError when required columns are absent.

Typical usage
-------------
>>> from src.data_utils import load_crime_data, clean_crime_data, engineer_features
>>> df_raw  = load_crime_data("data/Crimes.csv")
>>> df_clean = clean_crime_data(df_raw)
>>> df_feat  = engineer_features(df_clean)
"""

import os
import logging
from typing import Optional, List

import numpy as np
import pandas as pd
import holidays

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Columns that must be present in the raw CSV
REQUIRED_COLUMNS: List[str] = [
    "ID", "Case Number", "Date", "Block", "IUCR",
    "Primary Type", "Description", "Location Description",
    "Arrest", "Domestic", "Beat", "District", "Ward",
    "Community Area", "FBI Code", "Year",
    "Latitude", "Longitude", "Location",
]

# Raw Location Description values → grouped bucket labels
LOCATION_GROUP_MAP: dict = {
    "STREET":                   "Street / Alley",
    "ALLEY":                    "Street / Alley",
    "SIDEWALK":                 "Street / Alley",
    "RESIDENCE":                "Residential",
    "APARTMENT":                "Residential",
    "HOUSE":                    "Residential",
    "RESIDENCE-GARAGE":         "Residential",
    "DRIVEWAY - RESIDENTIAL":   "Residential",
    "VEHICLE NON-COMMERCIAL":   "Vehicle",
    "AUTO":                     "Vehicle",
    "TAXICAB":                  "Vehicle",
    "COMMERCIAL / BUSINESS OFFICE": "Commercial",
    "SMALL RETAIL STORE":       "Commercial",
    "DEPARTMENT STORE":         "Commercial",
    "GAS STATION":              "Commercial",
    "RESTAURANT":               "Commercial",
    "GROCERY FOOD STORE":       "Commercial",
    "BANK":                     "Financial / Government",
    "CURRENCY EXCHANGE":        "Financial / Government",
    "GOVERNMENT BUILDING/PROPERTY": "Financial / Government",
    "SCHOOL, PUBLIC, BUILDING": "School / University",
    "SCHOOL, PUBLIC, GROUNDS":  "School / University",
    "UNIVERSITY - GROUNDS":     "School / University",
    "CTA PLATFORM":             "Transit",
    "CTA TRAIN":                "Transit",
    "CTA BUS":                  "Transit",
    "PARKING LOT/GARAGE(NON.RESID.)": "Parking",
    "PARK PROPERTY":            "Park / Open Space",
    "FOREST PRESERVE":          "Park / Open Space",
    "BAR OR TAVERN":            "Entertainment",
    "ATHLETIC CLUB":            "Entertainment",
    "HOTEL/MOTEL":              "Hotel",
}


# ── Public functions ──────────────────────────────────────────────────────────

def load_crime_data(filepath: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Load the Chicago Crimes CSV into a pandas DataFrame.

    Parameters
    ----------
    filepath : str
        Path to Crimes.csv (e.g. ``"data/Crimes.csv"``).
    nrows : int, optional
        If provided, load only the first *nrows* rows — useful for quick
        exploratory work without loading the full ~1.7 GB file.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame with original column names and dtypes.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If expected columns are absent from the CSV.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'. "
            "Run setup.sh or download Crimes.csv manually."
        )

    log.info("Loading dataset from '%s' ...", filepath)
    df = pd.read_csv(
        filepath,
        low_memory=False,     # Suppress mixed-type dtype warnings during initial read
        nrows=nrows,
    )
    log.info("Loaded %d rows × %d columns.", len(df), df.shape[1])

    # Validate expected columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"The following expected columns are absent from the CSV: {missing_cols}"
        )

    return df


def audit_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a column-level data-quality summary.

    For each column, reports:
      - dtype
      - total missing values and missing %
      - number of unique values
      - a sample of the most frequent value

    Parameters
    ----------
    df : pd.DataFrame
        Any DataFrame (raw or cleaned).

    Returns
    -------
    pd.DataFrame
        Quality summary with one row per column.
    """
    rows = []
    n = len(df)
    for col in df.columns:
        n_missing = df[col].isna().sum()
        n_unique  = df[col].nunique(dropna=True)
        try:
            top_val = df[col].value_counts(dropna=True).idxmax()
        except Exception:
            top_val = None
        rows.append({
            "column":      col,
            "dtype":       str(df[col].dtype),
            "n_missing":   int(n_missing),
            "pct_missing": round(100 * n_missing / n, 2),
            "n_unique":    int(n_unique),
            "top_value":   str(top_val),
        })
    return pd.DataFrame(rows).set_index("column")


def clean_crime_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps to the raw crime DataFrame.

    Steps applied (in order):
    1.  Remove rows where ``Location`` is NaN (mixed float/str column).
    2.  Standardise column name ``Community Area`` → ``Community_Area``.
    3.  Drop rows with missing ``Latitude`` / ``Longitude``.
    4.  Drop exact duplicate rows (same ``Case Number``).
    5.  Parse ``Date`` to datetime.
    6.  Cast ``Arrest`` and ``Domestic`` to bool.
    7.  Cast ``Ward``, ``Beat``, ``District``, ``Community_Area`` to Int64
        (nullable integer — handles remaining NaN values gracefully).
    8.  Strip whitespace from string columns.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame as returned by :func:`load_crime_data`.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame. The index is reset.

    Notes
    -----
    Rows with missing ``Latitude``/``Longitude`` account for < 1 % of the
    full dataset and are dropped rather than imputed to avoid introducing
    spatial error into geospatial analyses. An alternative imputation
    approach using shapely point-in-polygon is provided in geo_utils.py.
    """
    df = df.copy()
    initial_rows = len(df)

    # 1. Remove rows where Location is NaN (float NaN rather than a coordinate string)
    df = df[df["Location"].apply(lambda x: isinstance(x, str))]
    log.info("After dropping NaN Location: %d rows (removed %d).",
             len(df), initial_rows - len(df))

    # 2. Rename 'Community Area' → 'Community_Area' for easier attribute access
    if "Community Area" in df.columns:
        df = df.rename(columns={"Community Area": "Community_Area"})

    # 3. Drop rows missing both Latitude and Longitude
    before = len(df)
    df = df.dropna(subset=["Latitude", "Longitude"])
    log.info("After dropping missing coordinates: %d rows (removed %d).",
             len(df), before - len(df))

    # 4. Drop exact duplicate Case Numbers (keep first occurrence)
    before = len(df)
    df = df.drop_duplicates(subset=["Case Number"], keep="first")
    log.info("After deduplication: %d rows (removed %d).",
             len(df), before - len(df))

    # 5. Parse Date column to datetime
    df["Date"] = pd.to_datetime(df["Date"], infer_datetime_format=True, errors="coerce")
    n_bad_dates = df["Date"].isna().sum()
    if n_bad_dates:
        log.warning("%d rows had unparseable dates and will be dropped.", n_bad_dates)
        df = df.dropna(subset=["Date"])

    # 6. Cast boolean columns
    for col in ["Arrest", "Domestic"]:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    # 7. Cast integer columns (nullable Int64 allows remaining NaN values)
    for col in ["Ward", "Beat", "District", "Community_Area", "Year"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # 8. Strip leading/trailing whitespace from object columns
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    df = df.reset_index(drop=True)
    log.info("Cleaning complete. Final shape: %d rows × %d columns.", *df.shape)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive temporal, categorical, and binary features from the cleaned DataFrame.

    New columns created
    -------------------
    Hour           : int — hour of day (0–23)
    DayOfWeek      : int — day of week (0=Monday … 6=Sunday)
    DayOfWeekName  : str — e.g. "Monday"
    Month          : int — month (1–12)
    MonthName      : str — e.g. "January"
    YearActual     : int — 4-digit year extracted from Date
    IsWeekend      : bool — True if DayOfWeek ∈ {5, 6}
    IsHoliday      : bool — True if date falls on a US federal holiday
    Quarter        : int — fiscal quarter (1–4)
    Season         : str — "Winter", "Spring", "Summer", or "Fall"
    LocationGrouped: str — grouped Location Description bucket

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame as returned by :func:`clean_crime_data`.
        Must contain a parsed ``Date`` column.

    Returns
    -------
    pd.DataFrame
        DataFrame with all original columns plus the new feature columns.
    """
    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        raise ValueError("'Date' column must be datetime. Run clean_crime_data first.")

    # ── Temporal features ─────────────────────────────────────────────────────
    df["Hour"]        = df["Date"].dt.hour
    df["DayOfWeek"]   = df["Date"].dt.dayofweek          # 0=Monday
    df["DayOfWeekName"] = df["Date"].dt.day_name()
    df["Month"]       = df["Date"].dt.month
    df["MonthName"]   = df["Date"].dt.month_name()
    df["YearActual"]  = df["Date"].dt.year
    df["Quarter"]     = df["Date"].dt.quarter
    df["IsWeekend"]   = df["DayOfWeek"].isin([5, 6])

    # ── Season ────────────────────────────────────────────────────────────────
    def _month_to_season(month: int) -> str:
        """Map a month number to a meteorological season."""
        if month in (12, 1, 2):
            return "Winter"
        if month in (3, 4, 5):
            return "Spring"
        if month in (6, 7, 8):
            return "Summer"
        return "Fall"

    df["Season"] = df["Month"].map(_month_to_season)

    # ── US federal holidays ───────────────────────────────────────────────────
    us_holidays = holidays.UnitedStates()
    dates_as_date = df["Date"].dt.date
    df["IsHoliday"] = dates_as_date.apply(lambda d: d in us_holidays)

    # ── Location description grouping ─────────────────────────────────────────
    if "Location Description" in df.columns:
        df["LocationGrouped"] = (
            df["Location Description"]
            .str.upper()
            .str.strip()
            .map(LOCATION_GROUP_MAP)
            .fillna("Other")
        )

    log.info("Feature engineering complete. Shape: %d × %d.", *df.shape)
    return df


def train_test_split_temporal(
    df: pd.DataFrame,
    train_frac: float = 0.8,
    date_col: str = "Date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a time-ordered DataFrame into train and test sets chronologically.

    Chronological splitting is critical for time-series data to prevent
    data leakage — i.e. the model must never see future crime records during
    training.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned and feature-engineered DataFrame, containing *date_col*.
    train_frac : float
        Fraction of data (by time) to include in the training set.
        Default 0.8 (80 % train, 20 % test).
    date_col : str
        Name of the datetime column to sort by. Default ``"Date"``.

    Returns
    -------
    (train, test) : tuple of pd.DataFrame
        Training and test DataFrames sorted by date.

    Raises
    ------
    ValueError
        If *train_frac* is not in (0, 1).
    """
    if not (0 < train_frac < 1):
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}.")

    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    cutoff_idx = int(len(df_sorted) * train_frac)

    train = df_sorted.iloc[:cutoff_idx].copy()
    test  = df_sorted.iloc[cutoff_idx:].copy()

    log.info(
        "Temporal split: %d train rows (up to %s) | %d test rows (from %s).",
        len(train), train[date_col].max().date(),
        len(test),  test[date_col].min().date(),
    )
    return train, test
