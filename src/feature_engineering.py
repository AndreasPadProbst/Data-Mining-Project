"""
feature_engineering.py — Derive analysis-ready features from cleaned data.

This module transforms the cleaned Chicago Crimes DataFrame into features
suitable for machine-learning models and statistical analyses.

Functions
---------
add_time_features(df)
    Cyclical encodings of Hour, DayOfWeek, Month.

add_lag_features(df, group_col, target_col, lags)
    Lag features for time-series-based models.

add_rolling_features(df, group_col, target_col, windows)
    Rolling mean/std features.

encode_categoricals(df, columns, method)
    One-hot or ordinal encoding of categorical columns.

build_ml_feature_matrix(df, target, feature_cols)
    Assemble a clean X, y pair for sklearn.

aggregate_monthly(df)
    Aggregate crime counts to monthly level for time-series analysis.

aggregate_by_district(df)
    Crime count per (Year, Month, District) for panel analysis.
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sine/cosine cyclical encodings for Hour, DayOfWeek, and Month.

    Linear encodings of cyclical features mislead models by treating
    hour 23 and hour 0 as far apart.  Sine/cosine projections preserve
    the circular distance.

    New columns added:
    - ``Hour_sin``, ``Hour_cos``
    - ``DayOfWeek_sin``, ``DayOfWeek_cos``
    - ``Month_sin``, ``Month_cos``

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``Hour``, ``DayOfWeek``, ``Month`` columns.

    Returns
    -------
    pd.DataFrame
    """
    logger.info("Adding cyclical time features …")
    df = df.copy()

    def _cyclic(series, period):
        vals = pd.to_numeric(series, errors="coerce").fillna(0)
        return (
            np.sin(2 * np.pi * vals / period),
            np.cos(2 * np.pi * vals / period),
        )

    if "Hour" in df.columns:
        df["Hour_sin"], df["Hour_cos"] = _cyclic(df["Hour"], 24)
    if "DayOfWeek" in df.columns:
        df["DayOfWeek_sin"], df["DayOfWeek_cos"] = _cyclic(df["DayOfWeek"], 7)
    if "Month" in df.columns:
        df["Month_sin"], df["Month_cos"] = _cyclic(df["Month"], 12)

    logger.info("  Cyclical features added.")
    return df


def add_lag_features(
    df: pd.DataFrame,
    group_col: str,
    target_col: str,
    lags: list[int],
) -> pd.DataFrame:
    """
    Add lag features for time-series regression.

    For each lag *k* in *lags*, a new column ``{target_col}_lag{k}`` is
    created containing the value of *target_col* from *k* rows earlier
    within each *group_col* group (e.g., each District).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame sorted by time within each group.
    group_col : str
        Column to group by (e.g., ``'District'``).
    target_col : str
        Column whose values will be lagged (e.g., ``'CrimeCount'``).
    lags : list[int]
        List of lag offsets (e.g., ``[1, 2, 3, 6, 12]``).

    Returns
    -------
    pd.DataFrame
    """
    logger.info("Adding lag features for '%s' grouped by '%s' …", target_col, group_col)
    df = df.copy()
    grouped = df.groupby(group_col)[target_col]
    for k in lags:
        col_name = f"{target_col}_lag{k}"
        df[col_name] = grouped.shift(k)
    logger.info("  Added %d lag columns.", len(lags))
    return df


def add_rolling_features(
    df: pd.DataFrame,
    group_col: str,
    target_col: str,
    windows: list[int],
) -> pd.DataFrame:
    """
    Add rolling-window mean and standard deviation features.

    For each window size *w*, two new columns are added:
    ``{target_col}_rollmean{w}`` and ``{target_col}_rollstd{w}``.

    Parameters
    ----------
    df : pd.DataFrame
    group_col : str
    target_col : str
    windows : list[int]
        E.g., ``[3, 6, 12]`` for 3-, 6-, and 12-period rolling stats.

    Returns
    -------
    pd.DataFrame
    """
    logger.info("Adding rolling features for '%s' …", target_col)
    df = df.copy()
    grouped = df.groupby(group_col)[target_col]
    for w in windows:
        df[f"{target_col}_rollmean{w}"] = grouped.transform(lambda x: x.rolling(w, min_periods=1).mean())
        df[f"{target_col}_rollstd{w}"]  = grouped.transform(lambda x: x.rolling(w, min_periods=1).std())
    logger.info("  Added %d rolling feature pairs.", len(windows))
    return df


def encode_categoricals(
    df: pd.DataFrame,
    columns: list[str],
    method: Literal["onehot", "ordinal"] = "onehot",
) -> pd.DataFrame:
    """
    Encode categorical columns for machine-learning models.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list[str]
        Column names to encode.
    method : {'onehot', 'ordinal'}
        - ``'onehot'``: Pandas ``get_dummies``; adds binary indicator columns.
        - ``'ordinal'``: sklearn ``OrdinalEncoder``; replaces column with int.

    Returns
    -------
    pd.DataFrame
    """
    logger.info("Encoding columns %s with method='%s' …", columns, method)
    df = df.copy()

    if method == "onehot":
        df = pd.get_dummies(df, columns=[c for c in columns if c in df.columns], drop_first=True)
    elif method == "ordinal":
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        valid = [c for c in columns if c in df.columns]
        df[valid] = enc.fit_transform(df[valid].astype(str))
    else:
        raise ValueError(f"method must be 'onehot' or 'ordinal', got '{method}'.")

    logger.info("  Encoding complete.")
    return df


def build_ml_feature_matrix(
    df: pd.DataFrame,
    target: str,
    feature_cols: list[str],
    dropna: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Assemble a (X, y) pair for sklearn.

    Parameters
    ----------
    df : pd.DataFrame
    target : str
        Name of the label column.
    feature_cols : list[str]
        Feature column names.
    dropna : bool, default True
        Drop rows where any feature or target is NaN before returning.

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    """
    available_features = [c for c in feature_cols if c in df.columns]
    missing = set(feature_cols) - set(available_features)
    if missing:
        logger.warning("Features not found in DataFrame: %s", missing)

    sub = df[available_features + [target]].copy()
    if dropna:
        before = len(sub)
        sub = sub.dropna()
        if len(sub) < before:
            logger.info("  Dropped %d rows with NaN in features/target.", before - len(sub))

    X = sub[available_features]
    y = sub[target]
    logger.info("  Feature matrix: %d rows × %d columns, target='%s'.", len(X), len(X.columns), target)
    return X, y


def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate crime counts to monthly level.

    Returns a DataFrame indexed by a monthly ``Timestamp`` with a single
    column ``Crime_Number``, suitable for time-series analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned crimes DataFrame with a ``Datetime`` column.

    Returns
    -------
    pd.DataFrame
        Columns: [``Timestamp`` (index), ``Crime_Number``]
    """
    logger.info("Aggregating to monthly crime counts …")
    df = df.copy()
    df["YearMonth"] = df["Datetime"].dt.to_period("M")
    monthly = (
        df.groupby("YearMonth")
        .size()
        .reset_index(name="Crime_Number")
    )
    monthly["Timestamp"] = monthly["YearMonth"].dt.to_timestamp()
    monthly = monthly.set_index("Timestamp")[["Crime_Number"]].sort_index()
    logger.info("  Monthly series: %d periods.", len(monthly))
    return monthly


def aggregate_by_district(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate crime counts per (Year, Month, District).

    Useful for panel data models and district-level forecasting.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Columns: ``Year``, ``Month``, ``District``, ``CrimeCount``.
    """
    logger.info("Aggregating by (Year, Month, District) …")
    panel = (
        df.groupby(["Year", "Month", "District"])
        .size()
        .reset_index(name="CrimeCount")
        .sort_values(["District", "Year", "Month"])
    )
    logger.info("  Panel shape: %d rows.", len(panel))
    return panel
