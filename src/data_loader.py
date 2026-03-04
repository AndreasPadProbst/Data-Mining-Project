"""
data_loader.py — Load and validate the Chicago Crimes dataset.

This module provides functions for safely loading the raw Crimes.csv file
(or its processed Parquet counterpart) with dtype enforcement, basic schema
validation, and optional memory-reduction tricks for working on machines
with limited RAM.

Functions
---------
load_raw_csv(path, nrows=None, low_memory=True)
    Load the raw CSV with enforced dtypes; return a DataFrame.

load_processed(path)
    Load a previously saved Parquet file; return a DataFrame.

validate_schema(df, required_columns)
    Assert that all required columns are present; raise ValueError if not.

summarise(df)
    Print a concise quality summary: shape, dtypes, null counts, memory.

reduce_memory(df)
    Downcast numeric columns to save RAM; return the modified DataFrame.
"""

import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ── Logger ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ── Expected schema for the raw Chicago Crimes CSV ────────────────────────────
RAW_DTYPE_MAP: dict = {
    "ID": "Int64",
    "Case Number": "object",
    "Date": "object",          # parsed explicitly later
    "Block": "object",
    "IUCR": "object",
    "Primary Type": "object",
    "Description": "object",
    "Location Description": "object",
    "Arrest": "boolean",
    "Domestic": "boolean",
    "Beat": "Int16",
    "District": "Int8",
    "Ward": "Int8",
    "Community Area": "Int8",
    "FBI Code": "object",
    "X Coordinate": "Int32",
    "Y Coordinate": "Int32",
    "Year": "Int16",
    "Updated On": "object",
    "Latitude": "float64",
    "Longitude": "float64",
    "Location": "object",
}

REQUIRED_COLUMNS: list[str] = [
    "ID", "Case Number", "Date", "Primary Type",
    "Arrest", "Domestic", "Beat", "District",
    "Latitude", "Longitude", "Year",
]


def load_raw_csv(
    path: str | Path,
    nrows: int | None = None,
    low_memory: bool = True,
) -> pd.DataFrame:
    """
    Load the raw Chicago Crimes CSV file with dtype enforcement.

    Parameters
    ----------
    path : str or Path
        Path to Crimes.csv (or any compatible CSV).
    nrows : int, optional
        If set, only the first *nrows* rows are loaded. Useful for quick
        development iterations on a large file (~1.7 GB).
    low_memory : bool, default True
        Pass ``low_memory=True`` to pandas to chunk the file internally,
        reducing peak RAM usage at the cost of slightly slower reads.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame with column dtypes cast as far as possible.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If required columns are absent from the CSV.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. "
            "Run `bash docs/download_data.sh` to fetch it."
        )

    logger.info("Loading raw CSV from %s (nrows=%s) …", path, nrows)

    df = pd.read_csv(
        path,
        dtype=str,           # read everything as str first — safest
        nrows=nrows,
        low_memory=low_memory,
    )

    logger.info("  Raw shape: %s rows × %s columns", *df.shape)

    # ── Cast columns according to RAW_DTYPE_MAP where possible ───────────────
    for col, dtype in RAW_DTYPE_MAP.items():
        if col not in df.columns:
            continue
        try:
            if dtype == "boolean":
                df[col] = df[col].map({"true": True, "false": False, "True": True, "False": False})
                df[col] = df[col].astype("boolean")
            elif dtype.startswith("Int"):
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)
            elif dtype == "float64":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            # object columns stay as-is
        except Exception as exc:
            logger.warning("  Could not cast column '%s' to %s: %s", col, dtype, exc)

    validate_schema(df, REQUIRED_COLUMNS)
    logger.info("  Schema validation passed.")
    return df


def load_processed(path: str | Path) -> pd.DataFrame:
    """
    Load a previously cleaned and saved Parquet file.

    Parameters
    ----------
    path : str or Path
        Path to the .parquet file produced by the cleaning pipeline.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Processed file not found at '{path}'. "
            "Run notebook 02_data_cleaning.ipynb first."
        )
    logger.info("Loading processed Parquet from %s …", path)
    df = pd.read_parquet(path)
    logger.info("  Loaded shape: %s rows × %s columns", *df.shape)
    return df


def validate_schema(df: pd.DataFrame, required_columns: list[str]) -> None:
    """
    Assert that all required columns are present in *df*.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    required_columns : list[str]
        Column names that must be present.

    Raises
    ------
    ValueError
        With a clear message listing any missing columns.
    """
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Schema validation failed. Missing columns: {missing}\n"
            f"Actual columns: {list(df.columns)}"
        )


def summarise(df: pd.DataFrame) -> None:
    """
    Print a concise quality summary of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Any DataFrame.
    """
    total = len(df)
    print(f"\n{'='*60}")
    print(f"  Shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print(f"\n  Column          | dtype      | nulls  | null%")
    print(f"  {'─'*55}")
    for col in df.columns:
        n_null = df[col].isna().sum()
        pct = 100 * n_null / total if total else 0
        print(f"  {col:<22} {str(df[col].dtype):<12} {n_null:<8,} {pct:.1f}%")
    print(f"{'='*60}\n")


def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcast numeric columns in-place to reduce RAM usage.

    Floats are cast to float32; integers are shrunk to the smallest signed
    integer type that can represent the range of values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to optimise.

    Returns
    -------
    pd.DataFrame
        The modified DataFrame (same object, returned for chaining).
    """
    before = df.memory_usage(deep=True).sum() / 1e6
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64", "int32"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    after = df.memory_usage(deep=True).sum() / 1e6
    logger.info("Memory reduced: %.1f MB → %.1f MB (%.0f%% saving)", before, after, 100 * (1 - after / before))
    return df
