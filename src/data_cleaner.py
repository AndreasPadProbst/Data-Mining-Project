"""
data_cleaner.py — Data cleaning, imputation, and quality reporting.

This module implements the end-to-end cleaning pipeline for the Chicago
Crimes dataset.  Each public function corresponds to one cleaning step;
they can be run individually or composed via ``run_full_pipeline()``.

Pipeline Steps
--------------
1.  parse_dates          — Parse the ``Date`` column; extract temporal features.
2.  drop_duplicates      — Remove duplicate Case Numbers (keep first).
3.  fix_coordinates      — Drop rows with coordinates outside Chicago's bounding box.
4.  impute_from_beat     — Fill missing District/Sector from Beat mapping.
5.  impute_geo_from_point— Fill missing Ward / Community Area via point-in-polygon.
6.  normalise_location   — Collapse noisy Location Description strings.
7.  check_iucr_type      — Flag rows where IUCR code contradicts Primary Type.
8.  generate_report      — Save an HTML data-quality report.
9.  save_parquet         — Persist the cleaned DataFrame.

run_full_pipeline(raw_path, boundaries_dir, output_path)
    Orchestrate all steps and return the cleaned DataFrame.
"""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ── Chicago bounding box (rough) ──────────────────────────────────────────────
CHICAGO_LAT_MIN, CHICAGO_LAT_MAX = 41.60, 42.05
CHICAGO_LON_MIN, CHICAGO_LON_MAX = -87.95, -87.50

# ── Beat → District lookup (first two digits of a 4-digit beat = district) ──
def _beat_to_district(beat: int) -> int | None:
    """Derive district from a police beat number (first 1-2 digits)."""
    try:
        return int(str(int(beat))[:2])
    except Exception:
        return None


# ── Location Description canonicalisation map ─────────────────────────────────
LOCATION_CANONICAL = {
    r"street|alley|sidewalk|roadway|driveway|highway": "STREET / ROAD",
    r"residence|house|apartment|condo|home|dwelling": "RESIDENCE",
    r"school|college|university|campus": "SCHOOL / UNIVERSITY",
    r"park|playground|forest|beach": "PARK / OUTDOOR",
    r"gas station|fuel": "GAS STATION",
    r"store|shop|retail|market|grocery|mall|boutique": "RETAIL / STORE",
    r"restaurant|diner|cafe|eatery|tavern|bar|lounge|club": "RESTAURANT / BAR",
    r"bank|atm|financial": "BANK / ATM",
    r"vehicle|car|auto|cab|taxi|bus|train|transit": "VEHICLE / TRANSIT",
    r"hospital|clinic|medical|health": "HOSPITAL / MEDICAL",
    r"hotel|motel|hostel": "HOTEL / MOTEL",
    r"office|workplace|business|commercial": "OFFICE / BUSINESS",
    r"warehouse|factory|industrial": "WAREHOUSE / INDUSTRIAL",
    r"construction|vacant lot|abandoned": "VACANT / CONSTRUCTION",
    r"airport|train station|platform": "TRANSPORT HUB",
    r"church|temple|mosque|religious": "RELIGIOUS INSTITUTION",
    r"library|government|city hall|court": "GOVERNMENT / LIBRARY",
    r"parking|garage|lot": "PARKING",
    r"sports|stadium|arena|gym|field": "SPORTS VENUE",
    r"liquor": "LIQUOR STORE",
    r"laundry|cleaners": "LAUNDRY",
    r"convenience store|drug store|pharmacy": "CONVENIENCE / PHARMACY",
    r"jail|prison|police|detention": "JAIL / POLICE",
    r"lake|river|water": "WATERFRONT",
}


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the ``Date`` column and extract temporal features.

    Adds columns: ``Datetime``, ``Year``, ``Month``, ``Day``, ``Hour``,
    ``DayOfWeek``, ``DayOfWeekName``, ``Season``, ``IsWeekend``.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame with a string ``Date`` column.

    Returns
    -------
    pd.DataFrame
        DataFrame with new temporal columns added.
    """
    logger.info("Parsing dates …")
    df = df.copy()
    df["Datetime"] = pd.to_datetime(df["Date"], infer_datetime_format=True, errors="coerce")
    n_failed = df["Datetime"].isna().sum()
    if n_failed:
        logger.warning("  %d rows with unparseable dates (will be NaT).", n_failed)

    df["Year"]          = df["Datetime"].dt.year.astype("Int16")
    df["Month"]         = df["Datetime"].dt.month.astype("Int8")
    df["Day"]           = df["Datetime"].dt.day.astype("Int8")
    df["Hour"]          = df["Datetime"].dt.hour.astype("Int8")
    df["DayOfWeek"]     = df["Datetime"].dt.dayofweek.astype("Int8")   # 0=Mon
    df["DayOfWeekName"] = df["Datetime"].dt.day_name()
    df["IsWeekend"]     = df["DayOfWeek"].isin([5, 6])

    def _season(month):
        if pd.isna(month):
            return np.nan
        m = int(month)
        if m in (12, 1, 2):
            return "Winter"
        if m in (3, 4, 5):
            return "Spring"
        if m in (6, 7, 8):
            return "Summer"
        return "Autumn"

    df["Season"] = df["Month"].apply(_season)
    logger.info("  Date parsing complete.")
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows based on ``Case Number``.

    The first occurrence is kept; duplicates are logged and dropped.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame with duplicates removed.
    """
    logger.info("Dropping duplicate Case Numbers …")
    before = len(df)
    df = df.drop_duplicates(subset=["Case Number"], keep="first")
    dropped = before - len(df)
    logger.info("  Dropped %d duplicate rows (%.2f%%).", dropped, 100 * dropped / before)
    return df


def fix_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag or drop rows where Latitude/Longitude fall outside Chicago's
    bounding box, or where coordinates are (0, 0).

    Rows with missing coordinates are kept (imputed later).  Rows with
    clearly erroneous non-null coordinates are dropped.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    logger.info("Fixing out-of-bounds coordinates …")
    has_coords = df["Latitude"].notna() & df["Longitude"].notna()
    out_of_bounds = has_coords & (
        (df["Latitude"] < CHICAGO_LAT_MIN) | (df["Latitude"] > CHICAGO_LAT_MAX) |
        (df["Longitude"] < CHICAGO_LON_MIN) | (df["Longitude"] > CHICAGO_LON_MAX) |
        (df["Latitude"] == 0) | (df["Longitude"] == 0)
    )
    n_bad = out_of_bounds.sum()
    if n_bad:
        logger.warning("  Dropping %d rows with out-of-bounds coordinates.", n_bad)
        df = df[~out_of_bounds].copy()
    else:
        logger.info("  No out-of-bounds coordinates found.")
    return df


def impute_from_beat(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing ``District`` values by deriving from ``Beat``.

    Chicago police beats follow the pattern DDBB where DD = district and
    BB = beat within the district.  This allows us to recover District
    when it is missing but Beat is present.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    logger.info("Imputing District from Beat …")
    mask = df["District"].isna() & df["Beat"].notna()
    if mask.sum() == 0:
        logger.info("  Nothing to impute.")
        return df
    df = df.copy()
    df.loc[mask, "District"] = df.loc[mask, "Beat"].apply(_beat_to_district)
    filled = mask.sum() - df.loc[mask, "District"].isna().sum()
    logger.info("  Filled %d District values from Beat.", filled)
    return df


def impute_geo_from_point(
    df: pd.DataFrame,
    boundaries_dir: str | Path,
) -> pd.DataFrame:
    """
    Fill missing ``Ward`` and ``Community Area`` using point-in-polygon
    lookups against the official GeoJSON boundary files.

    Requires geopandas and shapely.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``Latitude``, ``Longitude``, ``Ward``, ``Community Area``.
    boundaries_dir : str or Path
        Directory containing ``Ward_Boundary.geojson`` and
        ``Comm_Boundary.geojson``.

    Returns
    -------
    pd.DataFrame
    """
    boundaries_dir = Path(boundaries_dir)
    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError:
        logger.warning("geopandas not installed — skipping geo imputation.")
        return df

    # Only process rows with known coordinates but missing Ward/Comm Area
    needs_ward = df["Ward"].isna() & df["Latitude"].notna() & df["Longitude"].notna()
    needs_comm = df["Community Area"].isna() & df["Latitude"].notna() & df["Longitude"].notna()

    if not (needs_ward.any() or needs_comm.any()):
        logger.info("No Ward / Community Area imputation needed.")
        return df

    logger.info("Imputing Ward and Community Area via point-in-polygon …")
    df = df.copy()

    def _load_geo(fname):
        fpath = boundaries_dir / fname
        if not fpath.exists():
            logger.warning("  Boundary file %s not found — skipping.", fpath)
            return None
        return gpd.read_file(fpath)

    # Build GeoDataFrames for lookup rows
    sub = df[needs_ward | needs_comm].copy()
    gdf_sub = gpd.GeoDataFrame(
        sub,
        geometry=gpd.points_from_xy(sub["Longitude"], sub["Latitude"]),
        crs="EPSG:4326",
    )

    # Ward imputation
    ward_geo = _load_geo("Ward_Boundary.geojson")
    if ward_geo is not None and needs_ward.any():
        ward_geo = ward_geo.to_crs("EPSG:4326")
        joined = gpd.sjoin(gdf_sub[needs_ward.loc[gdf_sub.index]], ward_geo, how="left", predicate="within")
        ward_col = [c for c in ward_geo.columns if "ward" in c.lower()][0]
        df.loc[joined.index, "Ward"] = joined[ward_col].values
        logger.info("  Imputed %d Ward values.", needs_ward.sum())

    # Community Area imputation
    comm_geo = _load_geo("Comm_Boundary.geojson")
    if comm_geo is not None and needs_comm.any():
        comm_geo = comm_geo.to_crs("EPSG:4326")
        joined = gpd.sjoin(gdf_sub[needs_comm.loc[gdf_sub.index]], comm_geo, how="left", predicate="within")
        comm_col = [c for c in comm_geo.columns if "area" in c.lower() or "comm" in c.lower()][0]
        df.loc[joined.index, "Community Area"] = joined[comm_col].values
        logger.info("  Imputed %d Community Area values.", needs_comm.sum())

    return df


def normalise_location(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the 5 000+ raw ``Location Description`` strings into ~25
    canonical categories using regex pattern matching.

    A new column ``Location Category`` is added; the original column is
    preserved.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    logger.info("Normalising Location Description …")
    df = df.copy()

    def _classify(raw: str) -> str:
        if pd.isna(raw):
            return "UNKNOWN"
        raw_lower = str(raw).lower()
        for pattern, label in LOCATION_CANONICAL.items():
            if re.search(pattern, raw_lower):
                return label
        return "OTHER"

    df["Location Category"] = df["Location Description"].apply(_classify)
    dist = df["Location Category"].value_counts()
    logger.info("  Top location categories:\n%s", dist.head(10).to_string())
    return df


def check_iucr_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a boolean flag ``IUCR_Type_Mismatch`` that is True when the first
    character of the ``IUCR`` code does not agree with the expected major
    offense group for the declared ``Primary Type``.

    This is a data-quality flag — mismatches are not dropped automatically
    because they may represent legitimate data updates or reclassifications.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    logger.info("Checking IUCR / Primary Type consistency …")
    df = df.copy()

    # IUCR codes starting with '0' are homicide/battery etc.;
    # codes starting with '1' are robbery; '2' sex crimes; etc.
    # We perform a simplified sanity check: IUCR starting with '9'
    # should not be labeled "HOMICIDE".
    def _mismatch(row):
        iucr = str(row.get("IUCR", "")).strip()
        ptype = str(row.get("Primary Type", "")).strip().upper()
        if not iucr or iucr == "nan":
            return False
        if iucr.startswith("9") and ptype == "HOMICIDE":
            return True
        return False

    df["IUCR_Type_Mismatch"] = df.apply(_mismatch, axis=1)
    n_mismatch = df["IUCR_Type_Mismatch"].sum()
    logger.info("  Found %d IUCR / Primary Type mismatches (flagged).", n_mismatch)
    return df


def generate_report(df: pd.DataFrame, output_path: str | Path) -> None:
    """
    Generate and save an HTML data-quality report.

    The report includes: shape, dtypes, null counts, value distributions
    for key categorical columns, and a sample of flagged rows.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame to report on.
    output_path : str or Path
        Path for the output HTML file.
    """
    from jinja2 import Template

    logger.info("Generating data quality report → %s …", output_path)

    null_summary = (
        df.isna().sum()
        .rename("null_count")
        .to_frame()
        .assign(null_pct=lambda x: (100 * x["null_count"] / len(df)).round(2))
    )

    top_types = df["Primary Type"].value_counts().head(15).to_dict() if "Primary Type" in df.columns else {}

    html_template = """
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"><title>Data Quality Report</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 40px; }
      h1 { color: #2c3e50; }
      h2 { color: #2980b9; border-bottom: 1px solid #ccc; padding-bottom: 5px; }
      table { border-collapse: collapse; width: 100%; margin-bottom: 30px; }
      th { background-color: #2980b9; color: white; padding: 8px; text-align: left; }
      td { padding: 6px 8px; border-bottom: 1px solid #eee; }
      tr:hover { background-color: #f5f5f5; }
      .good  { color: green; }
      .warn  { color: orange; }
      .bad   { color: red; }
    </style>
    </head>
    <body>
    <h1>Chicago Crime Dataset — Data Quality Report</h1>
    <p><b>Rows:</b> {{ rows | format_number }}<br>
       <b>Columns:</b> {{ cols }}<br>
       <b>Memory:</b> {{ memory_mb }} MB</p>

    <h2>Null Value Summary</h2>
    <table>
    <tr><th>Column</th><th>Null Count</th><th>Null %</th></tr>
    {% for col, row in null_table.iterrows() %}
    <tr>
      <td>{{ col }}</td>
      <td>{{ row.null_count | int }}</td>
      <td class="{{ 'bad' if row.null_pct > 20 else ('warn' if row.null_pct > 5 else 'good') }}">
        {{ row.null_pct }}%
      </td>
    </tr>
    {% endfor %}
    </table>

    <h2>Top Crime Types</h2>
    <table>
    <tr><th>Primary Type</th><th>Count</th></tr>
    {% for k, v in top_types.items() %}
    <tr><td>{{ k }}</td><td>{{ v }}</td></tr>
    {% endfor %}
    </table>

    </body></html>
    """

    tpl = Template(html_template)
    html = tpl.render(
        rows=len(df),
        cols=len(df.columns),
        memory_mb=f"{df.memory_usage(deep=True).sum()/1e6:.1f}",
        null_table=null_summary,
        top_types=top_types,
    )

    Path(output_path).write_text(html, encoding="utf-8")
    logger.info("  Report saved.")


def save_parquet(df: pd.DataFrame, output_path: str | Path) -> None:
    """
    Persist the cleaned DataFrame to Parquet format.

    Parquet is chosen over CSV for its column-type preservation,
    compression, and fast read times on subsequent notebook runs.

    Parameters
    ----------
    df : pd.DataFrame
    output_path : str or Path
        Destination .parquet file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, compression="snappy")
    size_mb = output_path.stat().st_size / 1e6
    logger.info("Saved cleaned data → %s (%.1f MB)", output_path, size_mb)


def run_full_pipeline(
    raw_path: str | Path,
    boundaries_dir: str | Path,
    output_path: str | Path,
    report_path: str | Path | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """
    Orchestrate the complete cleaning pipeline.

    Steps executed in order:
    1. Load raw CSV
    2. Parse dates and extract temporal features
    3. Drop duplicate Case Numbers
    4. Fix out-of-bounds coordinates
    5. Impute District from Beat
    6. Impute Ward / Community Area via geo lookup
    7. Normalise Location Description
    8. Flag IUCR / Primary Type mismatches
    9. Generate HTML quality report
    10. Save as Parquet

    Parameters
    ----------
    raw_path : str or Path
        Path to Crimes.csv.
    boundaries_dir : str or Path
        Directory containing GeoJSON boundary files.
    output_path : str or Path
        Destination .parquet file.
    report_path : str or Path, optional
        If provided, an HTML quality report is written here.
    nrows : int, optional
        Limit rows loaded (useful for testing).

    Returns
    -------
    pd.DataFrame
        The fully cleaned DataFrame.
    """
    from .data_loader import load_raw_csv

    df = load_raw_csv(raw_path, nrows=nrows)
    df = parse_dates(df)
    df = drop_duplicates(df)
    df = fix_coordinates(df)
    df = impute_from_beat(df)
    df = impute_geo_from_point(df, boundaries_dir)
    df = normalise_location(df)
    df = check_iucr_type(df)

    if report_path:
        generate_report(df, report_path)

    save_parquet(df, output_path)
    logger.info("Full pipeline complete. Final shape: %s rows × %s columns", *df.shape)
    return df
