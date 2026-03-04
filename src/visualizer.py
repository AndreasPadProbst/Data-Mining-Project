"""
visualizer.py — Plotting utilities for the Chicago Crime project.

All public functions save their figure to ``output_dir`` and return the
matplotlib Figure (or a Folium Map object for interactive maps).  This
allows the same functions to be used both inside Jupyter notebooks
(for inline display) and in batch scripts (saving files only).

Functions
---------
Static (Matplotlib / Seaborn)
    plot_crimes_per_year(df, output_dir)
    plot_crime_by_type(df, output_dir, top_n)
    plot_arrest_rate_by_type(df, output_dir)
    plot_hour_day_heatmap(df, output_dir)
    plot_year_month_heatmap(df, output_dir)
    plot_season_violin(df, output_dir)
    plot_missing_values(df, output_dir)
    plot_correlation_matrix(df, output_dir)
    plot_domestic_trend(df, output_dir)
    plot_crime_type_change(df, output_dir)

Geospatial (Folium)
    make_choropleth(df, geojson_path, key_col, value_col, output_path)
    make_heatmap(df, output_path, sample_n)
    make_cluster_map(df, output_path, sample_n)
"""

import logging
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ── Global style ─────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "figure.dpi": 120,
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})
PALETTE = "Blues_r"
ACCENT  = "#1a6ea8"


def _save(fig: plt.Figure, output_dir: Path, filename: str) -> Path:
    """Save a matplotlib figure and close it."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fpath = output_dir / filename
    fig.savefig(fpath, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s", fpath)
    return fpath


# ── Static plots ─────────────────────────────────────────────────────────────

def plot_crimes_per_year(df: pd.DataFrame, output_dir: str | Path) -> plt.Figure:
    """
    Line chart of total crime incidents per year.

    Parameters
    ----------
    df : pd.DataFrame  — cleaned crimes DataFrame with ``Year`` column.
    output_dir : str or Path

    Returns
    -------
    plt.Figure
    """
    output_dir = Path(output_dir)
    yearly = df["Year"].dropna().astype(int).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(yearly.index, yearly.values, marker="o", linewidth=2, color=ACCENT)
    ax.fill_between(yearly.index, yearly.values, alpha=0.15, color=ACCENT)
    ax.set_title("Total Crime Incidents per Year (2001 – Present)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Incidents")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    _save(fig, output_dir, "01_crimes_per_year.png")
    return fig


def plot_crime_by_type(
    df: pd.DataFrame,
    output_dir: str | Path,
    top_n: int = 15,
) -> plt.Figure:
    """
    Horizontal bar chart of the top-N crime Primary Types by count.

    Parameters
    ----------
    df : pd.DataFrame
    output_dir : str or Path
    top_n : int, default 15

    Returns
    -------
    plt.Figure
    """
    output_dir = Path(output_dir)
    counts = df["Primary Type"].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(counts.index[::-1], counts.values[::-1], color=ACCENT, edgecolor="white")
    for bar, val in zip(bars, counts.values[::-1]):
        ax.text(bar.get_width() * 1.005, bar.get_y() + bar.get_height() / 2,
                f"{val:,.0f}", va="center", fontsize=9)
    ax.set_title(f"Top {top_n} Crime Types", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Incidents")
    ax.set_xlim(0, counts.max() * 1.12)
    _save(fig, output_dir, "02_crime_by_type.png")
    return fig


def plot_arrest_rate_by_type(df: pd.DataFrame, output_dir: str | Path) -> plt.Figure:
    """
    Stacked bar chart showing arrest vs. no-arrest counts per crime type.

    Parameters
    ----------
    df : pd.DataFrame  — must contain ``Primary Type`` and ``Arrest`` columns.
    output_dir : str or Path

    Returns
    -------
    plt.Figure
    """
    output_dir = Path(output_dir)
    top_types = df["Primary Type"].value_counts().head(12).index
    sub = df[df["Primary Type"].isin(top_types)].copy()
    sub["Arrest"] = sub["Arrest"].astype(bool)
    pivot = sub.groupby(["Primary Type", "Arrest"]).size().unstack(fill_value=0)
    pivot = pivot.sort_values(True, ascending=False) if True in pivot.columns else pivot

    fig, ax = plt.subplots(figsize=(12, 7))
    pivot.plot(kind="barh", stacked=True, ax=ax,
               color=["#d62728", "#1a6ea8"], edgecolor="white")
    ax.set_title("Arrest vs. No-Arrest by Crime Type (Top 12)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Incidents")
    ax.legend(["No Arrest", "Arrest"], loc="lower right")
    _save(fig, output_dir, "03_arrest_rate_by_type.png")
    return fig


def plot_hour_day_heatmap(df: pd.DataFrame, output_dir: str | Path) -> plt.Figure:
    """
    Heatmap of crime counts by Hour (0–23) × Day of Week.

    Parameters
    ----------
    df : pd.DataFrame  — must contain ``Hour`` and ``DayOfWeek`` columns.
    output_dir : str or Path

    Returns
    -------
    plt.Figure
    """
    output_dir = Path(output_dir)
    pivot = df.groupby(["DayOfWeek", "Hour"]).size().unstack(fill_value=0)
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pivot.index = [day_labels[i] for i in pivot.index if i < 7]

    fig, ax = plt.subplots(figsize=(16, 5))
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax, linewidths=0.3, fmt=",d",
                annot=False, cbar_kws={"label": "Crime Count"})
    ax.set_title("Crime Incidents by Day of Week × Hour of Day", fontsize=14, fontweight="bold")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Day of Week")
    _save(fig, output_dir, "04_hour_day_heatmap.png")
    return fig


def plot_year_month_heatmap(df: pd.DataFrame, output_dir: str | Path) -> plt.Figure:
    """
    Heatmap of crime counts by Year × Month.

    Parameters
    ----------
    df : pd.DataFrame
    output_dir : str or Path

    Returns
    -------
    plt.Figure
    """
    output_dir = Path(output_dir)
    pivot = df.groupby(["Year", "Month"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(16, 10))
    sns.heatmap(pivot, cmap="Blues", ax=ax, linewidths=0.3,
                cbar_kws={"label": "Crime Count"})
    ax.set_title("Crime Incidents by Year × Month", fontsize=14, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    _save(fig, output_dir, "05_year_month_heatmap.png")
    return fig


def plot_season_violin(df: pd.DataFrame, output_dir: str | Path) -> plt.Figure:
    """
    Violin plot of daily crime counts per season.

    Parameters
    ----------
    df : pd.DataFrame  — must contain ``Datetime`` and ``Season`` columns.
    output_dir : str or Path

    Returns
    -------
    plt.Figure
    """
    output_dir = Path(output_dir)
    if "Season" not in df.columns or "Datetime" not in df.columns:
        logger.warning("Season or Datetime column missing — skipping violin plot.")
        return None

    daily = (
        df.assign(Date=df["Datetime"].dt.date)
        .groupby(["Date", "Season"])
        .size()
        .reset_index(name="DailyCount")
    )
    season_order = ["Spring", "Summer", "Autumn", "Winter"]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=daily, x="Season", y="DailyCount", order=season_order,
                   palette="Set2", ax=ax, inner="quartile", linewidth=1.2)
    ax.set_title("Daily Crime Count Distribution by Season", fontsize=14, fontweight="bold")
    ax.set_xlabel("Season")
    ax.set_ylabel("Daily Crime Count")
    _save(fig, output_dir, "06_season_violin.png")
    return fig


def plot_missing_values(df: pd.DataFrame, output_dir: str | Path) -> plt.Figure:
    """
    missingno-style heatmap of missing value patterns.

    Parameters
    ----------
    df : pd.DataFrame
    output_dir : str or Path

    Returns
    -------
    plt.Figure
    """
    output_dir = Path(output_dir)
    try:
        import missingno as msno
    except ImportError:
        logger.warning("missingno not installed — skipping missing value plot.")
        return None

    fig = msno.heatmap(df.sample(min(10_000, len(df)), random_state=42), figsize=(12, 8))
    fig = fig.get_figure()
    fig.suptitle("Missing Value Correlation Heatmap", fontsize=14, fontweight="bold", y=1.01)
    _save(fig, output_dir, "07_missing_values_heatmap.png")
    return fig


def plot_correlation_matrix(df: pd.DataFrame, output_dir: str | Path) -> plt.Figure:
    """
    Seaborn correlation heatmap for numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
    output_dir : str or Path

    Returns
    -------
    plt.Figure
    """
    output_dir = Path(output_dir)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Keep only columns with manageable cardinality
    useful = [c for c in numeric_cols
              if df[c].nunique() > 1 and c not in ["ID", "X Coordinate", "Y Coordinate"]]
    corr = df[useful].corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0, annot=True, fmt=".2f",
                ax=ax, linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Matrix (Numeric Features)", fontsize=14, fontweight="bold")
    _save(fig, output_dir, "08_correlation_matrix.png")
    return fig


def plot_domestic_trend(df: pd.DataFrame, output_dir: str | Path) -> plt.Figure:
    """
    Area chart of domestic vs. non-domestic crime proportion over years.

    Parameters
    ----------
    df : pd.DataFrame
    output_dir : str or Path

    Returns
    -------
    plt.Figure
    """
    output_dir = Path(output_dir)
    yearly = (
        df.groupby(["Year", "Domestic"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={True: "Domestic", False: "Non-Domestic"})
    )
    yearly_pct = yearly.div(yearly.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.stackplot(
        yearly_pct.index,
        yearly_pct.get("Domestic", pd.Series(dtype=float)),
        yearly_pct.get("Non-Domestic", pd.Series(dtype=float)),
        labels=["Domestic", "Non-Domestic"],
        colors=["#d62728", "#1a6ea8"],
        alpha=0.8,
    )
    ax.set_title("Domestic vs. Non-Domestic Crime Proportion Over Years", fontsize=14, fontweight="bold")
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("Year")
    ax.legend(loc="upper right")
    _save(fig, output_dir, "09_domestic_trend.png")
    return fig


def plot_crime_type_change(df: pd.DataFrame, output_dir: str | Path, top_n: int = 10) -> plt.Figure:
    """
    Grouped bar chart comparing crime type proportions between the first
    5 years (2001-2005) and the most recent 5 years of the dataset.

    Parameters
    ----------
    df : pd.DataFrame
    output_dir : str or Path
    top_n : int, default 10

    Returns
    -------
    plt.Figure
    """
    output_dir = Path(output_dir)
    years = sorted(df["Year"].dropna().astype(int).unique())
    early_years = years[:5]
    recent_years = years[-5:]

    early  = df[df["Year"].isin(early_years)]["Primary Type"].value_counts(normalize=True).head(top_n)
    recent = df[df["Year"].isin(recent_years)]["Primary Type"].value_counts(normalize=True)

    types = early.index
    early_vals  = early.values * 100
    recent_vals = recent.reindex(types).fillna(0).values * 100

    x = np.arange(len(types))
    width = 0.4

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width / 2, early_vals,  width, label=f"{early_years[0]}–{early_years[-1]}",  color="#1a6ea8", alpha=0.85)
    ax.bar(x + width / 2, recent_vals, width, label=f"{recent_years[0]}–{recent_years[-1]}", color="#d62728", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(types, rotation=30, ha="right")
    ax.set_ylabel("Share of All Crimes (%)")
    ax.set_title(f"Crime Type Proportion Change: Early Period vs. Recent Period (Top {top_n})",
                 fontsize=13, fontweight="bold")
    ax.legend()
    _save(fig, output_dir, "10_crime_type_change.png")
    return fig


# ── Geospatial (Folium) plots ─────────────────────────────────────────────────

def make_choropleth(
    df: pd.DataFrame,
    geojson_path: str | Path,
    key_col: str,
    value_col: str,
    output_path: str | Path,
    title: str = "Crime Density Choropleth",
    geojson_key: str | None = None,
) -> "folium.Map":
    """
    Generate a Folium choropleth map coloured by crime density.

    Parameters
    ----------
    df : pd.DataFrame
        Aggregated DataFrame with at least *key_col* and *value_col*.
    geojson_path : str or Path
        Path to the GeoJSON boundary file.
    key_col : str
        Column in *df* matching the GeoJSON feature property (e.g., ``'District'``).
    value_col : str
        Column with the values to colour by (e.g., ``'CrimeCount'``).
    output_path : str or Path
        Save the HTML interactive map here.
    title : str
    geojson_key : str, optional
        The GeoJSON property name matching *key_col*. Auto-detected if None.

    Returns
    -------
    folium.Map
    """
    try:
        import folium
        from folium.features import GeoJsonTooltip
    except ImportError:
        logger.warning("folium not installed — skipping choropleth.")
        return None

    import json

    geojson_path = Path(geojson_path)
    output_path  = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(geojson_path) as f:
        geo = json.load(f)

    # Auto-detect GeoJSON key property
    if geojson_key is None:
        props = geo["features"][0]["properties"]
        candidates = [k for k in props if any(x in k.lower() for x in [key_col.lower(), "id", "num", "ward", "dist", "area"])]
        geojson_key = candidates[0] if candidates else list(props.keys())[0]
        logger.info("  Auto-detected GeoJSON key: '%s'", geojson_key)

    m = folium.Map(location=[41.85, -87.65], zoom_start=10, tiles="CartoDB positron")

    folium.Choropleth(
        geo_data=geo,
        data=df,
        columns=[key_col, value_col],
        key_on=f"feature.properties.{geojson_key}",
        fill_color="YlOrRd",
        fill_opacity=0.75,
        line_opacity=0.4,
        legend_name=value_col,
        name=title,
    ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save(str(output_path))
    logger.info("  Choropleth saved → %s", output_path)
    return m


def make_heatmap(
    df: pd.DataFrame,
    output_path: str | Path,
    sample_n: int = 100_000,
) -> "folium.Map":
    """
    Generate a Folium kernel-density heatmap of crime locations.

    Parameters
    ----------
    df : pd.DataFrame  — must contain ``Latitude`` and ``Longitude``.
    output_path : str or Path
    sample_n : int, default 100_000
        Number of random rows to sample (for performance).

    Returns
    -------
    folium.Map
    """
    try:
        from folium.plugins import HeatMap
        import folium
    except ImportError:
        logger.warning("folium not installed — skipping heatmap.")
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    coords = df[["Latitude", "Longitude"]].dropna()
    if len(coords) > sample_n:
        coords = coords.sample(sample_n, random_state=42)

    m = folium.Map(location=[41.85, -87.65], zoom_start=10, tiles="CartoDB dark_matter")
    HeatMap(
        data=coords.values.tolist(),
        radius=12,
        blur=15,
        min_opacity=0.3,
        max_zoom=13,
    ).add_to(m)

    m.save(str(output_path))
    logger.info("  Heatmap saved → %s", output_path)
    return m


def make_cluster_map(
    df: pd.DataFrame,
    output_path: str | Path,
    sample_n: int = 30_000,
) -> "folium.Map":
    """
    Generate a Folium MarkerCluster map of crime locations.

    Parameters
    ----------
    df : pd.DataFrame
    output_path : str or Path
    sample_n : int, default 30_000

    Returns
    -------
    folium.Map
    """
    try:
        from folium.plugins import MarkerCluster
        import folium
    except ImportError:
        logger.warning("folium not installed — skipping cluster map.")
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sub = df[["Latitude", "Longitude", "Primary Type"]].dropna()
    if len(sub) > sample_n:
        sub = sub.sample(sample_n, random_state=42)

    m = folium.Map(location=[41.85, -87.65], zoom_start=11, tiles="CartoDB positron")
    cluster = MarkerCluster().add_to(m)

    for _, row in sub.iterrows():
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=4,
            color="#1a6ea8",
            fill=True,
            fill_opacity=0.6,
            popup=row["Primary Type"],
        ).add_to(cluster)

    m.save(str(output_path))
    logger.info("  Cluster map saved → %s", output_path)
    return m
