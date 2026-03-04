"""
geo_utils.py
============
Geospatial helper functions for the Chicago crime analysis project.

Provides:
- Point-in-polygon imputation of missing Ward / Community Area / Beat values
  using official Chicago GeoJSON boundary files.
- Folium choropleth and heat-map builder functions.
- Beat-level crime count aggregation.

Typical usage
-------------
>>> from src.geo_utils import impute_spatial_columns, build_choropleth_map
>>> df = impute_spatial_columns(df, "boundaries/")
>>> m  = build_choropleth_map(df, "boundaries/Comm_Boundary.geojson")
>>> m.save("figures/choropleth.html")
"""

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import HeatMap
from shapely.geometry import Point

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# GeoJSON field names for each boundary type
GEOJSON_FIELD = {
    "ward":      "ward",
    "community": "area_numbe",
    "beat":      "beat_num",
    "district":  "dist_num",
}

# Chicago city centre — used as the default map centroid
CHICAGO_CENTRE = [41.8781, -87.6298]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_boundary(filepath: str) -> gpd.GeoDataFrame:
    """
    Load a GeoJSON boundary file into a GeoDataFrame.

    Parameters
    ----------
    filepath : str
        Path to the .geojson file.

    Returns
    -------
    gpd.GeoDataFrame
        Boundary GeoDataFrame in EPSG:4326 (WGS84).

    Raises
    ------
    FileNotFoundError
        If the file is absent.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Boundary file not found: '{filepath}'")
    gdf = gpd.read_file(filepath)
    return gdf.to_crs(epsg=4326)


def _assign_area(
    lon: float, lat: float, boundary_gdf: gpd.GeoDataFrame, field: str
) -> Optional[str]:
    """
    Determine which polygon in *boundary_gdf* contains the point (lon, lat).

    Parameters
    ----------
    lon : float
        Longitude coordinate.
    lat : float
        Latitude coordinate.
    boundary_gdf : gpd.GeoDataFrame
        GeoDataFrame with polygon geometries and the target *field*.
    field : str
        Column name in *boundary_gdf* that holds the area identifier.

    Returns
    -------
    str or None
        The area identifier value for the containing polygon, or None if the
        point does not fall within any polygon (e.g. lies just outside the
        city boundary).
    """
    point = Point(lon, lat)
    for _, row in boundary_gdf.iterrows():
        if row.geometry.contains(point):
            return row[field]
    return None


# ── Public functions ──────────────────────────────────────────────────────────

def impute_spatial_columns(
    df: pd.DataFrame,
    boundary_dir: str = "boundaries/",
    sample_limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Fill missing ``Ward``, ``Community_Area``, and ``Beat`` values using
    point-in-polygon lookups against official Chicago boundary files.

    Only rows that have valid ``Latitude`` / ``Longitude`` values AND at
    least one missing spatial column are processed — all other rows are
    returned unchanged.

    .. warning::
       For the full ~7M row dataset this operation is computationally
       expensive (O(rows × polygons)). Use *sample_limit* or run on a
       pre-filtered subset with missing values only.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned crime DataFrame containing ``Latitude``, ``Longitude``,
        and the spatial columns to be imputed.
    boundary_dir : str
        Directory containing the GeoJSON files:
        ``Beat_Boundary.geojson``, ``Comm_Boundary.geojson``,
        ``Ward_Boundary.geojson``.
    sample_limit : int, optional
        If set, only impute the first *sample_limit* rows that need
        imputation (useful for testing).

    Returns
    -------
    pd.DataFrame
        DataFrame with spatial columns filled where possible.
    """
    df = df.copy()

    # Load boundaries
    ward_gdf = _load_boundary(os.path.join(boundary_dir, "Ward_Boundary.geojson"))
    comm_gdf = _load_boundary(os.path.join(boundary_dir, "Comm_Boundary.geojson"))
    beat_gdf = _load_boundary(os.path.join(boundary_dir, "Beat_Boundary.geojson"))

    # Identify rows that need imputation
    needs_imputation = (
        df["Ward"].isna() | df["Community_Area"].isna() | df["Beat"].isna()
    ) & df["Latitude"].notna() & df["Longitude"].notna()

    idx_to_fix = df.index[needs_imputation].tolist()
    if sample_limit:
        idx_to_fix = idx_to_fix[:sample_limit]

    log.info("Imputing spatial columns for %d rows ...", len(idx_to_fix))

    for i, idx in enumerate(idx_to_fix):
        lon = df.at[idx, "Longitude"]
        lat = df.at[idx, "Latitude"]

        if pd.isna(df.at[idx, "Ward"]):
            df.at[idx, "Ward"] = _assign_area(lon, lat, ward_gdf, GEOJSON_FIELD["ward"])
        if pd.isna(df.at[idx, "Community_Area"]):
            df.at[idx, "Community_Area"] = _assign_area(
                lon, lat, comm_gdf, GEOJSON_FIELD["community"]
            )
        if pd.isna(df.at[idx, "Beat"]):
            df.at[idx, "Beat"] = _assign_area(lon, lat, beat_gdf, GEOJSON_FIELD["beat"])

        if (i + 1) % 500 == 0:
            log.info("  ... %d / %d rows processed.", i + 1, len(idx_to_fix))

    log.info("Spatial imputation complete.")
    return df


def build_choropleth_map(
    df: pd.DataFrame,
    geojson_path: str,
    geo_key: str = "area_numbe",
    df_key: str = "Community_Area",
    title: str = "Crime Count by Community Area",
    zoom_start: int = 11,
) -> folium.Map:
    """
    Build a Folium choropleth map showing crime density by geographic area.

    Parameters
    ----------
    df : pd.DataFrame
        Crime DataFrame containing the column specified by *df_key*.
    geojson_path : str
        Path to the GeoJSON boundary file.
    geo_key : str
        Feature property name in the GeoJSON used to join with *df_key*.
        Default ``"area_numbe"`` (Community Area boundaries).
    df_key : str
        Column in *df* that identifies the area. Default ``"Community_Area"``.
    title : str
        Map title displayed as an HTML legend header.
    zoom_start : int
        Initial Leaflet zoom level. Default 11 (city-wide view).

    Returns
    -------
    folium.Map
        Interactive choropleth map. Save with ``m.save("path/to/map.html")``.
    """
    # Aggregate crime counts by area
    crime_counts = (
        df.groupby(df_key).size().reset_index(name="CrimeCount")
    )
    crime_counts[df_key] = crime_counts[df_key].astype(str)

    # Build base map centred on Chicago
    m = folium.Map(location=CHICAGO_CENTRE, zoom_start=zoom_start, tiles="CartoDB positron")

    folium.Choropleth(
        geo_data=geojson_path,
        name="choropleth",
        data=crime_counts,
        columns=[df_key, "CrimeCount"],
        key_on=f"feature.properties.{geo_key}",
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.3,
        legend_name="Number of Crimes",
        nan_fill_color="lightgray",
    ).add_to(m)

    folium.LayerControl().add_to(m)
    return m


def build_heat_map(
    df: pd.DataFrame,
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
    max_points: int = 50_000,
    zoom_start: int = 11,
    radius: int = 8,
) -> folium.Map:
    """
    Build a Folium heat map from crime incident coordinates.

    .. note::
       For performance, only a random sample of up to *max_points* rows is
       plotted. With ~7M records, plotting all points would make the map
       unusable in a browser.

    Parameters
    ----------
    df : pd.DataFrame
        Crime DataFrame with latitude and longitude columns.
    lat_col : str
        Name of the latitude column. Default ``"Latitude"``.
    lon_col : str
        Name of the longitude column. Default ``"Longitude"``.
    max_points : int
        Maximum number of coordinate pairs to pass to the heat map.
        Default 50,000.
    zoom_start : int
        Initial Leaflet zoom level. Default 11.
    radius : int
        Pixel radius of each heat-map point. Default 8.

    Returns
    -------
    folium.Map
        Interactive heat map. Save with ``m.save("path/to/heatmap.html")``.
    """
    # Sample if needed
    sample_df = df.dropna(subset=[lat_col, lon_col])
    if len(sample_df) > max_points:
        sample_df = sample_df.sample(n=max_points, random_state=42)

    heat_data = list(zip(sample_df[lat_col], sample_df[lon_col]))

    m = folium.Map(location=CHICAGO_CENTRE, zoom_start=zoom_start, tiles="CartoDB dark_matter")
    HeatMap(heat_data, radius=radius, blur=10, max_zoom=13).add_to(m)
    return m
