#!/usr/bin/env python3
"""
script_01_cleaning.py
=====================
Part 1: Data Cleaning & Feature Engineering
Run: python script_01_cleaning.py
Output: data/Crimes_Cleaned.csv, data/train.csv, data/test.csv
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
os.makedirs("data",    exist_ok=True)

print("=" * 60)
print("  PART 1 — Data Cleaning & Feature Engineering")
print("=" * 60)

# ── 1. Load raw data ──────────────────────────────────────────────────────────
print("\n[1/7] Loading raw dataset ...")
df = pd.read_csv("data/Crimes.csv", low_memory=False)
print(f"      Raw shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"      Raw columns: {list(df.columns)}")

# ── 2. Normalise column names (spaces → underscores) ─────────────────────────
#       The real Chicago dataset uses 'Primary Type', 'Community Area', etc.
#       We rename everything so the rest of the code can use simple names.
print("\n[2/7] Normalising column names ...")
df.columns = (
    df.columns
      .str.strip()
      .str.replace(" ", "_", regex=False)
      .str.replace(r"[^\w]", "_", regex=True)
)
print(f"      Renamed columns: {list(df.columns)}")

# ── 3. Drop rows with no location at all ────────────────────────────────────
print("\n[3/7] Dropping rows with missing Location ...")
before = len(df)
df = df[df["Location"].notna()].copy()
print(f"      Dropped {before - len(df):,} rows with NaN Location. Remaining: {len(df):,}")

# ── 4. Drop rows with missing coordinates ────────────────────────────────────
print("\n[4/7] Dropping rows with missing Latitude/Longitude ...")
before = len(df)
df = df[df["Latitude"].notna() & df["Longitude"].notna()].copy()
print(f"      Dropped {before - len(df):,} rows without coordinates. Remaining: {len(df):,}")

# ── 5. Deduplicate on Case Number ─────────────────────────────────────────────
print("\n[5/7] Deduplicating on Case_Number ...")
before = len(df)
df = df.drop_duplicates(subset=["Case_Number"]).copy()
print(f"      Dropped {before - len(df):,} duplicate rows. Remaining: {len(df):,}")

# ── 6. Parse dates & cast types ──────────────────────────────────────────────
print("\n[6/7] Parsing dates and casting types ...")
df["Date"]     = pd.to_datetime(df["Date"], infer_datetime_format=True)
df["Arrest"]   = df["Arrest"].astype(bool)
df["Domestic"] = df["Domestic"].astype(bool)

for col in ["Beat","District","Ward","Community_Area"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

for col in df.select_dtypes("object").columns:
    df[col] = df[col].str.strip()

# ── 7. Feature engineering ────────────────────────────────────────────────────
print("\n[7/7] Engineering features ...")
df["Hour"]        = df["Date"].dt.hour
df["DayOfWeek"]   = df["Date"].dt.dayofweek          # 0=Mon, 6=Sun
df["DayOfWeekName"] = df["Date"].dt.day_name()
df["Month"]       = df["Date"].dt.month
df["YearActual"]  = df["Date"].dt.year
df["IsWeekend"]   = df["DayOfWeek"].isin([5, 6]).astype(int)

df["Season"] = df["Month"].map({
    12:"Winter", 1:"Winter",  2:"Winter",
     3:"Spring", 4:"Spring",  5:"Spring",
     6:"Summer", 7:"Summer",  8:"Summer",
     9:"Fall",  10:"Fall",   11:"Fall"
})

# Condensed location groups (300+ raw values → 15 buckets)
LOCATION_MAP = {
    "STREET":              "Street/Alley",
    "SIDEWALK":            "Street/Alley",
    "ALLEY":               "Street/Alley",
    "RESIDENCE":           "Residential",
    "APARTMENT":           "Residential",
    "HOUSE":               "Residential",
    "RESIDENCE-GARAGE":    "Residential",
    "CHA APARTMENT":       "Residential",
    "VEHICLE NON-COMMERCIAL": "Vehicle",
    "AUTO":                "Vehicle",
    "VEHICLE-COMMERCIAL":  "Vehicle",
    "TAXICAB":             "Vehicle",
    "RETAIL STORE":        "Commercial",
    "DEPARTMENT STORE":    "Commercial",
    "GROCERY FOOD STORE":  "Commercial",
    "CONVENIENCE STORE":   "Commercial",
    "RESTAURANT":          "Commercial",
    "BAR OR TAVERN":       "Entertainment",
    "HOTEL/MOTEL":         "Entertainment",
    "PARK PROPERTY":       "Outdoor/Park",
    "FOREST PRESERVE":     "Outdoor/Park",
    "SCHOOL, PUBLIC, BUILDING": "School",
    "SCHOOL, PUBLIC, GROUNDS":  "School",
    "SCHOOL, PRIVATE, BUILDING":"School",
    "CTA TRAIN":           "Transit",
    "CTA BUS":             "Transit",
    "CTA PLATFORM":        "Transit",
    "OTHER":               "Other",
}
df["LocationGrouped"] = df["Location_Description"].map(LOCATION_MAP).fillna("Other")

# ── Missing value bar chart ───────────────────────────────────────────────────
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing) > 0:
    fig, ax = plt.subplots(figsize=(10, 4))
    missing.plot.bar(ax=ax, color="#2C7BB6", edgecolor="white")
    ax.set_title("Missing Values per Column (after cleaning)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Missing Values")
    ax.set_xlabel("")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig("figures/01_missing_values.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("      Saved: figures/01_missing_values.png")

# Location group bar chart
top_locs = df["LocationGrouped"].value_counts()
fig, ax = plt.subplots(figsize=(10, 4))
top_locs.sort_values().plot.barh(ax=ax, color="#2C7BB6", edgecolor="white")
ax.set_title("Crime Count by Location Group", fontsize=12, fontweight="bold")
ax.set_xlabel("Number of Incidents")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
plt.savefig("figures/01_location_groups.png", dpi=150, bbox_inches="tight")
plt.close()
print("      Saved: figures/01_location_groups.png")

# ── Train / test split (chronological 80/20) ─────────────────────────────────
df_sorted = df.sort_values("Date").reset_index(drop=True)
split_idx = int(len(df_sorted) * 0.8)
train = df_sorted.iloc[:split_idx]
test  = df_sorted.iloc[split_idx:]
print(f"\n      Train: {len(train):,} rows  |  Test: {len(test):,} rows")

# ── Save outputs ──────────────────────────────────────────────────────────────
df.to_csv("data/Crimes_Cleaned.csv", index=False)
train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv",  index=False)

print(f"\n      Saved: data/Crimes_Cleaned.csv  ({os.path.getsize('data/Crimes_Cleaned.csv')/1e6:.0f} MB)")
print(       "      Saved: data/train.csv")
print(       "      Saved: data/test.csv")
print(f"\n      Final columns: {list(df.columns)}")

print("\n" + "=" * 60)
print("  PART 1 COMPLETE")
print("=" * 60)
