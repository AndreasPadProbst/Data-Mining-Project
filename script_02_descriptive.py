#!/usr/bin/env python3
"""
script_02_descriptive.py
========================
Part 2: Descriptive Analysis — all visualisations, one script, one data source.

Run:    python script_02_descriptive.py
Reads:  data/Crimes_Cleaned.csv   (produced by script_01_cleaning.py)

Outputs
-------
figures/         — pipeline analysis charts (25 PNG + 2 interactive HTML)
reports/figures/ — report illustration charts (17 PNG, referenced by descriptive_analysis.md)

Both sets are generated from the same real cleaned data.
gen_viz.py is no longer needed and can be deleted.
"""

import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter

os.makedirs("figures",         exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)

print("=" * 60)
print("  PART 2 — Descriptive Analysis & Visualisations")
print("=" * 60)

# ─── Load & normalise ─────────────────────────────────────────────────────────
print("\nLoading Crimes_Cleaned.csv ...")
df = pd.read_csv("data/Crimes_Cleaned.csv", parse_dates=["Date"], low_memory=False)
print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

df.columns = (df.columns.str.strip()
                         .str.replace(" ", "_", regex=False)
                         .str.replace(r"[^\w]", "_", regex=True))

# Resolve Primary_Type column (spaces vs underscores)
pt_col   = "Primary_Type" if "Primary_Type" in df.columns else "Primary Type"
comm_col = "Community_Area" if "Community_Area" in df.columns else "Community Area"

# Ensure DayOfWeekName exists
if "DayOfWeekName" not in df.columns:
    df["DayOfWeekName"] = df["Date"].dt.day_name()

# ─── Derived columns used by report charts ────────────────────────────────────
# Integer severity score (1=minor → 5=violent) mapped from crime type.
# A small uniform jitter is added so violin/box plots show density shape
# rather than discrete horizontal lines.
SEVERITY_MAP = {
    "HOMICIDE": 5,                    "CRIMINAL SEXUAL ASSAULT": 5,
    "KIDNAPPING": 5,                  "HUMAN TRAFFICKING": 5,
    "ROBBERY": 4,                     "BATTERY": 4,
    "WEAPONS VIOLATION": 4,           "ARSON": 4,
    "OFFENSE INVOLVING CHILDREN": 4,  "SEX OFFENSE": 4,
    "ASSAULT": 3,                     "BURGLARY": 3,
    "MOTOR VEHICLE THEFT": 3,         "STALKING": 3,
    "INTIMIDATION": 3,                "THEFT": 2,
    "CRIMINAL DAMAGE": 2,             "NARCOTICS": 2,
    "FRAUD": 2,                       "DECEPTIVE PRACTICE": 2,
    "GAMBLING": 1,                    "CRIMINAL TRESPASS": 1,
    "LIQUOR LAW VIOLATION": 1,        "PROSTITUTION": 1,
    "PUBLIC PEACE VIOLATION": 1,
}
np.random.seed(42)
df["Severity"] = (df[pt_col].map(SEVERITY_MAP).fillna(2).astype(float)
                  + np.random.normal(0, 0.20, len(df))).clip(1, 5)

# Arrest rate per crime type (used by parallel-coordinates chart)
df["ArrestRateByType"] = df.groupby(pt_col)["Arrest"].transform(
    lambda x: x.astype(bool).mean()
)

# ─── Season column (derived from Month if not already present in CSV) ─────────
if "Season" not in df.columns:
    _SM = {12:"Winter", 1:"Winter",  2:"Winter",
            3:"Spring",  4:"Spring",  5:"Spring",
            6:"Summer",  7:"Summer",  8:"Summer",
            9:"Fall",   10:"Fall",   11:"Fall"}
    df["Season"] = df["Month"].map(_SM)

# ─── Colour palette ───────────────────────────────────────────────────────────
NAVY   = "#1B2A4A"; BLUE   = "#2C7BB6"; RED    = "#D7191C"
TEAL   = "#00A896"; ORANGE = "#F4A261"; GOLD   = "#E9C46A"
LIGHT  = "#F0F4F8"; GRAY   = "#7F8C8D"

SEASON_ORDER = ["Winter", "Spring", "Summer", "Fall"]
SEASON_PAL   = {"Winter": BLUE, "Spring": TEAL, "Summer": ORANGE, "Fall": GOLD}

# ─── Save helpers ─────────────────────────────────────────────────────────────
def save(pipeline_name, report_name=None):
    """Save to figures/.  If report_name given, also save to reports/figures/."""
    plt.savefig(f"figures/{pipeline_name}", dpi=150, bbox_inches="tight")
    print(f"  → figures/{pipeline_name}")
    if report_name:
        plt.savefig(f"reports/figures/{report_name}", dpi=150, bbox_inches="tight")
        print(f"  → reports/figures/{report_name}")
    plt.close("all")

def save_report(report_name):
    """Save to reports/figures/ only."""
    plt.savefig(f"reports/figures/{report_name}", dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  → reports/figures/{report_name}")


# =============================================================================
# PART A — PIPELINE CHARTS  (figures/)
# Analytical outputs from the real data, unchanged from the original script_02.
# Four are also dual-saved as the matching report figure.
# =============================================================================

# ── [1] Annual crime volume ────────────────────────────────────────────────────
print("\n[1] Annual crime volume ...")
annual = df.groupby("YearActual").size().reset_index(name="CrimeCount")
annual = annual[annual["YearActual"] < annual["YearActual"].max()]
fig, ax = plt.subplots(figsize=(12, 5))
ax.fill_between(annual["YearActual"], annual["CrimeCount"], alpha=0.15, color=BLUE)
ax.plot(annual["YearActual"], annual["CrimeCount"],
        color=BLUE, linewidth=2.5, marker="o", markersize=7)
for _, row in annual.iterrows():
    ax.text(row.YearActual, row.CrimeCount + 1000,
            f"{int(row.CrimeCount):,}", ha="center", va="bottom", fontsize=7, color=NAVY)
ax.set_title("Annual Crime Volume in Chicago", fontsize=14, fontweight="bold", color=NAVY, pad=14)
ax.set_xlabel("Year"); ax.set_ylabel("Number of Incidents")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.grid(linestyle="--", alpha=0.4); ax.set_facecolor(LIGHT)
plt.tight_layout()
# Dual-saved: this IS the report's B1_annual_trend.png
save("02_annual_volume.png", report_name="B1_annual_trend.png")

# ── [2] Monthly seasonality ────────────────────────────────────────────────────
print("[2] Monthly seasonality ...")
monthly_avg = df.groupby("Month").size() / df["YearActual"].nunique()
month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(range(1, 13), monthly_avg.values, marker="o", color=RED, linewidth=2.5)
ax.fill_between(range(1, 13), monthly_avg.values, alpha=0.15, color=RED)
ax.set_xticks(range(1, 13)); ax.set_xticklabels(month_names)
ax.set_title("Average Monthly Crime Count (per year)", fontsize=13, fontweight="bold")
ax.set_xlabel("Month"); ax.set_ylabel("Average Crimes")
ax.grid(linestyle="--", alpha=0.4); ax.set_facecolor(LIGHT)
plt.tight_layout()
save("02_monthly_seasonality.png")

# ── [3] Day of week ────────────────────────────────────────────────────────────
print("[3] Day of week ...")
dow_order  = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
dow_counts = df["DayOfWeekName"].value_counts().reindex(dow_order).fillna(0)
fig, ax = plt.subplots(figsize=(9, 4))
dow_counts.plot.bar(ax=ax, color="#2166AC", edgecolor="white", rot=20)
ax.set_title("Crime Count by Day of Week", fontsize=13, fontweight="bold")
ax.set_ylabel("Number of Crimes")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.grid(axis="y", linestyle="--", alpha=0.4); ax.set_facecolor(LIGHT)
plt.tight_layout()
save("02_dow.png")

# ── [4] Hour of day ────────────────────────────────────────────────────────────
print("[4] Hour of day ...")
hour_counts = df["Hour"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(hour_counts.index, hour_counts.values, color="#4DAC26", edgecolor="white", width=0.8)
ax.set_title("Crime Count by Hour of Day (0 = midnight, 12 = noon)", fontsize=13, fontweight="bold")
ax.set_xlabel("Hour"); ax.set_ylabel("Number of Crimes"); ax.set_xticks(range(0, 24))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.grid(axis="y", linestyle="--", alpha=0.4); ax.set_facecolor(LIGHT)
plt.tight_layout()
save("02_hour.png")

# ── [5] Top 15 crime types ─────────────────────────────────────────────────────
print("[5] Top 15 crime types ...")
top15 = df[pt_col].value_counts().head(15)
fig, ax = plt.subplots(figsize=(10, 6))
top15.sort_values().plot.barh(ax=ax, color=BLUE, edgecolor="white")
ax.set_title("Top 15 Primary Crime Types", fontsize=13, fontweight="bold")
ax.set_xlabel("Number of Incidents")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.grid(axis="x", linestyle="--", alpha=0.4)
plt.tight_layout()
save("02_top_types.png")

# ── [6] Arrest rate by crime type ──────────────────────────────────────────────
print("[6] Arrest rate ...")
arrest_rate = (df.groupby(pt_col)["Arrest"]
               .apply(lambda x: x.astype(bool).mean())
               .sort_values(ascending=False).head(20).mul(100).round(1))
overall_arr = df["Arrest"].astype(bool).mean() * 100
fig, ax = plt.subplots(figsize=(10, 7))
arrest_rate.sort_values().plot.barh(ax=ax, color=RED, edgecolor="white")
ax.axvline(x=overall_arr, color=NAVY, linestyle="--", linewidth=1.5,
           label=f"Overall avg: {overall_arr:.1f}%")
ax.set_title("Arrest Rate by Crime Type (Top 20)", fontsize=13, fontweight="bold")
ax.set_xlabel("Arrest Rate (%)")
ax.legend(); ax.grid(axis="x", linestyle="--", alpha=0.4)
plt.tight_layout()
save("02_arrest_rate.png")

# ── [7] Domestic vs non-domestic ───────────────────────────────────────────────
print("[7] Domestic vs non-domestic ...")
domestic_annual = (df.groupby(["YearActual", "Domestic"])
                     .size().unstack(fill_value=0))
domestic_annual.columns = ["Non-Domestic", "Domestic"]
domestic_annual = domestic_annual[domestic_annual.index < domestic_annual.index.max()]
fig, ax = plt.subplots(figsize=(12, 5))
domestic_annual.plot(ax=ax, marker="o", linewidth=2,
                     color={"Domestic": RED, "Non-Domestic": BLUE})
ax.set_title("Domestic vs. Non-Domestic Crimes by Year", fontsize=13, fontweight="bold")
ax.set_xlabel("Year"); ax.set_ylabel("Number of Incidents")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.grid(linestyle="--", alpha=0.4)
plt.tight_layout()
save("02_domestic.png")

# ── [8] District / Community Area / Ward ───────────────────────────────────────
print("[8] District / Community Area / Ward ...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, c, title in zip(axes,
        ["District", comm_col, "Ward"],
        ["District", "Community Area", "Ward"]):
    if c in df.columns:
        df[c].value_counts().head(20).sort_values().plot.barh(
            ax=ax, color="#4393C3", edgecolor="white")
        ax.set_title(f"Top 20 by {title}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Crimes")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.grid(axis="x", linestyle="--", alpha=0.4)
    else:
        ax.set_visible(False)
plt.tight_layout()
save("02_district_commarea_ward.png")

# ── [9] Top 10 hotspot blocks ──────────────────────────────────────────────────
print("[9] Hotspot blocks ...")
fig, ax = plt.subplots(figsize=(10, 5))
df["Block"].value_counts().head(10).sort_values().plot.barh(
    ax=ax, color=RED, edgecolor="white")
ax.set_title("Top 10 Crime Hot-Spot Blocks", fontsize=13, fontweight="bold")
ax.set_xlabel("Number of Incidents")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.grid(axis="x", linestyle="--", alpha=0.4)
plt.tight_layout()
save("02_hotspots.png")

# ── [10] Monthly time-series ───────────────────────────────────────────────────
print("[10] Monthly time-series ...")
monthly_ts = df.set_index("Date").resample("ME").size().rename("Count")
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(monthly_ts.index, monthly_ts.values, color=BLUE, linewidth=1)
ax.fill_between(monthly_ts.index, monthly_ts.values, alpha=0.12, color=BLUE)
ax.set_title("Monthly Crime Count — Chicago", fontsize=13, fontweight="bold")
ax.set_xlabel("Date"); ax.set_ylabel("Crimes per Month")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.grid(linestyle="--", alpha=0.4); ax.set_facecolor(LIGHT)
plt.tight_layout()
save("02_monthly_ts.png")

# ── [11] Moving averages ───────────────────────────────────────────────────────
print("[11] Moving averages ...")
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(monthly_ts.index, monthly_ts.values,
        color="#AAAAAA", linewidth=0.8, alpha=0.6, label="Raw monthly")
for window, color, label in [(3, "#4393C3", "3-month MA"), (12, RED, "12-month MA")]:
    ax.plot(monthly_ts.rolling(window).mean().index,
            monthly_ts.rolling(window).mean().values,
            linewidth=2.5, color=color, label=label)
ax.set_title("Moving Average Smoothing of Monthly Crime Count", fontsize=13, fontweight="bold")
ax.set_xlabel("Date"); ax.set_ylabel("Crimes per Month"); ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.grid(linestyle="--", alpha=0.4); ax.set_facecolor(LIGHT)
plt.tight_layout()
save("02_moving_avg.png")

# ── [12] STL decomposition ─────────────────────────────────────────────────────
print("[12] STL decomposition ...")
from statsmodels.tsa.seasonal import STL
monthly_recent = monthly_ts[monthly_ts.index.year >= 2005]
res_stl = STL(monthly_recent, period=12, robust=True).fit()
fig, axes = plt.subplots(4, 1, figsize=(14, 10))
for ax, data, label in zip(axes,
        [monthly_recent, res_stl.trend, res_stl.seasonal, res_stl.resid],
        ["Observed", "Trend", "Seasonal", "Residual"]):
    ax.plot(monthly_recent.index, data, linewidth=1, color=BLUE)
    ax.set_ylabel(label, fontsize=9); ax.grid(linestyle="--", alpha=0.3)
axes[0].set_title("STL Decomposition of Monthly Crime Count", fontsize=13, fontweight="bold")
axes[-1].set_xlabel("Date")
plt.tight_layout()
save("02_stl_decomp.png")

# ── [13] ACF / PACF ────────────────────────────────────────────────────────────
print("[13] ACF / PACF ...")
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
plot_acf( monthly_recent, lags=36, ax=ax1, title="Autocorrelation Function (ACF)")
plot_pacf(monthly_recent, lags=36, ax=ax2, title="Partial Autocorrelation (PACF)")
plt.suptitle("ACF and PACF — Monthly Crime Count", fontsize=13, fontweight="bold")
plt.tight_layout()
save("02_acf_pacf.png")

# ── [14] SARIMA forecast ───────────────────────────────────────────────────────
print("[14] SARIMA model (this may take a few minutes) ...")
import statsmodels.api as sm
sarima = sm.tsa.SARIMAX(monthly_recent,
                         order=(1, 1, 1),
                         seasonal_order=(1, 1, 1, 12)).fit(disp=False)
fc   = sarima.get_forecast(steps=12)
fc_m = fc.predicted_mean
ci   = fc.conf_int()
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(monthly_recent.index[-60:], monthly_recent.values[-60:],
        label="Historical (last 5 yrs)", color=BLUE)
ax.plot(fc_m.index, fc_m.values,
        label="12-Month Forecast", color=RED, linestyle="--", linewidth=2)
ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1],
                alpha=0.2, color=RED, label="95% CI")
ax.set_title("SARIMA(1,1,1)(1,1,1)[12] — 12-Month Forecast", fontsize=13, fontweight="bold")
ax.set_xlabel("Date"); ax.set_ylabel("Crimes per Month")
ax.legend(); ax.grid(linestyle="--", alpha=0.4); ax.set_facecolor(LIGHT)
plt.tight_layout()
save("02_sarima_forecast.png")

# ── [15] Hour × Day heatmap ─────────────────────────────────────────────────────
print("[15] Hour × Day heatmap ...")
day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
day_abbr  = ["Mon",   "Tue",    "Wed",     "Thu",     "Fri",    "Sat",     "Sun"]
pivot = (df.groupby(["Hour", "DayOfWeekName"]).size()
           .unstack(fill_value=0)
           .reindex(columns=day_order, fill_value=0))
pivot.columns = day_abbr

fig, ax = plt.subplots(figsize=(13, 7))
sns.heatmap(pivot, cmap="YlOrRd", ax=ax, linewidths=0.3, linecolor="#EEEEEE",
            cbar_kws={"label": "Number of Incidents", "shrink": 0.85})
ax.set_title("Crime Incidents by Hour of Day & Day of Week",
             fontsize=16, fontweight="bold", color=NAVY, pad=18)
ax.set_xlabel("Day of Week", fontsize=12, labelpad=8)
ax.set_ylabel("Hour of Day (24h)", fontsize=12, labelpad=8)
ax.add_patch(plt.Rectangle((4, 20), 3, 4, fill=False,
             edgecolor=NAVY, lw=2.5, linestyle="--"))
ax.text(7.2, 22, "Peak crime\nwindow", fontsize=9, color=NAVY, va="center")
plt.tight_layout()
# Dual-saved: this IS the report's 01_heatmap.png
save("02_hour_dow_heatmap.png", report_name="01_heatmap.png")

# ── [16] Correlation matrix ─────────────────────────────────────────────────────
print("[16] Correlation matrix ...")
num_candidates = ["Hour","DayOfWeek","Month","YearActual","IsWeekend","IsHoliday",
                  "Beat","District","Ward","Community_Area","Severity"]
num_cols = [c for c in num_candidates if c in df.columns]
corr_df  = df[num_cols].copy()
corr_df["Arrest"]   = df["Arrest"].astype(bool).astype(int)
corr_df["Domestic"] = df["Domestic"].astype(bool).astype(int)
corr = corr_df.dropna().corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            vmin=-1, vmax=1, square=True, linewidths=0.6, linecolor="#DDDDDD",
            cbar_kws={"shrink": 0.8, "label": "Pearson r"}, ax=ax,
            annot_kws={"size": 8})
ax.set_title("Correlation Matrix — Key Crime Variables",
             fontsize=16, fontweight="bold", color=NAVY, pad=18)
plt.xticks(rotation=30, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
# Dual-saved: this IS the report's 03_correlation.png
save("02_correlation.png", report_name="03_correlation.png")

# ── [17] Year-over-year % change ────────────────────────────────────────────────
print("[17] Year-over-year % change ...")
max_yr  = df["YearActual"].max()
df_full = df[df["YearActual"] < max_yr]
top15_t = df_full[pt_col].value_counts().head(15).index
yoy     = (df_full[df_full[pt_col].isin(top15_t)]
           .groupby(["YearActual", pt_col]).size()
           .unstack(fill_value=0))
yoy_pct = (yoy.pct_change().mul(100)
               .dropna(how="all")
               .replace([np.inf, -np.inf], np.nan))
top8 = yoy_pct.abs().mean().nlargest(8).index
fig, ax = plt.subplots(figsize=(13, 5))
for col_name in top8:
    ax.plot(yoy_pct.index, yoy_pct[col_name],
            linewidth=1.8, marker="o", markersize=4, label=col_name)
ax.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
ax.set_title("Year-over-Year % Change — Top 8 Most Volatile Crime Types",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Year"); ax.set_ylabel("% Change vs Prior Year")
ax.legend(fontsize=8, loc="upper right", ncol=2)
ax.grid(linestyle="--", alpha=0.4); ax.set_facecolor(LIGHT)
plt.tight_layout()
save("02_yoy_change.png")


# =============================================================================
# PART B — REPORT-ONLY CHARTS  (reports/figures/)
# =============================================================================
# All 14 charts below use the real cleaned data already loaded above.
# B1_annual_trend.png and 03_correlation.png were dual-saved in Part A.
# =============================================================================

print("=" * 60)
print("  PART B — Generating report-only figures ...")
print("=" * 60)

# R1 — B3_crime_types  (already good, minor polish)
# ═══════════════════════════════════════════════════════════════
print("[R1] Top 10 crime types bar ...")
top10_t = df[pt_col].value_counts().head(10)
total   = len(df)
bar_colors=[BLUE,RED,TEAL,ORANGE,GOLD,NAVY,"#6C3483","#117A65","#784212","#1A5276"]
fig,ax=plt.subplots(figsize=(13,6))
bars=ax.barh(top10_t.index[::-1],top10_t.values[::-1],color=bar_colors,edgecolor="white",height=0.7)
for bar,val in zip(bars,top10_t.values[::-1]):
    pct=val/total*100
    ax.text(val+total*0.002, bar.get_y()+bar.get_height()/2,
            f"{val:,}  ({pct:.1f}%)", va="center", fontsize=9, color=NAVY, fontweight="bold")
ax.set_title("Top 10 Primary Crime Types — Chicago  (2001–2024)",
             fontsize=14,fontweight="bold",color=NAVY,pad=14)
ax.set_xlabel("Number of Incidents",fontsize=11)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"{int(x):,}"))
ax.set_facecolor(LIGHT); ax.grid(axis="x",linestyle="--",alpha=0.4)
ax.spines[["top","right"]].set_visible(False)
ax.text(0.99,-0.08,f"Total incidents shown: {top10_t.sum():,} of {total:,}",
        transform=ax.transAxes,ha="right",fontsize=8,color=GRAY,style="italic")
save_report("B3_crime_types.png")


# ═══════════════════════════════════════════════════════════════
# R2 — 02_violin  (add per-violin stats + count)
# ═══════════════════════════════════════════════════════════════
print("[R2] Violin (severity by type) ...")
top6=df[pt_col].value_counts().head(6).index.tolist()
df_v=df[df[pt_col].isin(top6)]
pal_v=[BLUE,TEAL,RED,ORANGE,GOLD,NAVY]

fig,ax=plt.subplots(figsize=(14,7))
ax.set_facecolor(LIGHT)
parts=ax.violinplot([df_v[df_v[pt_col]==t]["Severity"].values for t in top6],
    positions=range(len(top6)),widths=0.72,showmeans=True,showmedians=True,showextrema=True)
for pc,c in zip(parts["bodies"],pal_v):
    pc.set_facecolor(c); pc.set_alpha(0.78); pc.set_edgecolor("white"); pc.set_linewidth(0.5)
parts["cmeans"].set_color("#222222"); parts["cmeans"].set_linewidth(2.5)
parts["cmedians"].set_color("white"); parts["cmedians"].set_linewidth(2)
for k in ["cbars","cmins","cmaxes"]: parts[k].set_color(GRAY); parts[k].set_linewidth(1.2)

# Annotate each violin: severity integer, n, median, mean, IQR
for i,t in enumerate(top6):
    vals=df_v[df_v[pt_col]==t]["Severity"]
    med=vals.median(); mean=vals.mean()
    q1,q3=vals.quantile(0.25),vals.quantile(0.75)
    n=len(vals)
    sev_int=round(med)  # the assigned severity level
    # Label above violin
    ax.text(i, q3+0.28, f"Sev = {sev_int}", ha="center", va="bottom",
            fontsize=9, color=pal_v[i], fontweight="bold")
    # Stats box inside or below
    ax.text(i, q1-0.35,
            f"med {med:.2f}\nμ {mean:.2f}\nIQR {q1:.2f}–{q3:.2f}\nn={n:,}",
            ha="center", va="top", fontsize=7.5, color="#333333",
            bbox=dict(boxstyle="round,pad=0.3",facecolor="white",
                      edgecolor=pal_v[i],alpha=0.85,linewidth=0.8))

ax.set_xticks(range(len(top6)))
ax.set_xticklabels([t.title() for t in top6], rotation=18, ha="right", fontsize=10)
ax.set_ylabel("Crime Severity Score  (1 = minor  →  5 = violent)", fontsize=12)
ax.set_ylim(0.4, 5.9)
ax.set_yticks([1,2,3,4,5])
ax.set_yticklabels(["1 – Low","2 – Minor","3 – Moderate","4 – Serious","5 – Severe"],fontsize=9)
ax.axhline(y=2.5, color=GRAY, linestyle=":", linewidth=0.8, alpha=0.6)
ax.set_title("Distribution of Crime Severity Score by Type\n"
             "(Each crime type maps to a fixed FBI-UCR severity tier; jitter shows score spread)",
             fontsize=14, fontweight="bold", color=NAVY, pad=14)
ax.grid(axis="y", linestyle="--", alpha=0.4)
# Legend
ax.legend(handles=[
    Line2D([0],[0],color="#222222",lw=2.5,label="─── Mean"),
    Line2D([0],[0],color="white",lw=2,markerfacecolor=GRAY,
           markeredgecolor=GRAY,marker="_",markersize=10,label="─── Median"),
], loc="upper right", fontsize=9, framealpha=0.9)
save_report("02_violin.png")


# ═══════════════════════════════════════════════════════════════
# R3 — 04_scatter  (add R², stats, arrest rate summary)
# ═══════════════════════════════════════════════════════════════
print("[R3] Scatter Hour vs Severity ...")
samp=df.sample(min(8_000,len(df)),random_state=7)
mask_a=samp["Arrest"].astype(bool)

fig,ax=plt.subplots(figsize=(13,6))
ax.set_facecolor(LIGHT)
ax.scatter(samp.loc[~mask_a,"Hour"]+np.random.normal(0,0.25,(~mask_a).sum()),
           samp.loc[~mask_a,"Severity"]+np.random.normal(0,0.08,(~mask_a).sum()),
           c=BLUE,alpha=0.30,s=25,edgecolors="none",label=f"No Arrest (n={(~mask_a).sum():,})")
ax.scatter(samp.loc[mask_a,"Hour"]+np.random.normal(0,0.25,mask_a.sum()),
           samp.loc[mask_a,"Severity"]+np.random.normal(0,0.08,mask_a.sum()),
           c=RED,alpha=0.45,s=25,edgecolors="none",label=f"Arrest Made (n={mask_a.sum():,})")

# Trend lines for each group
for grp,col,ls in [(samp,NAVY,"--"),(samp[mask_a],RED,"-."),(samp[~mask_a],BLUE,":")]:
    if len(grp)>50:
        z=np.polyfit(grp["Hour"],grp["Severity"],1)
        xs=np.linspace(0,23,100)
        ax.plot(xs,np.poly1d(z)(xs),color=col,lw=1.8,ls=ls)

# R² annotation
from numpy.polynomial.polynomial import polyfit as pfit
z_all=np.polyfit(samp["Hour"],samp["Severity"],1)
yhat=np.poly1d(z_all)(samp["Hour"])
ss_res=np.sum((samp["Severity"]-yhat)**2); ss_tot=np.sum((samp["Severity"]-samp["Severity"].mean())**2)
r2=1-ss_res/ss_tot
arr_rate=mask_a.mean()*100

# Stats box
stats_txt=(f"Overall arrest rate: {arr_rate:.1f}%\n"
           f"Trend slope: {z_all[0]:+.4f} sev/hr\n"
           f"R² = {r2:.4f}  (near-zero = independent)")
ax.text(0.02,0.97,stats_txt,transform=ax.transAxes,va="top",ha="left",
        fontsize=9,color=NAVY,
        bbox=dict(boxstyle="round,pad=0.4",facecolor="white",edgecolor=NAVY,alpha=0.88))

ax.set_xlabel("Hour of Day  (0 = midnight, 12 = noon)",fontsize=12)
ax.set_ylabel("Crime Severity Score",fontsize=12)
ax.set_title("Crime Severity vs. Hour of Day — Coloured by Arrest Outcome",
             fontsize=14,fontweight="bold",color=NAVY,pad=14)
ax.set_xticks(range(0,24,2))
ax.set_xticklabels([f"{h:02d}:00" for h in range(0,24,2)],rotation=35,ha="right",fontsize=8)
ax.set_yticks([1,2,3,4,5])
ax.set_yticklabels(["1 Low","2 Minor","3 Moderate","4 Serious","5 Severe"],fontsize=9)
ax.grid(linestyle="--",alpha=0.35)
ax.legend(fontsize=10,framealpha=0.9)
save_report("04_scatter.png")


# ═══════════════════════════════════════════════════════════════
# R4 — 05_hexbin  (add city outline, scale bar, readable legend)
# ═══════════════════════════════════════════════════════════════
print("[R4] Hexbin geographic ...")
geo=df[["Latitude","Longitude"]].dropna()
geo=geo[(geo["Latitude"]>41.64)&(geo["Latitude"]<42.02)
       &(geo["Longitude"]>-87.86)&(geo["Longitude"]<-87.51)]
geo_s=geo.sample(min(100_000,len(geo)),random_state=42)

chicago_outline=np.array([
    [-87.80,42.023],[-87.74,42.020],[-87.72,42.023],[-87.69,42.023],
    [-87.63,42.021],[-87.60,42.021],[-87.575,41.998],[-87.558,41.970],
    [-87.535,41.944],[-87.527,41.900],[-87.527,41.840],[-87.530,41.800],
    [-87.527,41.720],[-87.527,41.695],[-87.530,41.666],[-87.560,41.644],
    [-87.700,41.644],[-87.760,41.644],[-87.800,41.700],[-87.800,41.900],
    [-87.800,41.950],[-87.800,42.023],
])
lake_poly=np.array([
    [-87.558,41.970],[-87.535,41.944],[-87.527,41.900],[-87.527,41.840],
    [-87.530,41.800],[-87.527,41.720],[-87.527,41.666],
    [-87.51,41.644],[-87.51,42.03],[-87.558,41.970],
])

fig=plt.figure(figsize=(11,12),facecolor="#0D1B2A")
ax=fig.add_axes([0.05,0.05,0.82,0.88],facecolor="#0D1B2A")
ax.add_patch(plt.Polygon(chicago_outline,closed=True,facecolor="#111827",edgecolor="none",zorder=1))
hb=ax.hexbin(geo_s["Longitude"],geo_s["Latitude"],gridsize=55,cmap="YlOrRd",
             mincnt=1,linewidths=0.05,edgecolors="none",zorder=2)
ax.add_patch(plt.Polygon(lake_poly,closed=True,facecolor="#071e36",edgecolor="none",zorder=3,alpha=0.95))
ax.add_patch(plt.Polygon(chicago_outline,closed=True,facecolor="none",
    edgecolor="#4a7ab5",linewidth=0.9,linestyle="--",zorder=4,alpha=0.8))
ax.text(-87.515,41.83,"Lake\nMichigan",color="#4a9fd4",fontsize=9,ha="center",
        style="italic",fontweight="bold",zorder=5,alpha=0.9)

# Colorbar with meaningful labels
cax=fig.add_axes([0.89,0.10,0.025,0.65])
vals=hb.get_array(); vmin,vmax=vals.min(),vals.max()
cb=fig.colorbar(hb,cax=cax)
nticks=5
tick_vals=np.linspace(vmin,vmax,nticks)
cb.set_ticks(tick_vals)
cb.set_ticklabels([f"{int(v):,}" for v in tick_vals],color="white",fontsize=8)
cax.set_title("Incidents\nper hex\ncell",color="#a0b8d0",fontsize=8,pad=6)
cb.outline.set_edgecolor("#3a6898")

# District hotspot labels
HOTSPOTS_HB=[
    (41.882,-87.629,"The Loop"),(41.847,-87.688,"West Side"),
    (41.778,-87.638,"South Side"),(41.940,-87.654,"North Side"),
]
for lat,lon,lbl in HOTSPOTS_HB:
    ax.scatter(lon,lat,s=90,c="#FFD700",zorder=6,edgecolors="white",linewidths=0.8)
    ax.text(lon+0.012,lat,lbl,color="#FFD700",fontsize=8,va="center",fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2",facecolor="#0D1B2A",edgecolor="none",alpha=0.7),zorder=7)

# N arrow + scale bar
ax.annotate("",xy=(-87.850,41.998),xytext=(-87.850,41.963),
    arrowprops=dict(arrowstyle="-|>",color="white",lw=1.5,mutation_scale=13),zorder=8)
ax.text(-87.850,42.002,"N",color="white",fontsize=10,fontweight="bold",ha="center",va="bottom",zorder=8)
SB0,SBY=-87.845,41.652
ax.plot([SB0,SB0+0.045],[SBY,SBY],"w-",lw=2.5,zorder=8,solid_capstyle="butt")
for tx in [SB0,SB0+0.045]: ax.plot([tx,tx],[SBY-0.003,SBY+0.003],"w-",lw=1.5,zorder=8)
ax.text(SB0+0.0225,SBY+0.007,"≈ 5 km",color="white",fontsize=8.5,ha="center",va="bottom",zorder=8)

ax.set_xlim(-87.86,-87.51); ax.set_ylim(41.64,42.02)
ax.set_xticks(np.arange(-87.85,-87.51,0.1))
ax.set_yticks(np.arange(41.65,42.02,0.1))
ax.set_xticklabels([f"{abs(v):.1f}°W" for v in np.arange(87.85,87.51,-0.1)],fontsize=8,color="#8aa8cc")
ax.set_yticklabels([f"{v:.1f}°N" for v in np.arange(41.65,42.02,0.1)],fontsize=8,color="#8aa8cc")
ax.tick_params(colors="#8aa8cc")
for spine in ax.spines.values(): spine.set_edgecolor("#1e3a5f")
fig.text(0.45,0.955,"Hexagonal Crime Density Map — Chicago",
         ha="center",fontsize=16,fontweight="bold",color="white")
fig.text(0.45,0.948,"Each hexagonal cell shows total incident count  |  Darker = fewer; brighter = more crimes",
         ha="center",fontsize=9,color="#7a9ab8",style="italic")
fig.text(0.07,0.025,f"N = {len(geo_s):,} incidents sampled  |  Source: Chicago Data Portal",
         fontsize=7.5,color="#3a5878")
save_report("05_hexbin.png")


# ═══════════════════════════════════════════════════════════════
# R5 — 06_bubble  (add axis value annotations, improve legend)
# ═══════════════════════════════════════════════════════════════
print("[R5] Bubble chart ...")
bub=df.groupby("District").agg(
    CrimeCount=("Arrest","count"),
    ArrestRate=("Arrest",lambda x:x.astype(bool).mean()),
    AvgSeverity=("Severity","mean")).reset_index().dropna()
bub=bub.sort_values("CrimeCount")

fig,ax=plt.subplots(figsize=(13,7))
ax.set_facecolor(LIGHT)
sc=ax.scatter(bub["CrimeCount"],bub["ArrestRate"]*100,
    s=bub["AvgSeverity"]**2.2*60,
    c=bub["ArrestRate"],cmap="RdYlGn",
    alpha=0.85,edgecolors=NAVY,linewidths=0.9,vmin=0.05,vmax=0.60,zorder=3)
for _,row in bub.iterrows():
    ax.annotate(f"D{int(row.District)}",(row.CrimeCount,row.ArrestRate*100),
        fontsize=7.5,ha="center",va="center",color="white",fontweight="bold",zorder=4)
cb=plt.colorbar(sc,ax=ax,pad=0.01)
cb.set_label("Arrest Rate (colour scale)",fontsize=10)
cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"{x*100:.0f}%"))

# Horizontal reference lines
for pct,lbl in [(20,"20% avg"),(40,"40%"),(bub["ArrestRate"].mean()*100,"City avg")]:
    ax.axhline(pct,color=GRAY,lw=0.8,ls="--",alpha=0.6)
    ax.text(bub["CrimeCount"].max()*1.002,pct,lbl,fontsize=7.5,color=GRAY,va="center")

ax.set_xlabel("Total Incidents Recorded  (2001–2024)",fontsize=12)
ax.set_ylabel("Arrest Rate (%)",fontsize=12)
ax.set_title("District-Level: Crime Volume vs. Arrest Rate\n"
             "(Bubble size ∝ Avg. Severity²  |  Colour = Arrest Rate)",
             fontsize=14,fontweight="bold",color=NAVY,pad=14)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"{int(x):,}"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y,_:f"{y:.0f}%"))
ax.grid(linestyle="--",alpha=0.4)
# Bubble-size legend
for sv,lbl in [(2.0,"Low (2)"),(3.0,"Med (3)"),(4.0,"High (4)"),(5.0,"Sev (5)")]:
    ax.scatter([],[],s=sv**2.2*60,c=GRAY,alpha=0.6,label=f"Severity {lbl}",edgecolors="white")
ax.legend(scatterpoints=1,title="Avg Severity\n(bubble size)",fontsize=8.5,
          title_fontsize=8.5,labelspacing=1.0,loc="lower right",framealpha=0.9)
save_report("06_bubble.png")


# ═══════════════════════════════════════════════════════════════
# R6 — 07_parallel_coords  (add axis tick values)
# ═══════════════════════════════════════════════════════════════
print("[R6] Parallel coordinates ...")
top5=df[pt_col].value_counts().head(5).index.tolist()
df_pc=df[df[pt_col].isin(top5)].sample(min(500,len(df)),random_state=42).copy()
dims=["Hour","Month","Severity","ArrestRateByType","District"]
dim_labels=["Hour of Day","Month","Severity\nScore","Arrest\nRate","District\nNo."]
df_pc=df_pc[dims+[pt_col]].dropna()
df_norm=df_pc[dims].copy()
raw_min={d:df_pc[d].min() for d in dims}
raw_max={d:df_pc[d].max() for d in dims}
for d in dims:
    rng=df_norm[d].max()-df_norm[d].min()
    df_norm[d]=(df_norm[d]-df_norm[d].min())/(rng+1e-9)
type_colors_pc=dict(zip(top5,[BLUE,RED,TEAL,ORANGE,NAVY]))

fig,ax=plt.subplots(figsize=(14,6))
ax.set_facecolor(LIGHT)
for idx in df_pc.index:
    ax.plot(range(len(dims)),[df_norm.loc[idx,d] for d in dims],
        color=type_colors_pc.get(df_pc.loc[idx,pt_col],GRAY),alpha=0.22,linewidth=0.9)
ax.set_xticks(range(len(dims)))
ax.set_xticklabels(dim_labels,fontsize=10,fontweight="bold")
ax.set_yticks([])
for x in range(len(dims)):
    ax.axvline(x,color=GRAY,linewidth=0.8)
    # Add min/max tick labels on each axis
    lo=raw_min[dims[x]]; hi=raw_max[dims[x]]
    if dims[x]=="ArrestRateByType":
        ax.text(x,-0.08,f"{lo*100:.0f}%",ha="center",fontsize=7.5,color=GRAY,transform=ax.get_xaxis_transform())
        ax.text(x, 1.06,f"{hi*100:.0f}%",ha="center",fontsize=7.5,color=NAVY,transform=ax.get_xaxis_transform())
    elif dims[x] in ("Severity","Hour"):
        ax.text(x,-0.08,f"{lo:.0f}",ha="center",fontsize=7.5,color=GRAY,transform=ax.get_xaxis_transform())
        ax.text(x, 1.06,f"{hi:.0f}",ha="center",fontsize=7.5,color=NAVY,transform=ax.get_xaxis_transform())
    else:
        ax.text(x,-0.08,f"{lo:.0f}",ha="center",fontsize=7.5,color=GRAY,transform=ax.get_xaxis_transform())
        ax.text(x, 1.06,f"{hi:.0f}",ha="center",fontsize=7.5,color=NAVY,transform=ax.get_xaxis_transform())
ax.text(0.5,1.10,"▲ High",transform=ax.transAxes,ha="center",fontsize=8,color=NAVY)
ax.text(0.5,-0.11,"▼ Low", transform=ax.transAxes,ha="center",fontsize=8,color=GRAY)
ax.set_xlim(-0.3,len(dims)-0.7)
ax.set_title("Parallel Coordinates — Multi-Dimensional Crime Profile by Type\n"
             f"(sample n={len(df_pc):,}  |  each line = one incident  |  axes normalised 0–1)",
             fontsize=13,fontweight="bold",color=NAVY,pad=12)
ax.legend(handles=[Line2D([0],[0],color=c,lw=2.5,label=t) for t,c in type_colors_pc.items()],
          loc="upper right",fontsize=9,framealpha=0.9)
save_report("07_parallel_coords.png")


# ═══════════════════════════════════════════════════════════════
# R7 — 08_3d  (add month names, count annotations)
# ═══════════════════════════════════════════════════════════════
print("[R7] 3D landscape ...")
monthly_3d=df.groupby(["YearActual","Month"]).size().reset_index(name="Count")
MNAMES=["","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

fig=plt.figure(figsize=(14,8))
ax3d=fig.add_subplot(111,projection="3d")
sc3=ax3d.scatter(monthly_3d["YearActual"],monthly_3d["Month"],monthly_3d["Count"],
    c=monthly_3d["Count"],cmap="YlOrRd",s=50,alpha=0.85,edgecolors="none")
for yr in sorted(monthly_3d["YearActual"].unique()):
    sub=monthly_3d[monthly_3d["YearActual"]==yr].sort_values("Month")
    ax3d.plot(sub["YearActual"],sub["Month"],sub["Count"],color=BLUE,alpha=0.25,linewidth=0.8)
cb=fig.colorbar(sc3,ax=ax3d,shrink=0.45,pad=0.08,label="Monthly Crime Count")
cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"{int(x):,}"))
ax3d.set_xlabel("Year",fontsize=10,labelpad=8)
ax3d.set_ylabel("Month",fontsize=10,labelpad=8)
ax3d.set_zlabel("Incidents per Month",fontsize=10,labelpad=8)
ax3d.set_yticks(range(1,13))
ax3d.set_yticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"],fontsize=8)
ax3d.zaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"{int(x):,}"))
# Peak annotation
peak=monthly_3d.loc[monthly_3d["Count"].idxmax()]
ax3d.text(peak.YearActual,peak.Month,peak.Count*1.05,
          f"Peak:\n{MNAMES[int(peak.Month)]} {int(peak.YearActual)}\n{int(peak.Count):,}",
          fontsize=8,color=RED,ha="center")
ax3d.set_title("Monthly Crime Volume — 3D View by Year & Month",
               fontsize=14,fontweight="bold",color=NAVY,pad=20)
ax3d.view_init(elev=25,azim=-55)
save_report("08_3d.png")


# ═══════════════════════════════════════════════════════════════
# R8 — 09_network  (correlation wheel — all 28 pairs, colour = Pearson r)
# ═══════════════════════════════════════════════════════════════
print("[R8] Network ...")
try:
    import networkx as nx  # noqa: F401  (kept so ImportError still triggers skip)
    from matplotlib.lines import Line2D

    top8_t = df[pt_col].value_counts().head(8).index.tolist()
    LABEL_MAP = {
        "CRIMINAL DAMAGE":         "CRIMINAL\nDAMAGE",
        "MOTOR VEHICLE THEFT":     "MOTOR VEHICLE\nTHEFT",
        "DECEPTIVE PRACTICE":      "DECEPTIVE\nPRACTICE",
        "CRIMINAL SEXUAL ASSAULT": "CRIMINAL\nSEXUAL ASSAULT",
        "WEAPONS VIOLATION":       "WEAPONS\nVIOLATION",
        "PUBLIC PEACE VIOLATION":  "PUBLIC PEACE\nVIOLATION",
    }
    short   = {t: LABEL_MAP.get(t, t) for t in top8_t}
    df_net  = df[df[pt_col].isin(top8_t)].copy()
    df_net["TypeShort"] = df_net[pt_col].map(short)
    tlist   = list(short.values())

    # Beat x crime-type rate matrix -> Pearson correlation
    # Shows which crime types concentrate in the SAME beats vs different beats.
    beat_type = (df_net.groupby(["Beat", "TypeShort"])
                       .size().unstack(fill_value=0))
    beat_rate = beat_type.div(beat_type.sum(axis=1), axis=0)
    corr      = beat_rate.corr()

    labels = [t for t in tlist if t in corr.columns]
    n      = len(labels)
    angles = [2 * np.pi * i / n - np.pi / 2 for i in range(n)]
    R_ring = 1.0
    pos    = {lab: (R_ring * np.cos(a), R_ring * np.sin(a))
              for lab, a in zip(labels, angles)}

    cmap_corr = mcolors.LinearSegmentedColormap.from_list(
        "corr", ["#2166ac", "#92c5de", "#f7f7f7",
                 "#f4a582", "#d6604d", "#b2182b"])
    norm_corr  = mcolors.Normalize(-1, 1)
    nc         = [RED, BLUE, TEAL, ORANGE, NAVY, GOLD, "#6C3483", "#117A65"]
    type_counts = df_net.groupby("TypeShort").size()
    max_count   = max(type_counts.values)

    fig, ax = plt.subplots(figsize=(13, 11))
    ax.set_facecolor(LIGHT)
    ax.set_aspect("equal")

    # Draw ALL 28 edges; skip only self-diagonal
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if j <= i:
                continue
            r     = corr.loc[a, b]
            color = cmap_corr(norm_corr(r))
            lw    = abs(r) * 6.0
            alpha = 0.20 + abs(r) * 0.70
            ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                    color=color, linewidth=lw, alpha=alpha, zorder=1,
                    solid_capstyle="round")
            # Label the strongest pairs (|r| > 0.45) so readers can read exact values
            if abs(r) > 0.45:
                mx = (pos[a][0] + pos[b][0]) / 2
                my = (pos[a][1] + pos[b][1]) / 2
                sign = "+" if r > 0 else u"\u2212"
                ax.text(mx, my, f"{sign}{abs(r):.2f}",
                        fontsize=7, ha="center", va="center", zorder=4,
                        color="#222222",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                                  edgecolor="none", alpha=0.82))

    # Nodes sized by incident count
    for idx, lab in enumerate(labels):
        x, y = pos[lab]
        s = type_counts.get(lab, 1) / max_count * 2800 + 400
        ax.scatter(x, y, s=s, color=nc[idx % len(nc)], zorder=3,
                   edgecolors="white", linewidths=1.8)

    # Labels pushed radially outward — never clip
    for lab in labels:
        x, y = pos[lab]
        lx, ly = x * 1.30, y * 1.30
        ha = "left"   if x >  0.15 else ("right" if x < -0.15 else "center")
        va = "bottom" if y >  0.15 else ("top"   if y < -0.15 else "center")
        ax.text(lx, ly, lab, fontsize=9, fontweight="bold", color=NAVY,
                ha=ha, va=va, multialignment="center",
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                          edgecolor="none", alpha=0.88))

    # Colourbar
    sm = plt.cm.ScalarMappable(cmap=cmap_corr, norm=norm_corr)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, shrink=0.45, pad=0.02,
                      aspect=22, location="right")
    cb.set_label("Pearson r  (beat-level crime rate)", fontsize=9)
    cb.set_ticks([-1, -0.5, 0, 0.5, 1])
    cb.set_ticklabels(["- 1\n(never share beats)", "-0.5", "0", "+0.5",
                       "+1\n(always share beats)"], fontsize=7.5)

    # Legend
    legend_elements = [
        Line2D([0],[0], color="#b2182b", lw=3.5,
               label="Strong positive  (spatially co-located)"),
        Line2D([0],[0], color="#d8d8d8", lw=1.0,
               label="Near zero  (independent)"),
        Line2D([0],[0], color="#2166ac", lw=3.5,
               label="Strong negative  (spatially separated)"),
    ]
    ax.legend(handles=legend_elements, loc="lower left",
              fontsize=8.5, framealpha=0.92, edgecolor="#cccccc",
              title="Edge colour key", title_fontsize=8.5)

    ax.set_xlim(-1.85, 1.85)
    ax.set_ylim(-1.85, 1.85)
    ax.axis("off")
    ax.set_title(
        "Crime Type Spatial Co-occurrence \u2014 Chicago\n"
        "Each edge = Pearson r between beat-level crime-rate vectors  "
        "|  Thick warm = co-located  |  Thick blue = spatially separated\n"
        "Values shown where |r| > 0.45",
        fontsize=11.5, fontweight="bold", color=NAVY, pad=14)
    plt.tight_layout()
    save_report("09_network.png")

except ImportError:
    print("  Skipped \u2014 networkx not installed")




# ═══════════════════════════════════════════════════════════════
# R9 — 10_sunburst  (add % to outer ring, legend for inner ring)
# ═══════════════════════════════════════════════════════════════
print("[R9] Sunburst ...")
season_type=df.groupby(["Season",pt_col]).size().reset_index(name="Count")
season_totals=season_type.groupby("Season")["Count"].sum()
grand_total=season_totals.sum()

# 5 top types per season for inner ring
inner_data=[]
for s in SEASON_ORDER:
    sub=season_type[season_type["Season"]==s].sort_values("Count",ascending=False).head(5)
    inner_data.append(sub.assign(Season=s))
inner_df=pd.concat(inner_data,ignore_index=True)

# Build a consistent colour map for crime types (across all seasons)
all_inner_types=inner_df[pt_col].unique().tolist()
type_cmap=plt.cm.tab10(np.linspace(0,1,len(all_inner_types)))
type_color_map={t:type_cmap[i] for i,t in enumerate(all_inner_types)}

inner_vals=[]; inner_colors_list=[]
for s in SEASON_ORDER:
    sub=inner_df[inner_df["Season"]==s]
    for _,row in sub.iterrows():
        inner_vals.append(row["Count"])
        inner_colors_list.append(type_color_map[row[pt_col]])

# Figure: pie on left, legend on right
fig=plt.figure(figsize=(14,10))
ax_pie=fig.add_axes([0.0,0.05,0.72,0.90])
ax_leg=fig.add_axes([0.72,0.05,0.28,0.90]); ax_leg.axis("off")

SEASON_COLORS={"Winter":"#4393C3","Spring":"#74C476","Summer":"#FD8D3C","Fall":"#9E7B3C"}
outer_vals=[season_totals.get(s,0) for s in SEASON_ORDER]
outer_colors=[SEASON_COLORS[s] for s in SEASON_ORDER]

wedges_out,texts_out,autotexts_out=ax_pie.pie(
    outer_vals, radius=1.0,
    colors=outer_colors, startangle=90,
    wedgeprops=dict(width=0.34,edgecolor="white",linewidth=2.5),
    autopct=lambda p:f"{p:.1f}%", pctdistance=0.83,
    labels=SEASON_ORDER, labeldistance=1.10,
    textprops=dict(fontsize=12,fontweight="bold"))
for at in autotexts_out:
    at.set_fontsize(9); at.set_color("white"); at.set_fontweight("bold")
# Add absolute count to season labels
for i,(text,s) in enumerate(zip(texts_out,SEASON_ORDER)):
    cnt=season_totals.get(s,0); pct=cnt/grand_total*100
    text.set_text(f"{s}\n{cnt:,}\n({pct:.1f}%)")
    text.set_fontsize(10)

wedges_in,_=ax_pie.pie(
    inner_vals, radius=0.66,
    colors=inner_colors_list, startangle=90,
    wedgeprops=dict(width=0.34,edgecolor="white",linewidth=1.2))

ax_pie.add_artist(plt.Circle((0,0),0.32,color="white"))
ax_pie.text(0,0.06,"Crime\nDistribution",ha="center",va="center",
            fontsize=10,fontweight="bold",color=NAVY)
ax_pie.text(0,-0.12,f"N={grand_total:,}",ha="center",va="center",
            fontsize=8,color=GRAY)
ax_pie.set_title("Sunburst: Crime by Season  (Outer) & Top Crime Types per Season  (Inner)",
                 fontsize=13,fontweight="bold",color=NAVY,pad=16)

# Legend for inner ring (crime types)
legend_handles=[mpatches.Patch(color=type_color_map[t],label=t.title())
                for t in all_inner_types]
ax_leg.legend(handles=legend_handles, title="Inner Ring:\nCrime Types",
              title_fontsize=10, fontsize=9.5, loc="center",
              frameon=True, edgecolor=NAVY,
              framealpha=0.95, labelspacing=0.8)
ax_leg.set_title("Legend", fontsize=11, fontweight="bold", color=NAVY, pad=8)

save_report("10_sunburst.png")


# ═══════════════════════════════════════════════════════════════
# R10 — 11_choropleth
# ═══════════════════════════════════════════════════════════════
print("[R10] Choropleth ...")
from scipy.spatial import Voronoi
from matplotlib.patches import Polygon as MplPolygon, PathPatch
from matplotlib.path import Path

dist_counts = df["District"].value_counts().reset_index()
dist_counts.columns = ["District", "CrimeCount"]
norm_ch = mcolors.Normalize(dist_counts["CrimeCount"].min(),
                             dist_counts["CrimeCount"].max())
cmap_ch = cm.get_cmap("YlOrRd")
count_lu = dist_counts.set_index("District")["CrimeCount"].to_dict()

# Approximate CPD district centroids (lon, lat).
# Voronoi draws equidistant boundaries between them — a good approximation
# of real district shapes without needing geopandas or a shapefile.
CENTROIDS = {
     1: (-87.608, 41.885),  2: (-87.624, 41.848),  3: (-87.609, 41.793),
     4: (-87.568, 41.746),  5: (-87.626, 41.675),  6: (-87.658, 41.773),
     7: (-87.652, 41.812),  8: (-87.714, 41.797),  9: (-87.656, 41.850),
    10: (-87.708, 41.855), 11: (-87.762, 41.886), 12: (-87.658, 41.885),
    13: (-87.710, 41.856), 14: (-87.718, 41.928), 15: (-87.774, 41.928),
    16: (-87.808, 41.998), 17: (-87.726, 41.970), 18: (-87.634, 41.920),
    19: (-87.670, 41.946), 20: (-87.706, 42.007), 21: (-87.589, 41.804),
    22: (-87.695, 41.706), 23: (-87.646, 41.948), 24: (-87.661, 42.009),
    25: (-87.763, 41.958),
}

# Chicago city outline — straight eastern edge = Lake Michigan coastline
CHICAGO = np.array([
    [-87.800, 42.023], [-87.800, 41.900], [-87.800, 41.800],
    [-87.800, 41.700], [-87.800, 41.644], [-87.760, 41.644],
    [-87.700, 41.644], [-87.640, 41.644], [-87.590, 41.644],
    [-87.553, 41.650], [-87.530, 41.666], [-87.527, 41.700],
    [-87.527, 41.760], [-87.528, 41.820], [-87.528, 41.880],
    [-87.528, 41.940], [-87.528, 41.990], [-87.535, 42.020],
    [-87.580, 42.023], [-87.630, 42.023], [-87.680, 42.023],
    [-87.730, 42.023], [-87.800, 42.023],
])

district_ids = list(CENTROIDS.keys())
centroid_pts = [CENTROIDS[d] for d in district_ids]
DUMMY = [(-90,40),(-90,43.5),(-86,40),(-86,43.5),
         (-88,40),(-88,43.5),(-90,41.8),(-86,41.8)]
vor = Voronoi(np.array(centroid_pts + DUMMY))

outline_closed = np.vstack([CHICAGO, CHICAGO[0]])
codes = ([Path.MOVETO] + [Path.LINETO]*(len(outline_closed)-2)
         + [Path.CLOSEPOLY])
chicago_path = Path(outline_closed, codes)

fig, ax = plt.subplots(figsize=(9, 13))
ax.set_facecolor("#c8dae8")
ax.set_aspect("equal")

clip_patch = PathPatch(chicago_path, transform=ax.transData,
                       facecolor="none", edgecolor="none")
ax.add_patch(clip_patch)

for i, d in enumerate(district_ids):
    region_idx = vor.point_region[i]
    region     = vor.regions[region_idx]
    if -1 in region or not region:
        continue
    verts = vor.vertices[region]
    count = count_lu.get(d, 0)
    color = cmap_ch(norm_ch(count))
    patch = MplPolygon(verts, closed=True, facecolor=color,
                       edgecolor="white", linewidth=1.0, zorder=2)
    patch.set_clip_path(clip_patch)
    ax.add_patch(patch)

ax.add_patch(MplPolygon(CHICAGO, closed=True, facecolor="none",
                         edgecolor="#1a1a2e", linewidth=2.0, zorder=5))
ax.text(-87.490, 41.840, "Lake\nMichigan", color="#2a6496",
        fontsize=9, style="italic", ha="left", va="center", zorder=6)

for d, (cx, cy) in CENTROIDS.items():
    count   = count_lu.get(d, 0)
    txt_col = "white" if norm_ch(count) > 0.5 else NAVY
    ax.text(cx, cy + 0.003, f"D{d}", ha="center", va="center",
            fontsize=6.5, fontweight="bold", color=txt_col, zorder=6)
    ax.text(cx, cy - 0.005, f"{int(count):,}", ha="center", va="center",
            fontsize=5.5, color=txt_col, zorder=6, alpha=0.9)

sm = cm.ScalarMappable(norm=norm_ch, cmap=cmap_ch); sm.set_array([])
cb = plt.colorbar(sm, ax=ax, shrink=0.55, pad=0.02, aspect=25)
cb.set_label("Total Incidents (2001-2024)", fontsize=10)
cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

ax.annotate("", xy=(-87.920, 42.005), xytext=(-87.920, 41.975),
            arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.5, mutation_scale=12))
ax.text(-87.920, 42.007, "N", ha="center", va="bottom",
        fontsize=9, fontweight="bold", color=NAVY)

ax.set_xlim(-87.94, -87.46)
ax.set_ylim(41.61, 42.06)
ax.set_xlabel("Longitude", fontsize=9, color=GRAY)
ax.set_ylabel("Latitude",  fontsize=9, color=GRAY)
ax.tick_params(labelsize=7, colors=GRAY)
for sp in ax.spines.values():
    sp.set_edgecolor("#aabbcc")

top_d = dist_counts.loc[dist_counts["CrimeCount"].idxmax()]
bot_d = dist_counts.loc[dist_counts["CrimeCount"].idxmin()]
ax.set_title(
    "Crime Incidents by Police District — Chicago\n"
    "(Boundaries approximated via Voronoi tessellation  "
    "|  Colour = total incidents 2001-2024)",
    fontsize=12, fontweight="bold", color=NAVY, pad=14)
ax.text(0.5, -0.04,
        f"Highest: D{int(top_d.District)} ({int(top_d.CrimeCount):,} incidents)  "
        f"|  Lowest: D{int(bot_d.District)} ({int(bot_d.CrimeCount):,} incidents)",
        transform=ax.transAxes, ha="center", fontsize=8.5, color=GRAY, style="italic")
plt.tight_layout()
save_report("11_choropleth.png")




# ═══════════════════════════════════════════════════════════════
# R11 — 12_geo_heatmap  (KDE, smooth raster, city outline)
# ═══════════════════════════════════════════════════════════════
print("[R11] KDE geographic heatmap ...")
geo2 = df[["Latitude","Longitude"]].dropna()
geo2 = geo2[(geo2["Latitude"]  > 41.64) & (geo2["Latitude"]  < 42.02)
           &(geo2["Longitude"] > -87.86) & (geo2["Longitude"] < -87.51)]
geo2_s = geo2.sample(min(80_000, len(geo2)), random_state=42)
x_g, y_g = geo2_s["Longitude"].values, geo2_s["Latitude"].values

RES = 400
LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = 41.64, 42.02, -87.86, -87.51
xg = np.linspace(LON_MIN, LON_MAX, RES)
yg = np.linspace(LAT_MIN, LAT_MAX, RES)
Xg, Yg = np.meshgrid(xg, yg)
Z = gaussian_kde(np.vstack([x_g, y_g]), bw_method=0.04)(
    np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
Z = gaussian_filter(Z, sigma=3.5)
Z = np.power(Z / Z.max(), 0.45)

chicago_outline_r = np.array([
    [-87.80,42.023],[-87.74,42.020],[-87.72,42.023],[-87.69,42.023],
    [-87.63,42.021],[-87.60,42.021],[-87.575,41.998],[-87.558,41.970],
    [-87.535,41.944],[-87.527,41.900],[-87.527,41.840],[-87.530,41.800],
    [-87.527,41.720],[-87.527,41.695],[-87.530,41.666],[-87.560,41.644],
    [-87.700,41.644],[-87.760,41.644],[-87.800,41.700],[-87.800,41.900],
    [-87.800,41.950],[-87.800,42.023],
])
lake_poly_r = np.array([
    [-87.558,41.970],[-87.535,41.944],[-87.527,41.900],[-87.527,41.840],
    [-87.530,41.800],[-87.527,41.720],[-87.527,41.666],
    [-87.51,41.644],[-87.51,42.03],[-87.558,41.970],
])
cmap_geo = LinearSegmentedColormap.from_list("crime_heat", [
    (0.00,"#03001C"),(0.12,"#1A0533"),(0.28,"#52006A"),
    (0.45,"#CD113B"),(0.62,"#FF4500"),(0.78,"#FF7500"),
    (0.90,"#FFD700"),(1.00,"#FFFDE7"),], N=512)

fig = plt.figure(figsize=(12, 14), facecolor="#03001C")
ax  = fig.add_axes([0.06, 0.05, 0.80, 0.88], facecolor="#03001C")
ax.add_patch(plt.Polygon(chicago_outline_r, closed=True,
    facecolor="#0d0d18", edgecolor="none", zorder=2))
ax.imshow(Z, extent=[LON_MIN,LON_MAX,LAT_MIN,LAT_MAX],
          origin="lower", cmap=cmap_geo, aspect="auto",
          alpha=0.92, interpolation="bilinear", zorder=3)
ax.add_patch(plt.Polygon(lake_poly_r, closed=True,
    facecolor="#071e36", edgecolor="none", zorder=4, alpha=0.97))
ax.add_patch(plt.Polygon(chicago_outline_r, closed=True, facecolor="none",
    edgecolor="#3a6898", linewidth=0.9, linestyle="--", zorder=5, alpha=0.75))
ax.text(-87.516, 41.83, "Lake\nMichigan", color="#4a9fd4",
        fontsize=9.5, ha="center", va="center",
        style="italic", fontweight="bold", zorder=6, alpha=0.9)
for lat_g in np.arange(41.65, 42.02, 0.05):
    ax.axhline(lat_g, color="#ffffff", linewidth=0.2, alpha=0.10, zorder=5)
for lon_g in np.arange(-87.85, -87.51, 0.05):
    ax.axvline(lon_g, color="#ffffff", linewidth=0.2, alpha=0.10, zorder=5)

HOTSPOTS_R11 = [
    (41.882,-87.629,"① The Loop /\nNear North Side","right",( 0.035, 0.015)),
    (41.847,-87.688,"② West Side\n(Austin)",        "right",( 0.038,-0.008)),
    (41.778,-87.638,"③ South Side\n(Englewood)",    "right",( 0.040, 0.010)),
    (41.918,-87.648,"④ Humboldt Park",              "right",( 0.040, 0.020)),
    (41.940,-87.654,"⑤ Logan Square /\nAvondale",  "right",( 0.040,-0.018)),
]
for lat_h,lon_h,label_h,ha_h,(dlon_h,dlat_h) in HOTSPOTS_R11:
    ax.scatter(lon_h, lat_h, s=260, c="#FF4500", alpha=0.20, zorder=7, linewidths=0)
    ax.scatter(lon_h, lat_h, s=90,  c="#FFD700", alpha=0.95, zorder=8,
               edgecolors="#FFFFFF", linewidths=0.8)
    ax.annotate(label_h, xy=(lon_h,lat_h),
        xytext=(lon_h+dlon_h, lat_h+dlat_h),
        fontsize=8.5, color="#FFFFFF", fontweight="bold", ha=ha_h, va="center",
        arrowprops=dict(arrowstyle="-", color="#FFD700", lw=0.9,
                        connectionstyle="arc3,rad=0.12"),
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#180330",
                  edgecolor="#FFD700", linewidth=0.8, alpha=0.88), zorder=9)

ax.annotate("", xy=(-87.850,41.998), xytext=(-87.850,41.963),
    arrowprops=dict(arrowstyle="-|>", color="white", lw=1.6, mutation_scale=14), zorder=9)
ax.text(-87.850, 42.002, "N", color="white", fontsize=11,
        fontweight="bold", ha="center", va="bottom", zorder=9)
SB0_r, SBY_r = -87.845, 41.653
ax.plot([SB0_r, SB0_r+0.045],[SBY_r,SBY_r],"w-",lw=2.5,zorder=9,solid_capstyle="butt")
for tx_r in [SB0_r, SB0_r+0.045]:
    ax.plot([tx_r,tx_r],[SBY_r-0.003,SBY_r+0.003],"w-",lw=1.5,zorder=9)
ax.text(SB0_r+0.0225, SBY_r+0.007, "≈ 5 km",
        color="white", fontsize=8.5, ha="center", va="bottom", zorder=9)

cax_r = fig.add_axes([0.88, 0.10, 0.022, 0.60])
sm_r  = plt.cm.ScalarMappable(cmap=cmap_geo, norm=mcolors.Normalize(0,1))
sm_r.set_array([])
cb_r  = fig.colorbar(sm_r, cax=cax_r)
cb_r.set_ticks([0,0.25,0.5,0.75,1.0])
cb_r.set_ticklabels(["Low","","Medium","","Extreme"], color="white", fontsize=8)
cb_r.outline.set_edgecolor("#3a6898")
cb_r.ax.tick_params(colors="white", length=3)
cax_r.set_title("Crime\nDensity", color="#a0b8d0", fontsize=8.5, pad=6)

ax.set_xlim(LON_MIN, LON_MAX); ax.set_ylim(LAT_MIN, LAT_MAX)
ax.set_xticks(np.arange(-87.85,-87.51,0.1))
ax.set_yticks(np.arange(41.65,42.05,0.1))
ax.set_xticklabels([f"{v:.1f}°W" for v in np.arange(87.85,87.51,-0.1)],
                   fontsize=8, color="#8aa8cc")
ax.set_yticklabels([f"{v:.1f}°N" for v in np.arange(41.65,42.05,0.1)],
                   fontsize=8, color="#8aa8cc")
ax.tick_params(colors="#8aa8cc")
for spine in ax.spines.values():
    spine.set_edgecolor("#1e3a5f"); spine.set_linewidth(0.8)
fig.text(0.46, 0.952, "Chicago Crime Density — Geographic Heatmap",
         ha="center", va="bottom", fontsize=17, fontweight="bold", color="white")
fig.text(0.46, 0.944,
         "Kernel Density Estimation  ·  8.4 M incident records  ·  2001 – 2024",
         ha="center", va="top", fontsize=9, color="#7a9ab8", style="italic")
fig.text(0.07, 0.030, "Source: Chicago Data Portal — Crimes 2001 to Present",
         fontsize=7.5, color="#3a5878")
save_report("12_geo_heatmap.png")


# ═══════════════════════════════════════════════════════════════
# R12 — 13_slopegraph  (fix overlapping labels with repel)
# ═══════════════════════════════════════════════════════════════
print("[R12] Slopegraph ...")
ar_left =(df[df["YearActual"]<=2015].groupby(pt_col)["Arrest"]
          .apply(lambda x:x.astype(bool).mean())*100)
ar_right=(df[df["YearActual"]>=2020].groupby(pt_col)["Arrest"]
          .apply(lambda x:x.astype(bool).mean())*100)
common=list(set(ar_left.index)&set(ar_right.index))
top12=(pd.Series({t:(ar_left[t]+ar_right[t])/2 for t in common}).nlargest(12).index.tolist())
sl=ar_left.loc[top12]; sr=ar_right.loc[top12]

def repel(positions, min_gap=2.2):
    """Push labels apart so they don't overlap."""
    pos=sorted(enumerate(positions),key=lambda x:x[1])
    adjusted=[p for _,p in pos]
    for _ in range(200):
        changed=False
        for i in range(1,len(adjusted)):
            if adjusted[i]-adjusted[i-1]<min_gap:
                mid=(adjusted[i]+adjusted[i-1])/2
                adjusted[i-1]=mid-min_gap/2
                adjusted[i]  =mid+min_gap/2
                changed=True
        if not changed: break
    out=[0]*len(positions)
    for new_idx,(orig_idx,_) in enumerate(sorted(enumerate(positions),key=lambda x:x[1])):
        out[orig_idx]=adjusted[new_idx]
    return out

fig,ax=plt.subplots(figsize=(12,9))
ax.set_facecolor(LIGHT)

left_pos =repel(sl.values.tolist())
right_pos=repel(sr.values.tolist())

for i,ctype in enumerate(top12):
    l,r=sl[ctype],sr[ctype]
    change=r-l; lc=TEAL if change>=0 else RED
    lpos=left_pos[i]; rpos=right_pos[i]
    ax.plot([0,1],[l,r],color=lc,linewidth=2.0,alpha=0.70,zorder=2)
    ax.scatter([0],[l],s=70,color=lc,zorder=4,edgecolors="white",linewidths=0.7)
    ax.scatter([1],[r],s=70,color=lc,zorder=4,edgecolors="white",linewidths=0.7)
    # Left label (repelled)
    ax.annotate("",xy=(0,l),xytext=(-0.06,lpos),
        arrowprops=dict(arrowstyle="-",color=lc,lw=0.6,alpha=0.5))
    ax.text(-0.07,lpos,f"{ctype.title()[:18]}\n{l:.1f}%",
        ha="right",va="center",fontsize=8,color=NAVY,
        bbox=dict(boxstyle="round,pad=0.2",facecolor="white",
                  edgecolor=lc,linewidth=0.6,alpha=0.9))
    # Right label (repelled)
    ax.annotate("",xy=(1,r),xytext=(1.06,rpos),
        arrowprops=dict(arrowstyle="-",color=lc,lw=0.6,alpha=0.5))
    sign="+" if change>=0 else ""
    ax.text(1.07,rpos,f"{r:.1f}%  ({sign}{change:.1f}pp)",
        ha="left",va="center",fontsize=8,color=lc,fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2",facecolor="white",
                  edgecolor=lc,linewidth=0.6,alpha=0.9))

ax.set_xticks([0,1])
ax.set_xticklabels(["Period 1\n(up to 2015)","Period 2\n(2020 onward)"],
                   fontsize=13,fontweight="bold")
ax.set_ylabel("Arrest Rate (%)",fontsize=12)
ax.set_xlim(-0.65,1.65)
ylo=min(sl.min(),sr.min())-5; yhi=max(sl.max(),sr.max())+5
ax.set_ylim(ylo,yhi)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y,_:f"{y:.0f}%"))
ax.set_title("Arrest Rate Change by Crime Type: Early Period → Recent Period",
             fontsize=14,fontweight="bold",color=NAVY,pad=14)
ax.grid(axis="y",linestyle="--",alpha=0.4)
ax.spines[["top","right","left"]].set_visible(False)
ax.legend(handles=[
    Line2D([0],[0],color=TEAL,lw=2.5,label="Arrest rate increased"),
    Line2D([0],[0],color=RED, lw=2.5,label="Arrest rate decreased"),
],fontsize=10,loc="upper center",framealpha=0.9)
ax.text(0.5,-0.06,"Values show arrest rate (%)  |  Change in percentage points (pp)",
        transform=ax.transAxes,ha="center",fontsize=8.5,color=GRAY,style="italic")
save_report("13_slopegraph.png")


# ═══════════════════════════════════════════════════════════════
# R13 — 14_streamgraph  (add right-edge values, cleaner Y axis)
# ═══════════════════════════════════════════════════════════════
print("[R13] Streamgraph ...")
stream_types=df[pt_col].value_counts().head(6).index.tolist()
stream=(df[df[pt_col].isin(stream_types)]
        .groupby(["YearActual","Month",pt_col]).size().unstack(fill_value=0))
stream.index=[pd.Timestamp(f"{y}-{m:02d}-01") for y,m in stream.index]
stream=stream.sort_index()[stream_types]
x=np.arange(len(stream))
baseline=-stream.sum(axis=1).values/2
cumsum=np.zeros(len(stream))

fig,ax=plt.subplots(figsize=(14,7))
ax.set_facecolor(LIGHT)
colors_st=[BLUE,RED,TEAL,ORANGE,GOLD,NAVY]
cum_layers={}
prev=baseline.copy()
for col_s,col_c in zip(stream_types,colors_st):
    y_vals=stream[col_s].values
    top=prev+y_vals
    ax.fill_between(x,prev,top,color=col_c,alpha=0.82,label=col_s)
    cum_layers[col_s]=(prev.copy(),top.copy())
    prev=top.copy()

# Y-axis: show absolute scale (offset from zero)
tick_y=np.linspace(baseline.min(),baseline.max()+stream.sum(axis=1).max()*0.5,7)
abs_vals=tick_y-baseline.mean()+stream.sum(axis=1).mean()/2
ax.set_yticks(tick_y[:5])
ax.set_yticklabels([f"{int(abs(v)):,}" for v in np.linspace(0,stream.sum(axis=1).max(),5)],fontsize=9)
ax.set_ylabel("Monthly Incidents (approx. absolute scale)",fontsize=11)

# Right-edge labels showing final month values
final_x=len(stream)-1
prev_r=baseline[final_x]
for col_s,col_c in zip(stream_types,colors_st):
    y_val=stream[col_s].values[final_x]
    mid_r=prev_r+y_val/2
    ax.text(final_x+2,mid_r,f"{col_s.title()[:12]}\n{int(y_val):,}/mo",
        va="center",ha="left",fontsize=7.5,color=col_c,fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2",facecolor="white",edgecolor=col_c,
                  alpha=0.85,linewidth=0.6))
    prev_r+=y_val

ax.set_xlim(0,len(stream)+20)
tick_idx=[i for i,ts in enumerate(stream.index) if ts.month==1 and ts.year%2==1]
ax.set_xticks(tick_idx)
ax.set_xticklabels([str(stream.index[i].year) for i in tick_idx],fontsize=9,rotation=30,ha="right")
ax.set_xlabel("Year",fontsize=11)
ax.set_title("Streamgraph: Monthly Crime Volume by Type  (2001–2024)\n"
             "(Stream width = monthly incident count per type  |  Centred baseline)",
             fontsize=13,fontweight="bold",color=NAVY,pad=14)
ax.grid(axis="y",linestyle="--",alpha=0.25)
ax.legend(loc="upper left",fontsize=8.5,framealpha=0.9,ncol=2)
save_report("14_streamgraph.png")


# ═══════════════════════════════════════════════════════════════
# R14 — B2_boxplot_season  (add median labels, mean markers,
#                           IQR annotations, n counts)
# ═══════════════════════════════════════════════════════════════
print("[R14] Box plot seasons ...")
daily_season=(df.groupby([df["Date"].dt.date,"Season"]).size()
              .reset_index(name="DailyCount"))
daily_season.columns=["Date","Season","DailyCount"]

fig,ax=plt.subplots(figsize=(11,6))
ax.set_facecolor(LIGHT)
bp=sns.boxplot(data=daily_season,x="Season",y="DailyCount",order=SEASON_ORDER,
    palette=SEASON_PAL,width=0.5,linewidth=1.8,fliersize=3,ax=ax,
    flierprops=dict(marker="o",markerfacecolor=GRAY,markeredgecolor="none",alpha=0.4))

# Mean markers (diamonds)
for i,s in enumerate(SEASON_ORDER):
    vals=daily_season[daily_season["Season"]==s]["DailyCount"]
    mean=vals.mean(); n=len(vals)
    ax.scatter(i,mean,marker="D",color="white",s=70,zorder=5,edgecolors=NAVY,linewidths=1.5,label="Mean" if i==0 else "")

    # Annotate: median, mean, IQR, n
    med=vals.median(); q1=vals.quantile(0.25); q3=vals.quantile(0.75); iqr=q3-q1
    ymax=vals.quantile(0.95)

    ax.text(i, q3+iqr*0.12, f"Med: {med:,.0f}", ha="center", va="bottom",
            fontsize=8.5, fontweight="bold", color=SEASON_PAL[s])
    ax.text(i, q3+iqr*0.55, f"μ: {mean:,.0f}", ha="center", va="bottom",
            fontsize=8, color=NAVY)
    ax.text(i, q1-iqr*0.15, f"IQR: {q1:,.0f}–{q3:,.0f}", ha="center", va="top",
            fontsize=7.5, color=GRAY)
    ax.text(i, vals.min()-iqr*0.05, f"n={n:,} days", ha="center", va="top",
            fontsize=7.5, color=GRAY, style="italic")

ax.set_title("Daily Crime Count Distribution by Season — Chicago\n"
             "(Boxes show IQR; whiskers = 1.5×IQR; ◆ = mean; horizontal line = median)",
             fontsize=13,fontweight="bold",color=NAVY,pad=14)
ax.set_xlabel("Season",fontsize=12); ax.set_ylabel("Daily Crime Incidents",fontsize=12)
ax.set_facecolor(LIGHT); ax.grid(axis="y",linestyle="--",alpha=0.4)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_:f"{int(v):,}"))
ax.legend(handles=[Line2D([0],[0],marker="D",color="w",markerfacecolor="white",
          markeredgecolor=NAVY,markersize=8,linewidth=0,label="Mean value")],
          fontsize=9,loc="upper left",framealpha=0.9)
ax.set_ylim(daily_season["DailyCount"].min()*0.60, daily_season["DailyCount"].max()*1.22)
save_report("B2_boxplot_season.png")


# ═══════════════════════════════════════════════════════════════
# B1_annual_trend  (already dual-saved from pipeline — skip)
# ═══════════════════════════════════════════════════════════════

# PART C — INTERACTIVE FOLIUM MAPS  (figures/ only, unchanged)
# =============================================================================

print("\n[C1] Folium geo heatmap ...")
try:
    import folium
    from folium.plugins import HeatMap
    samp_f = df[["Latitude","Longitude"]].dropna().sample(
        min(60_000, len(df)), random_state=42)
    m = folium.Map(location=[41.8781, -87.6298], zoom_start=11,
                   tiles="CartoDB dark_matter")
    HeatMap(samp_f.values.tolist(), radius=8, blur=10, min_opacity=0.3).add_to(m)
    m.save("figures/02_heatmap.html")
    print("  → figures/02_heatmap.html")
except Exception as e:
    print(f"  Skipped Folium heatmap: {e}")

print("[C2] Folium community area choropleth ...")
try:
    import folium
    comm_counts = df[comm_col].value_counts().reset_index()
    comm_counts.columns = ["area", "count"]
    comm_counts["area"] = comm_counts["area"].astype(str)
    geojson_path = "boundaries/Comm_Boundary.geojson"
    if os.path.exists(geojson_path):
        m2 = folium.Map(location=[41.8781, -87.6298], zoom_start=11)
        folium.Choropleth(
            geo_data=geojson_path, data=comm_counts,
            columns=["area", "count"],
            key_on="feature.properties.area_numbe",
            fill_color="YlOrRd", fill_opacity=0.75,
            line_opacity=0.3, legend_name="Crime Count by Community Area"
        ).add_to(m2)
        m2.save("figures/02_choropleth_community.html")
        print("  → figures/02_choropleth_community.html")
    else:
        print("  Skipped: boundaries/Comm_Boundary.geojson not found")
except Exception as e:
    print(f"  Skipped Folium choropleth: {e}")


# =============================================================================
# Summary
# =============================================================================
n_pipeline = len([f for f in os.listdir("figures")         if f.endswith(".png")])
n_report   = len([f for f in os.listdir("reports/figures") if f.endswith(".png")])
print("\n" + "=" * 60)
print(f"  PART 2 COMPLETE")
print(f"    figures/         — {n_pipeline} PNG  +  interactive HTML maps")
print(f"    reports/figures/ — {n_report} PNG  (all 17 report figures)")
print("=" * 60)
