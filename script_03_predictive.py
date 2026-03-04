#!/usr/bin/env python3
"""
script_03_predictive.py
=======================
Part 3: Predictive Analysis — 11 machine-learning tasks
Run: python script_03_predictive.py
Output: figures/03_*.png  +  printed evaluation tables
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                              mean_absolute_error, r2_score, mean_squared_error,
                              roc_auc_score, f1_score)
import xgboost as xgb
import statsmodels.api as sm

os.makedirs("figures", exist_ok=True)

print("=" * 60)
print("  PART 3 — Predictive Analysis & Machine Learning")
print("=" * 60)

print("\nLoading Crimes_Cleaned.csv ...")
df = pd.read_csv("data/Crimes_Cleaned.csv", parse_dates=["Date"], low_memory=False)

# ── Normalise column names ────────────────────────────────────────────────────
df.columns = (
    df.columns
      .str.strip()
      .str.replace(" ", "_", regex=False)
      .str.replace(r"[^\w]", "_", regex=True)
)
print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ── Resolve column names flexibly ─────────────────────────────────────────────
pt_col   = "Primary_Type"   if "Primary_Type"   in df.columns else "Primary Type"
loc_col  = "LocationGrouped" if "LocationGrouped" in df.columns else None
comm_col = "Community_Area" if "Community_Area"  in df.columns else "Community Area"

NAVY = "#1B2A4A"; BLUE = "#2C7BB6"; RED = "#D7191C"; LIGHT = "#F0F4F8"

def save(name):
    plt.savefig(f"figures/{name}", dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved: figures/{name}")

# ── Feature preparation helper ────────────────────────────────────────────────
def make_features(data, target, extra_cols=None):
    """
    Build a numeric feature matrix X and target series y.
    Automatically picks numeric columns and label-encodes categoricals.
    Drops the target and any date/id columns from features.

    Parameters
    ----------
    data       : pd.DataFrame — input data
    target     : str — column name of the target variable
    extra_cols : list of str — additional categorical columns to label-encode

    Returns
    -------
    X : pd.DataFrame — numeric feature matrix
    y : pd.Series    — target series
    """
    skip = {target, "Date", "ID", "Case_Number", "Updated_On",
            "Location", "Block", "Description", "IUCR", "FBI_Code",
            "X_Coordinate", "Y_Coordinate", "DayOfWeekName", "Season",
            "LocationGrouped", "Primary_Type"}
    num_cols = [c for c in data.select_dtypes(include=[np.number]).columns
                if c not in skip]
    X = data[num_cols].copy()

    # Label-encode any requested extra categorical columns
    for c in (extra_cols or []):
        if c in data.columns and c not in skip:
            le = LabelEncoder()
            X[c + "_enc"] = le.fit_transform(data[c].astype(str).fillna("UNKNOWN"))

    y = data[target].copy()
    return X.fillna(0), y

def eval_classifier(X, y, model, n_splits=3):
    """
    Evaluate a classifier with stratified k-fold CV.
    Returns dict with accuracy, f1_macro, roc_auc (where applicable).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_enc = LabelEncoder().fit_transform(y.astype(str))
    accs, f1s, aucs = [], [], []
    for tr, te in skf.split(X, y_enc):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y_enc[tr], y_enc[te]
        model.fit(Xtr, ytr)
        preds = model.predict(Xte)
        accs.append((preds == yte).mean())
        f1s.append(f1_score(yte, preds, average="macro", zero_division=0))
        if len(np.unique(y_enc)) == 2:
            proba = model.predict_proba(Xte)[:,1]
            aucs.append(roc_auc_score(yte, proba))
    result = {"Accuracy": f"{np.mean(accs):.3f} ± {np.std(accs):.3f}",
              "F1-Macro": f"{np.mean(f1s):.3f} ± {np.std(f1s):.3f}"}
    if aucs:
        result["ROC-AUC"] = f"{np.mean(aucs):.3f} ± {np.std(aucs):.3f}"
    return result

def eval_regressor(X, y, model, n_splits=5):
    """
    Evaluate a regression model with k-fold CV.
    Returns dict with MAE, RMSE, R².
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    maes, rmses, r2s = [], [], []
    for tr, te in kf.split(X):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]
        model.fit(Xtr, ytr)
        preds = model.predict(Xte)
        maes.append(mean_absolute_error(yte, preds))
        rmses.append(np.sqrt(mean_squared_error(yte, preds)))
        r2s.append(r2_score(yte, preds))
    return {"MAE":  f"{np.mean(maes):.1f} ± {np.std(maes):.1f}",
            "RMSE": f"{np.mean(rmses):.1f} ± {np.std(rmses):.1f}",
            "R²":   f"{np.mean(r2s):.3f} ± {np.std(r2s):.3f}"}

def plot_cm(y_true, y_pred, labels, title, fname):
    """Plot and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout(); save(fname)

def plot_fi(model, feat_names, title, fname, top_n=15):
    """Plot and save a feature importance bar chart."""
    fi = pd.Series(model.feature_importances_, index=feat_names)
    fi = fi.nlargest(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    fi.plot.barh(ax=ax, color=BLUE, edgecolor="white")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout(); save(fname)

# ── Sample for speed (use full data if RAM allows) ────────────────────────────
MAX_ROWS = 500_000
if len(df) > MAX_ROWS:
    print(f"\nSampling {MAX_ROWS:,} rows for ML tasks (full dataset has {len(df):,}) ...")
    df_ml = df.sample(MAX_ROWS, random_state=42).reset_index(drop=True)
else:
    df_ml = df.copy()

# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*55)
print("TASK 1 — Arrest Prediction (Binary Classification)")
print("─"*55)
df_t1 = df_ml[df_ml["Arrest"].notna()].copy()
df_t1["Arrest_bin"] = df_t1["Arrest"].astype(bool).astype(int)
X1, y1 = make_features(df_t1, "Arrest_bin",
                        extra_cols=[pt_col, loc_col] if loc_col else [pt_col])
print(f"Class balance:\n  No Arrest: {(y1==0).sum():,}  |  Arrest: {(y1==1).sum():,}")

rf1 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=8)
print("  Random Forest CV:", eval_classifier(X1, y1, rf1))

lr1 = LogisticRegression(max_iter=500, random_state=42, n_jobs=-1)
print("  Logistic Regression CV:", eval_classifier(X1, y1, lr1))

# Final model on 80/20 split
split = int(len(df_t1) * 0.8)
df_sorted = df_t1.sort_values("Date")
X_tr, X_te = X1.iloc[:split], X1.iloc[split:]
y_tr, y_te = y1.iloc[:split], y1.iloc[split:]
rf1.fit(X_tr, y_tr)
y_pred1 = rf1.predict(X_te)
print(classification_report(y_te, y_pred1, target_names=["No Arrest","Arrest"]))
plot_cm(y_te, y_pred1, ["No Arrest","Arrest"],
        "Task 1 — Arrest Prediction (Random Forest)", "03_task1_confusion.png")
plot_fi(rf1, list(X_tr.columns), "Task 1 — Feature Importances", "03_task1_importance.png")

# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*55)
print("TASK 2 — Crime Type Prediction (Multi-class)")
print("─"*55)
top10 = df_ml[pt_col].value_counts().head(10).index
df_t2 = df_ml[df_ml[pt_col].isin(top10)].copy()
X2, y2 = make_features(df_t2, pt_col)
xgb2 = xgb.XGBClassifier(n_estimators=100, random_state=42,
                           use_label_encoder=False, eval_metric="mlogloss",
                           verbosity=0, n_jobs=-1)
print("  XGBoost CV:", eval_classifier(X2, y2, xgb2, n_splits=3))

# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*55)
print("TASK 3 — Domestic Incident Prediction")
print("─"*55)
df_t3 = df_ml[df_ml["Domestic"].notna()].copy()
df_t3["Domestic_bin"] = df_t3["Domestic"].astype(bool).astype(int)
X3, y3 = make_features(df_t3, "Domestic_bin", extra_cols=[pt_col])
gb3 = GradientBoostingClassifier(n_estimators=80, random_state=42, max_depth=4)
print("  Gradient Boosting CV:", eval_classifier(X3, y3, gb3, n_splits=3))

# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*55)
print("TASK 4 — Daily Crime Count Regression")
print("─"*55)
daily = (df.groupby(df["Date"].dt.date)
           .agg(Count=("Arrest","count"),
                Month=("Month","first"),
                DayOfWeek=("DayOfWeek","first"),
                YearActual=("YearActual","first"),
                IsWeekend=("IsWeekend","first"))
           .reset_index())
X4 = daily[["Month","DayOfWeek","YearActual","IsWeekend"]].astype(float)
y4 = daily["Count"].astype(float)
xgb4 = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
lr4  = Ridge()
print("  XGBoost:", eval_regressor(X4, y4, xgb4))
print("  Ridge:  ", eval_regressor(X4, y4, lr4))

# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*55)
print("TASK 5 — Monthly Crime Count Forecast (SARIMA)")
print("─"*55)
monthly_ts = df.set_index("Date").resample("M").size().rename("Count")
monthly_recent = monthly_ts[monthly_ts.index.year >= 2010]
sarima = sm.tsa.SARIMAX(monthly_recent, order=(1,1,1),
                         seasonal_order=(1,1,1,12)).fit(disp=False)
fc = sarima.get_forecast(steps=12)
fc_m = fc.predicted_mean; ci = fc.conf_int()
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(monthly_recent.index[-60:], monthly_recent.values[-60:],
        label="Historical (last 5 yrs)", color=BLUE)
ax.plot(fc_m.index, fc_m.values, label="12-Month Forecast",
        color=RED, linestyle="--", linewidth=2)
ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1],
                alpha=0.2, color=RED, label="95% CI")
ax.set_title("Task 5 — SARIMA Monthly Crime Count Forecast",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Date"); ax.set_ylabel("Crimes per Month")
ax.legend(); ax.grid(linestyle="--", alpha=0.4)
plt.tight_layout(); save("03_task5_monthly_forecast.png")

# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*55)
print("TASK 6 — Crime Severity Regression")
print("─"*55)
fbi_sev = {"01A":1,"01B":2,"02":3,"03":4,"04A":5,"04B":6,"05":7,
           "06":8,"07":9,"08A":10,"08B":11,"09":12,"10":13,
           "11":14,"12":15,"13":16,"14":17,"15":18,"16":19,
           "17":20,"18":21,"19":22,"20":23,"24":24,"26":25}
fbi_col = "FBI_Code" if "FBI_Code" in df_ml.columns else "FBI Code"
df_t6 = df_ml.copy()
df_t6["Severity"] = df_t6[fbi_col].astype(str).map(fbi_sev).fillna(10)
X6, y6 = make_features(df_t6, "Severity")
ridge6 = Ridge()
print("  Ridge:", eval_regressor(X6, y6, ridge6))

# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*55)
print("TASK 7 — Hot-Spot Beat Classification")
print("─"*55)
beat_counts = df_ml["Beat"].value_counts()
threshold = beat_counts.quantile(0.80)
df_t7 = df_ml.copy()
df_t7["IsHotSpot"] = (df_t7["Beat"].map(beat_counts) >= threshold).astype(int)
X7, y7 = make_features(df_t7, "IsHotSpot", extra_cols=[pt_col])
rf7 = RandomForestClassifier(n_estimators=80, random_state=42, n_jobs=-1, max_depth=6)
print("  Random Forest CV:", eval_classifier(X7, y7, rf7, n_splits=3))

# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*55)
print("TASK 8 — Night-Time Spike Prediction")
print("─"*55)
df_t8 = df_ml.copy()
df_t8["IsNightSpike"] = df_t8["Hour"].isin([22,23,0,1,2,3]).astype(int)
X8, y8 = make_features(df_t8, "IsNightSpike")
lr8 = LogisticRegression(max_iter=500, random_state=42, n_jobs=-1)
print("  Logistic Regression CV:", eval_classifier(X8, y8, lr8, n_splits=3))

# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*55)
print("TASK 9 — Weekend vs Weekday Crime Volume")
print("─"*55)
daily2 = (df.groupby([df["Date"].dt.date,"IsWeekend"])
            .size().reset_index(name="Count"))
wkday = daily2[daily2["IsWeekend"]==0]["Count"].mean()
wkend = daily2[daily2["IsWeekend"]==1]["Count"].mean()
print(f"  Weekday avg: {wkday:.0f} crimes/day  |  Weekend avg: {wkend:.0f} crimes/day")
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(["Weekday","Weekend"], [wkday, wkend],
       color=[BLUE, RED], edgecolor="white", width=0.5)
ax.set_title("Average Daily Crime Count: Weekday vs. Weekend",
             fontsize=12, fontweight="bold")
ax.set_ylabel("Average Crimes per Day")
ax.grid(axis="y", linestyle="--", alpha=0.4); ax.set_facecolor(LIGHT)
plt.tight_layout(); save("03_task9_weekend.png")

# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*55)
print("TASK 10 — Quarterly Crime Count Forecast")
print("─"*55)
quarterly = df.set_index("Date").resample("Q").size().rename("Count")
qtr_recent = quarterly[quarterly.index.year >= 2010]
qmodel = sm.tsa.SARIMAX(qtr_recent, order=(1,1,1),
                         seasonal_order=(1,1,1,4)).fit(disp=False)
fc_q = qmodel.get_forecast(steps=4)
fc_qm = fc_q.predicted_mean; ci_q = fc_q.conf_int()
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(qtr_recent.index, qtr_recent.values, label="Historical", color=BLUE)
ax.plot(fc_qm.index, fc_qm.values, label="4-Quarter Forecast",
        color=RED, linestyle="--", linewidth=2)
ax.fill_between(ci_q.index, ci_q.iloc[:,0], ci_q.iloc[:,1],
                alpha=0.2, color=RED)
ax.set_title("Task 10 — Quarterly Crime Count Forecast",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Date"); ax.set_ylabel("Crimes per Quarter"); ax.legend()
ax.grid(linestyle="--", alpha=0.4)
plt.tight_layout(); save("03_task10_quarterly.png")

# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*55)
print("TASK 11 — XGBoost Daily Forecast with Lag Features")
print("─"*55)
daily_ts = (df.groupby(df["Date"].dt.date).size()
              .rename("Count").reset_index())
daily_ts.columns = ["Date","Count"]
daily_ts["Date"] = pd.to_datetime(daily_ts["Date"])
daily_ts = daily_ts.sort_values("Date").reset_index(drop=True)
for lag in [1, 7, 14, 30]:
    daily_ts[f"lag_{lag}"] = daily_ts["Count"].shift(lag)
daily_ts["roll_7"]  = daily_ts["Count"].shift(1).rolling(7).mean()
daily_ts["roll_30"] = daily_ts["Count"].shift(1).rolling(30).mean()
daily_ts["DayOfWeek"] = daily_ts["Date"].dt.dayofweek
daily_ts["Month"]     = daily_ts["Date"].dt.month
daily_ts["Year"]      = daily_ts["Date"].dt.year
daily_ts = daily_ts.dropna().reset_index(drop=True)
feat_lag = ["lag_1","lag_7","lag_14","lag_30","roll_7","roll_30","DayOfWeek","Month","Year"]
X11 = daily_ts[feat_lag]; y11 = daily_ts["Count"]
split11 = int(len(daily_ts) * 0.8)
Xtr11, Xte11 = X11.iloc[:split11], X11.iloc[split11:]
ytr11, yte11 = y11.iloc[:split11], y11.iloc[split11:]
xgb11 = xgb.XGBRegressor(n_estimators=200, max_depth=4,
                           learning_rate=0.05, random_state=42, verbosity=0)
xgb11.fit(Xtr11, ytr11)
preds11 = xgb11.predict(Xte11)
mae11 = mean_absolute_error(yte11, preds11)
r2_11 = r2_score(yte11, preds11)
print(f"  XGBoost daily forecast — MAE: {mae11:.1f} crimes/day  |  R²: {r2_11:.3f}")
n_plot = min(120, len(yte11))
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(daily_ts["Date"].iloc[split11:split11+n_plot],
        yte11.values[:n_plot], label="Actual", color=BLUE)
ax.plot(daily_ts["Date"].iloc[split11:split11+n_plot],
        preds11[:n_plot], label=f"XGBoost (MAE={mae11:.0f}, R²={r2_11:.2f})",
        color=RED, linestyle="--", alpha=0.85)
ax.set_title("Task 11 — XGBoost Daily Crime Forecast vs. Actual",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Date"); ax.set_ylabel("Daily Crime Count"); ax.legend()
ax.grid(linestyle="--", alpha=0.4)
plt.tight_layout(); save("03_task11_xgb_forecast.png")

print("\n" + "=" * 60)
total_png = len([f for f in os.listdir("figures") if f.startswith("03_")])
print(f"  PART 3 COMPLETE — {total_png} prediction charts saved")
print("=" * 60)
