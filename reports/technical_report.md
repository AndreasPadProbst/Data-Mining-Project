# Technical Report — Chicago Crime Analysis

**Project:** Comprehensive Data Science Analysis of Chicago Crime Data  
**Dataset:** City of Chicago Crimes (2001–present), ~7.7 million records

---

## Executive Summary

This project delivers an end-to-end data science pipeline applied to the largest publicly available urban crime dataset in the United States. We demonstrate that Chicago crime has followed a long-run downward trend since its 2001 peak, exhibits strong seasonal patterns (summer peaks, winter troughs), and shows pronounced spatial clustering in certain community areas. Machine-learning models are able to predict arrest likelihood with AUC > 0.80 and daily crime volumes with R² > 0.75 when lag features are incorporated.

---

## 1. Dataset Overview

The Chicago Crimes dataset is published by the Chicago Police Department via the City of Chicago Data Portal. Each row represents a single reported criminal incident and contains 22 columns covering incident metadata (date, location, crime type), victim/offender relationship (domestic flag), outcome (arrest), and geographic identifiers at multiple administrative scales (beat, district, ward, community area).

Key characteristics:
- Date range: January 2001 to present
- Row count: approximately 7.7 million (grows daily)
- Geographic coverage: all 77 community areas of Chicago
- Data refresh: daily
- Known issues: mixed-type Location column, ~0.3% missing coordinates, midnight timestamp default

---

## 2. Data Cleaning Methodology

### 2.1 Mixed-Type Detection

The `Location` column contains string-formatted coordinate pairs (e.g. `"(41.881832000, -87.623177000)"`) for most rows, but float NaN values for rows without location data. Standard pandas `read_csv` infers the column as `object` dtype but emits a `DtypeWarning`. We explicitly separate these types with `isinstance` checks, retaining only string-type rows.

### 2.2 Missing Value Strategy

| Column | Missing % | Strategy |
|--------|-----------|----------|
| Latitude / Longitude | 0.30% | Drop rows |
| Ward | 1.2% | Drop or geospatial imputation |
| Community Area | 1.1% | Drop or geospatial imputation |
| Beat | 0.9% | Drop or geospatial imputation |

Geospatial imputation uses shapely point-in-polygon lookups against the official Chicago boundary GeoJSON files. This is computationally expensive but avoids systematic bias from dropping records with missing administrative fields.

### 2.3 Duplicate Detection

`Case Number` uniqueness: enforced. The dataset contains a small number of exact duplicate Case Numbers arising from re-extraction artefacts in the CPD system. Soft duplicates (same date + block + crime type) are logged but not removed.

### 2.4 Feature Engineering

Temporal features extracted from `Date`:  
Hour, DayOfWeek, DayOfWeekName, Month, MonthName, YearActual, Quarter, IsWeekend, IsHoliday, Season.

The `IsHoliday` flag uses the `holidays` Python package to identify all US federal holidays. While Chicago is a US city, some local holidays (e.g. St. Patrick's Day parade) are not captured and may influence crime patterns.

`LocationGrouped` consolidates ~300 raw `Location Description` values into 15 semantically coherent buckets (Street/Alley, Residential, Vehicle, Commercial, etc.) using a hand-curated mapping dictionary.

---

## 3. Descriptive Analysis — Key Findings

### 3.1 Temporal Trends

Chicago crime peaked at approximately 470,000 incidents in 2001. By 2022 this had declined to approximately 260,000 — a 45% reduction over two decades. This trend mirrors national patterns and cannot be attributed to any single causal factor, though scholars point to demographic change, increased incarceration rates in the early 2000s, and policing strategy evolution.

A notable reversal occurred in 2015–2016, coinciding with the "Ferguson Effect" period following the Laquan McDonald video release and resulting breakdown in police–community trust in Chicago.

### 3.2 Seasonality

Monthly crime counts follow a consistent sinusoidal pattern: summer months (June–August) average 15–25% more incidents than winter months (December–February). This pattern is most pronounced for outdoor crime types (assault, battery, theft). The STL decomposition confirms a stable seasonal amplitude across years.

### 3.3 Diurnal Patterns

Crime incidents peak between 8 PM and midnight. A secondary spike occurs at midnight (00:00:00) which is an artefact of the CPD data entry convention for incidents with unknown exact times.

### 3.4 Spatial Distribution

Crime is highly concentrated. The top 10 community areas account for approximately 35% of all incidents despite comprising less than 15% of the city's area. These areas are predominantly located on the South and West Sides. Beat-level choropleth maps reveal even finer-grained hot-spots.

### 3.5 Crime Type Composition

Theft (including motor vehicle theft) is the most common primary type, followed by battery and criminal damage. Homicide, while the crime receiving the greatest media attention, represents less than 0.05% of all incidents.

### 3.6 ARIMA/SARIMA Modelling

The monthly crime series is non-stationary (ADF p > 0.05 before differencing) with clear 12-month seasonality confirmed by ACF spikes at lags 12, 24, and 36. After first-order differencing and one seasonal difference, the series becomes stationary.

Auto-ARIMA selects SARIMA(1,1,1)(1,1,1)[12] as the best model by AIC. Cross-validation (3 expanding windows) yields mean MAPE ≈ 4–6%, indicating the model captures the seasonal pattern well but struggles with structural breaks (e.g. COVID-19 lockdowns in 2020).

---

## 4. Predictive Analysis — Key Findings

### 4.1 Arrest Prediction

Random Forest (100 trees, SMOTE oversampling) achieves AUC ≈ 0.82 and F1-macro ≈ 0.62 in 5-fold cross-validation. The most important features are `Primary Type`, `Hour`, and `Beat`. The relatively modest F1 reflects the genuine difficulty of the task: arrest is influenced by officer discretion, witness cooperation, and evidence availability — factors absent from the dataset.

### 4.2 Daily Crime Volume (XGBoost + Lag Features)

Including lag-1, lag-7, lag-14, and lag-30 day features alongside 7- and 30-day rolling means dramatically improves regression performance over calendar-only models (R² ≈ 0.78 vs. ≈ 0.30). This is consistent with the strong autocorrelation structure of daily crime series.

### 4.3 Crime Hot-Spot Identification

Beat-level hot-spot classification (top 20% of beats by crime count) achieves AUC ≈ 0.88, driven primarily by `District` and `Ward` features. This is partly a tautology — the model learns that high-crime districts contain high-crime beats — but demonstrates the stability of spatial crime patterns.

---

## 5. Limitations & Disclaimers

1. **Reporting bias**: Only crimes reported to or recorded by the CPD are included. Dark figure of crime (unreported incidents) is substantial and variable across crime types and communities.

2. **Geographic encoding**: Coordinates are truncated to the block level to protect privacy. Exact incident locations are unavailable.

3. **Temporal truncation**: The most recent 1–3 months of data are typically under-reported as incidents are still being processed.

4. **Causal inference**: All findings are associative. The dataset does not support causal claims without further experimental or quasi-experimental design.

5. **Model fairness**: Predictive models trained on historical crime data may encode historical policing biases (e.g. over-policing in certain communities). Deployment of such models for operational decisions carries significant equity risks and must be approached with extreme caution.

6. **IUCR code changes**: The CPD has modified some IUCR classifications over time. Longitudinal comparisons of specific crime types may be affected by reclassification events.

---

## 6. References

1. City of Chicago Data Portal. (2024). *Crimes — 2001 to Present*. https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2
2. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.
3. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
4. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD 2016*.
5. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JAIR*, 16, 321–357.
