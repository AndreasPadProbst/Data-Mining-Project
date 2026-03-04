"""
time_series_utils.py
====================
Helper functions for time-series analysis and forecasting of Chicago
crime counts.

Provides:
- Aggregation of crime data to monthly time-series.
- Stationarity testing (ADF test).
- ACF / PACF plotting helpers.
- SARIMA model fitting via auto_arima (pmdarima).
- Expanding-window cross-validation.
- Forecast plotting with confidence intervals.

Typical usage
-------------
>>> from src.time_series_utils import build_monthly_series, fit_sarima, plot_forecast
>>> monthly = build_monthly_series(df)
>>> model   = fit_sarima(monthly)
>>> plot_forecast(model, monthly, steps=12)
"""

import logging
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, mean_squared_error

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


# ── Time-series construction ──────────────────────────────────────────────────

def build_monthly_series(
    df: pd.DataFrame,
    date_col: str = "Date",
) -> pd.Series:
    """
    Aggregate the crime DataFrame to a monthly crime-count time-series.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned crime DataFrame containing a parsed *date_col* column.
    date_col : str
        Name of the datetime column. Default ``"Date"``.

    Returns
    -------
    pd.Series
        Monthly crime counts, indexed by month-end date (``freq='M'``).
        Series name: ``"CrimeCount"``.
    """
    s = (
        df.set_index(date_col)
        .resample("M")
        .size()
        .rename("CrimeCount")
    )
    log.info("Monthly series: %d periods (%s → %s).", len(s), s.index[0].date(), s.index[-1].date())
    return s


# ── Stationarity ──────────────────────────────────────────────────────────────

def adf_test(series: pd.Series, significance: float = 0.05) -> dict:
    """
    Run the Augmented Dickey–Fuller test for stationarity.

    H₀: The series has a unit root (non-stationary).
    H₁: The series is stationary.

    Parameters
    ----------
    series : pd.Series
        The time-series to test.
    significance : float
        Significance level for the hypothesis test. Default 0.05.

    Returns
    -------
    dict
        Keys: ``"adf_statistic"``, ``"p_value"``, ``"n_lags"``,
        ``"n_obs"``, ``"critical_values"``, ``"is_stationary"``.
    """
    result = adfuller(series.dropna(), autolag="AIC")
    is_stationary = result[1] < significance
    summary = {
        "adf_statistic":  round(result[0], 4),
        "p_value":        round(result[1], 4),
        "n_lags":         result[2],
        "n_obs":          result[3],
        "critical_values": result[4],
        "is_stationary":  is_stationary,
    }
    verdict = "STATIONARY" if is_stationary else "NON-STATIONARY"
    log.info("ADF test: %s (p=%.4f, stat=%.4f).", verdict, result[1], result[0])
    return summary


# ── ACF / PACF plotting ───────────────────────────────────────────────────────

def plot_acf_pacf(
    series: pd.Series,
    lags: int = 36,
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot ACF and PACF side-by-side for visual lag-order identification.

    Use ACF to identify the MA order q, and PACF to identify the AR order p
    prior to ARIMA model specification.

    Parameters
    ----------
    series : pd.Series
        The time-series to analyse.
    lags : int
        Number of lags to display. Default 36 (3 years of monthly data).
    figsize : tuple
        Figure size (width, height) in inches.
    save_path : str, optional
        If provided, save the figure to this path instead of displaying it.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    plot_acf(series.dropna(), lags=lags, ax=axes[0], alpha=0.05)
    axes[0].set_title("Autocorrelation Function (ACF)")
    axes[0].set_xlabel("Lag (months)")

    plot_pacf(series.dropna(), lags=lags, ax=axes[1], alpha=0.05, method="ywm")
    axes[1].set_title("Partial Autocorrelation Function (PACF)")
    axes[1].set_xlabel("Lag (months)")

    plt.suptitle("ACF / PACF Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info("ACF/PACF plot saved to '%s'.", save_path)
    else:
        plt.show()


# ── SARIMA modelling ──────────────────────────────────────────────────────────

def fit_sarima(
    series: pd.Series,
    m: int = 12,
    max_p: int = 3,
    max_q: int = 3,
    seasonal: bool = True,
    information_criterion: str = "aic",
) -> pm.arima.ARIMA:
    """
    Automatically select and fit a SARIMA model using stepwise AIC search.

    Uses ``pmdarima.auto_arima`` to search over ARIMA(p, d, q)(P, D, Q)[m]
    orders and select the model that minimises the Akaike Information
    Criterion (AIC).

    Parameters
    ----------
    series : pd.Series
        Monthly crime-count time-series.
    m : int
        Seasonal period (number of periods per year). Default 12 for monthly.
    max_p : int
        Maximum non-seasonal AR order to evaluate. Default 3.
    max_q : int
        Maximum non-seasonal MA order to evaluate. Default 3.
    seasonal : bool
        Whether to include a seasonal component. Default True.
    information_criterion : str
        Information criterion used for order selection: ``"aic"`` or ``"bic"``.
        Default ``"aic"``.

    Returns
    -------
    pmdarima.arima.ARIMA
        Fitted ARIMA/SARIMA model object. Call ``.predict(n_periods=N)``
        or ``.conf_int()`` on the returned object.
    """
    log.info("Fitting SARIMA model via auto_arima (m=%d, max_p=%d, max_q=%d) ...", m, max_p, max_q)

    model = pm.auto_arima(
        series,
        m=m,
        seasonal=seasonal,
        max_p=max_p,
        max_q=max_q,
        max_P=2,
        max_Q=2,
        d=None,          # auto-select differencing order via KPSS test
        D=None,          # auto-select seasonal differencing
        information_criterion=information_criterion,
        stepwise=True,   # stepwise search is much faster than exhaustive grid
        error_action="ignore",
        suppress_warnings=True,
        n_jobs=1,
    )

    log.info("Best order: ARIMA%s%s.", model.order, model.seasonal_order)
    log.info("AIC=%.2f, BIC=%.2f.", model.aic(), model.bic())
    return model


def plot_forecast(
    model: pm.arima.ARIMA,
    history: pd.Series,
    steps: int = 12,
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Plot historical data alongside a SARIMA out-of-sample forecast.

    Produces a publication-quality time-series chart showing:
    - The full historical crime count series.
    - The point forecast for the next *steps* months.
    - 95 % confidence intervals around the forecast.

    Parameters
    ----------
    model : pmdarima.arima.ARIMA
        Fitted SARIMA model returned by :func:`fit_sarima`.
    history : pd.Series
        Original monthly time-series (used for plotting context).
    steps : int
        Number of future months to forecast. Default 12.
    figsize : tuple
        Figure size in inches.
    save_path : str, optional
        If provided, save the figure to this path.

    Returns
    -------
    pd.DataFrame
        Forecast DataFrame with columns ``["Forecast", "Lower_CI", "Upper_CI"]``
        and a DatetimeIndex covering the forecast period.
    """
    # Generate forecasts and confidence intervals
    fc, conf_int = model.predict(n_periods=steps, return_conf_int=True)

    # Build a DatetimeIndex for the forecast horizon
    last_date = history.index[-1]
    fc_index = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=steps,
        freq="M",
    )

    fc_series     = pd.Series(fc,              index=fc_index, name="Forecast")
    lower_series  = pd.Series(conf_int[:, 0],  index=fc_index, name="Lower_CI")
    upper_series  = pd.Series(conf_int[:, 1],  index=fc_index, name="Upper_CI")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(history.index, history.values, color="#2C7BB6", linewidth=1.5,
            label="Historical Crime Count")
    ax.plot(fc_series.index, fc_series.values, color="#D7191C", linewidth=2,
            linestyle="--", label=f"{steps}-month SARIMA Forecast")
    ax.fill_between(fc_series.index, lower_series, upper_series,
                    color="#D7191C", alpha=0.15, label="95% Confidence Interval")

    ax.set_title(f"Monthly Crime Count Forecast (next {steps} months)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Crimes")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info("Forecast plot saved to '%s'.", save_path)
    else:
        plt.show()

    return pd.DataFrame({
        "Forecast":  fc_series,
        "Lower_CI":  lower_series,
        "Upper_CI":  upper_series,
    })


# ── Cross-validation ──────────────────────────────────────────────────────────

def sarima_cross_validate(
    series: pd.Series,
    n_splits: int = 5,
    forecast_horizon: int = 12,
    m: int = 12,
) -> pd.DataFrame:
    """
    Evaluate SARIMA performance using expanding-window time-series cross-validation.

    In each fold, the model is trained on all data up to the fold cutoff,
    then evaluated on the next *forecast_horizon* months. This mirrors
    the real-world scenario where the model is retrained periodically.

    Parameters
    ----------
    series : pd.Series
        Full monthly crime-count time-series.
    n_splits : int
        Number of cross-validation folds. Default 5.
    forecast_horizon : int
        Number of months to forecast at each fold. Default 12.
    m : int
        Seasonal period passed to :func:`fit_sarima`. Default 12.

    Returns
    -------
    pd.DataFrame
        One row per fold with columns:
        ``["fold", "train_end", "test_start", "mae", "rmse", "mape"]``.
    """
    n = len(series)
    min_train = n - n_splits * forecast_horizon  # minimum training window size

    results = []
    for fold in range(n_splits):
        train_end = min_train + fold * forecast_horizon
        train = series.iloc[:train_end]
        test  = series.iloc[train_end: train_end + forecast_horizon]

        if len(test) == 0:
            break

        try:
            model = fit_sarima(train, m=m)
            fc, _ = model.predict(n_periods=len(test), return_conf_int=True)
        except Exception as exc:
            log.warning("Fold %d failed: %s", fold + 1, exc)
            continue

        mae  = mean_absolute_error(test.values, fc)
        rmse = np.sqrt(mean_squared_error(test.values, fc))
        mape = np.mean(np.abs((test.values - fc) / test.values)) * 100

        results.append({
            "fold":        fold + 1,
            "train_end":   train.index[-1].date(),
            "test_start":  test.index[0].date(),
            "mae":         round(mae, 1),
            "rmse":        round(rmse, 1),
            "mape":        round(mape, 2),
        })
        log.info("Fold %d — MAE=%.1f, RMSE=%.1f, MAPE=%.2f%%.",
                 fold + 1, mae, rmse, mape)

    return pd.DataFrame(results)
