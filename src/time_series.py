"""
time_series.py — Time-series analysis and forecasting for Chicago crime data.

Implements:
- Stationarity testing (Augmented Dickey-Fuller)
- ACF / PACF plotting
- Moving-average smoothing
- Exponential smoothing (single, double, triple / Holt-Winters)
- Seasonal decomposition (additive / multiplicative)
- SARIMA model fitting and forecasting
- Holt-Winters with anomaly detection (Brutlag method)

All plotting functions save figures to an output directory.

Classes
-------
HoltWinters
    Triple exponential smoothing with Brutlag anomaly detection.

Functions
---------
test_stationarity(series, window)
    ADF test + rolling mean/std plot.

plot_acf_pacf(series, lags, output_dir)
    ACF and PACF correlograms.

plot_moving_average(series, window, output_dir, plot_anomalies)
    Rolling mean trend + optional anomaly highlights.

plot_exponential_smoothing(series, alphas, output_dir)
    Single exponential smoothing for various alpha values.

fit_sarima(series, order, seasonal_order)
    Fit a SARIMA model; return fitted model object.

plot_sarima_forecast(model, series, n_periods, output_dir)
    Plot SARIMA in-sample fit + future forecast with confidence intervals.

plot_decomposition(series, period, output_dir)
    STL seasonal decomposition plot.
"""

import logging
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# ─────────────────────────────────────────────────────────────────────────────
# Holt-Winters with Brutlag anomaly detection
# ─────────────────────────────────────────────────────────────────────────────

class HoltWinters:
    """
    Holt-Winters triple exponential smoothing with Brutlag anomaly detection.

    Attributes
    ----------
    series : array-like
        The input time series.
    slen : int
        Season length (e.g., 12 for monthly data with yearly seasonality).
    alpha : float
        Level smoothing factor [0, 1].
    beta : float
        Trend smoothing factor [0, 1].
    gamma : float
        Seasonal smoothing factor [0, 1].
    n_preds : int
        Number of future periods to forecast.
    scaling_factor : float
        Width of the Brutlag confidence band (default 1.96 ≈ 95% CI).
    """

    def __init__(
        self,
        series: np.ndarray,
        slen: int,
        alpha: float,
        beta: float,
        gamma: float,
        n_preds: int,
        scaling_factor: float = 1.96,
    ) -> None:
        self.series = series
        self.slen = slen
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n_preds = n_preds
        self.scaling_factor = scaling_factor

    def initial_trend(self) -> float:
        """Compute the initial trend estimate from the first two seasons."""
        slen = self.slen
        return (
            sum((self.series[i + slen] - self.series[i]) / slen for i in range(slen)) / slen
        )

    def initial_seasonal_components(self) -> dict:
        """Compute initial seasonal components."""
        slen = self.slen
        n_seasons = int(len(self.series) / slen)
        season_averages = [
            np.mean(self.series[slen * j: slen * j + slen]) for j in range(n_seasons)
        ]
        return {
            i: np.mean([
                self.series[slen * j + i] / season_averages[j]
                for j in range(n_seasons)
                if season_averages[j] != 0
            ])
            for i in range(slen)
        }

    def triple_exponential_smoothing(self) -> None:
        """Run triple exponential smoothing and store results."""
        self.result = []
        self.Smooth = []
        self.Season = []
        self.Trend = []
        self.PredictedDeviation = []
        self.UpperBond = []
        self.LowerBond = []

        seasonals = self.initial_seasonal_components()
        smooth = self.series[0]
        trend  = self.initial_trend()

        for i in range(len(self.series) + self.n_preds):
            if i == 0:
                self.result.append(self.series[0])
                self.Smooth.append(smooth)
                self.Trend.append(trend)
                self.Season.append(seasonals[i % self.slen])
                self.PredictedDeviation.append(0)
            elif i >= len(self.series):
                # Forecasting future points
                m = i - len(self.series) + 1
                pred = (smooth + m * trend) * seasonals[i % self.slen]
                self.result.append(pred)
                self.PredictedDeviation.append(self.PredictedDeviation[-1] * 1.01)
            else:
                val        = self.series[i]
                last_smooth, smooth = smooth, (
                    self.alpha * (val / seasonals[i % self.slen])
                    + (1 - self.alpha) * (smooth + trend)
                )
                trend = self.beta * (smooth - last_smooth) + (1 - self.beta) * trend
                seasonals[i % self.slen] = (
                    self.gamma * (val / smooth)
                    + (1 - self.gamma) * seasonals[i % self.slen]
                )
                self.result.append(smooth * seasonals[i % self.slen])
                # Brutlag deviation
                self.PredictedDeviation.append(
                    self.gamma * abs(val - self.result[-1])
                    + (1 - self.gamma) * self.PredictedDeviation[-1]
                )

            self.UpperBond.append(
                self.result[-1] + self.scaling_factor * self.PredictedDeviation[-1]
            )
            self.LowerBond.append(
                self.result[-1] - self.scaling_factor * self.PredictedDeviation[-1]
            )
            self.Smooth.append(smooth)
            self.Trend.append(trend)
            self.Season.append(seasonals[i % self.slen])


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / name, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s", output_dir / name)


# ─────────────────────────────────────────────────────────────────────────────
# Public functions
# ─────────────────────────────────────────────────────────────────────────────

def test_stationarity(
    series: pd.Series,
    window: int = 12,
    output_dir: str | Path | None = None,
) -> dict:
    """
    Augmented Dickey-Fuller stationarity test + rolling stats visualisation.

    A series is stationary if its statistical properties (mean, variance)
    do not change over time.  Non-stationary series must be differenced
    before fitting ARIMA-family models.

    Parameters
    ----------
    series : pd.Series
        Monthly or periodic time series.
    window : int, default 12
        Rolling window for the mean / std overlay.
    output_dir : str or Path, optional
        If provided, the plot is saved here.

    Returns
    -------
    dict
        Keys: ``adf_statistic``, ``p_value``, ``is_stationary``.
    """
    from statsmodels.tsa.stattools import adfuller

    result = adfuller(series.dropna(), autolag="AIC")
    output = {
        "adf_statistic": result[0],
        "p_value": result[1],
        "is_stationary": result[1] <= 0.05,
    }
    logger.info(
        "ADF test: statistic=%.4f, p=%.4f — %s",
        result[0], result[1],
        "STATIONARY" if output["is_stationary"] else "NON-STATIONARY",
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        rolling_mean = series.rolling(window=window).mean()
        rolling_std  = series.rolling(window=window).std()

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(series.index, series, label="Original", alpha=0.8, color="#1a6ea8")
        ax.plot(rolling_mean.index, rolling_mean, label=f"Rolling Mean ({window})", color="orange", linewidth=2)
        ax.plot(rolling_std.index, rolling_std, label=f"Rolling Std ({window})", color="red", linestyle="--", linewidth=1.5)
        ax.set_title(
            f"Stationarity Check — ADF p-value: {result[1]:.4f} "
            f"({'Stationary' if output['is_stationary'] else 'Non-stationary'})",
            fontsize=13, fontweight="bold",
        )
        ax.legend()
        _save(fig, output_dir, "ts_01_stationarity.png")

    return output


def plot_acf_pacf(
    series: pd.Series,
    lags: int = 48,
    output_dir: str | Path = "outputs/figures",
) -> None:
    """
    Plot ACF and PACF correlograms side by side.

    The ACF shows how correlated a series is with its own lagged versions.
    The PACF shows the partial correlation at each lag after removing the
    effect of shorter lags.  Together they guide ARIMA order selection.

    Parameters
    ----------
    series : pd.Series
    lags : int, default 48
    output_dir : str or Path
    """
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    output_dir = Path(output_dir)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    plot_acf(series.dropna(), lags=lags, ax=ax1, alpha=0.05)
    ax1.set_title("Autocorrelation Function (ACF)", fontsize=12, fontweight="bold")
    plot_pacf(series.dropna(), lags=lags, ax=ax2, alpha=0.05)
    ax2.set_title("Partial Autocorrelation Function (PACF)", fontsize=12, fontweight="bold")
    fig.suptitle("ACF / PACF Analysis for ARIMA Order Selection", fontsize=13, y=1.02)
    _save(fig, output_dir, "ts_02_acf_pacf.png")


def plot_moving_average(
    series: pd.Series,
    window: int,
    output_dir: str | Path,
    plot_anomalies: bool = True,
) -> plt.Figure:
    """
    Plot the original series overlaid with its rolling mean.

    Anomalies are highlighted as points more than 1.96 × rolling-std
    from the rolling mean.

    Parameters
    ----------
    series : pd.Series
    window : int  — rolling window size.
    output_dir : str or Path
    plot_anomalies : bool, default True

    Returns
    -------
    plt.Figure
    """
    from sklearn.metrics import mean_absolute_error

    output_dir = Path(output_dir)
    rolling_mean = series.rolling(window=window).mean()
    mae = mean_absolute_error(series[window:], rolling_mean[window:])
    deviation = series.rolling(window=window).std()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(series.index, series, label="Actual", alpha=0.5, color="#1a6ea8")
    ax.plot(rolling_mean.index, rolling_mean, label=f"Rolling Mean (w={window})", color="orange", linewidth=2)
    ax.fill_between(
        rolling_mean.index,
        rolling_mean - 1.96 * deviation,
        rolling_mean + 1.96 * deviation,
        alpha=0.15, color="orange", label="95% CI",
    )

    if plot_anomalies:
        anomaly_mask = (series > rolling_mean + 1.96 * deviation) | (series < rolling_mean - 1.96 * deviation)
        ax.scatter(series[anomaly_mask].index, series[anomaly_mask].values,
                   color="red", zorder=5, s=50, label="Anomaly")

    ax.set_title(f"Moving Average (window={window}) — MAE: {mae:,.0f}", fontsize=13, fontweight="bold")
    ax.set_ylabel("Crime Count")
    ax.legend()
    _save(fig, output_dir, f"ts_03_moving_avg_{window}.png")
    return fig


def plot_exponential_smoothing(
    series: pd.Series,
    alphas: list[float],
    output_dir: str | Path,
) -> plt.Figure:
    """
    Plot single exponential smoothing for a list of alpha values.

    Parameters
    ----------
    series : pd.Series
    alphas : list[float]
        Smoothing factors to compare (e.g., [0.9, 0.5, 0.1]).
    output_dir : str or Path

    Returns
    -------
    plt.Figure
    """
    output_dir = Path(output_dir)

    def _smooth(s, a):
        result = [s.iloc[0]]
        for v in s.iloc[1:]:
            result.append(a * v + (1 - a) * result[-1])
        return result

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(series.index, series.values, label="Actual", alpha=0.6, color="gray", linewidth=1.5)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(alphas)))
    for i, a in enumerate(alphas):
        smoothed = _smooth(series, a)
        ax.plot(series.index, smoothed, label=f"α = {a}", color=colors[i], linewidth=1.8)

    ax.set_title("Single Exponential Smoothing — Multiple α Values", fontsize=13, fontweight="bold")
    ax.set_ylabel("Crime Count")
    ax.legend(loc="upper right")
    _save(fig, output_dir, "ts_04_exp_smoothing.png")
    return fig


def plot_holt_winters(
    hw_model: HoltWinters,
    series: pd.Series,
    output_dir: str | Path,
    plot_anomalies: bool = True,
) -> plt.Figure:
    """
    Visualise a fitted HoltWinters model with forecast and anomaly detection.

    Parameters
    ----------
    hw_model : HoltWinters
        A fitted HoltWinters instance (after calling ``triple_exponential_smoothing()``).
    series : pd.Series
        The original observed series.
    output_dir : str or Path
    plot_anomalies : bool, default True

    Returns
    -------
    plt.Figure
    """
    output_dir = Path(output_dir)
    from sklearn.metrics import mean_absolute_percentage_error

    n = len(series)
    mape = mean_absolute_percentage_error(series.values, hw_model.result[:n]) * 100

    fig, ax = plt.subplots(figsize=(18, 7))
    ax.plot(hw_model.result, label="Model (fitted + forecast)", color="#1a6ea8", linewidth=2)
    ax.plot(series.values, label="Actual", color="black", linewidth=1.5, alpha=0.8)
    ax.fill_between(
        range(len(hw_model.UpperBond)),
        hw_model.LowerBond,
        hw_model.UpperBond,
        alpha=0.2,
        color="#1a6ea8",
        label="95% CI (Brutlag)",
    )

    if plot_anomalies:
        anomalies = np.where(
            (series.values < np.array(hw_model.LowerBond[:n])) |
            (series.values > np.array(hw_model.UpperBond[:n]))
        )[0]
        ax.scatter(anomalies, series.values[anomalies], color="red", s=60, zorder=6, label="Anomalies")

    ax.axvline(x=n, color="gray", linestyle="--", linewidth=1.5, label="Forecast Start")
    ax.set_title(f"Holt-Winters Forecast — MAPE: {mape:.2f}%", fontsize=14, fontweight="bold")
    ax.set_ylabel("Monthly Crime Count")
    ax.set_xlabel("Time (months)")
    ax.legend()
    _save(fig, output_dir, "ts_05_holt_winters.png")
    return fig


def fit_sarima(
    series: pd.Series,
    order: tuple = (1, 1, 1),
    seasonal_order: tuple = (1, 1, 1, 12),
) -> "statsmodels.tsa.statespace.sarimax.SARIMAXResultsWrapper":
    """
    Fit a SARIMA model to a monthly time series.

    SARIMA(p,d,q)(P,D,Q,s) where s=12 captures yearly seasonality.

    Parameters
    ----------
    series : pd.Series
        Monthly time series (indexed by DatetimeIndex or PeriodIndex).
    order : tuple, default (1, 1, 1)
        Non-seasonal ARIMA order (p, d, q).
    seasonal_order : tuple, default (1, 1, 1, 12)
        Seasonal order (P, D, Q, s).

    Returns
    -------
    Fitted statsmodels SARIMAX results object.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    logger.info("Fitting SARIMA%s × %s …", order, seasonal_order)
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    result = model.fit(disp=False)
    logger.info("  AIC: %.2f | BIC: %.2f", result.aic, result.bic)
    return result


def plot_sarima_forecast(
    fitted_model,
    series: pd.Series,
    n_periods: int,
    output_dir: str | Path,
) -> plt.Figure:
    """
    Plot SARIMA in-sample fit plus n-period ahead forecast with CI.

    Parameters
    ----------
    fitted_model : SARIMAX result object
    series : pd.Series  — original observed series.
    n_periods : int  — number of future periods to forecast.
    output_dir : str or Path

    Returns
    -------
    plt.Figure
    """
    output_dir = Path(output_dir)
    pred = fitted_model.get_prediction(start=series.index[0], end=series.index[-1])
    forecast = fitted_model.get_forecast(steps=n_periods)

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(series, label="Actual", color="black", linewidth=1.5)
    ax.plot(pred.predicted_mean, label="In-sample fit", color="#1a6ea8", linewidth=1.5, alpha=0.8)

    forecast_index = pd.date_range(
        start=series.index[-1], periods=n_periods + 1, freq=pd.infer_freq(series.index)
    )[1:]
    ax.plot(forecast_index, forecast.predicted_mean.values, label="Forecast", color="orange", linewidth=2)

    ci = forecast.conf_int()
    ax.fill_between(
        forecast_index,
        ci.iloc[:, 0].values,
        ci.iloc[:, 1].values,
        alpha=0.25, color="orange", label="95% CI",
    )

    ax.axvline(x=series.index[-1], color="gray", linestyle="--", linewidth=1.2)
    ax.set_title(f"SARIMA Forecast — {n_periods}-Period Ahead", fontsize=13, fontweight="bold")
    ax.set_ylabel("Monthly Crime Count")
    ax.legend()
    _save(fig, output_dir, "ts_06_sarima_forecast.png")
    return fig


def plot_decomposition(
    series: pd.Series,
    period: int = 12,
    output_dir: str | Path = "outputs/figures",
    model: str = "additive",
) -> plt.Figure:
    """
    Seasonal decomposition using STL.

    Decomposes the series into trend, seasonal, and residual components.

    Parameters
    ----------
    series : pd.Series
    period : int, default 12
        Seasonality period.
    output_dir : str or Path
    model : {'additive', 'multiplicative'}, default 'additive'

    Returns
    -------
    plt.Figure
    """
    from statsmodels.tsa.seasonal import seasonal_decompose

    output_dir = Path(output_dir)
    decomp = seasonal_decompose(series.dropna(), model=model, period=period, extrapolate_trend="freq")

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    for ax, component, name, color in zip(
        axes,
        [series, decomp.trend, decomp.seasonal, decomp.resid],
        ["Observed", "Trend", "Seasonal", "Residual"],
        ["#1a6ea8", "orange", "#2ca02c", "#d62728"],
    ):
        ax.plot(series.index, component, color=color)
        ax.set_ylabel(name, fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    axes[0].set_title(f"Seasonal Decomposition ({model.title()}) — Period={period}",
                      fontsize=13, fontweight="bold")
    _save(fig, output_dir, "ts_07_decomposition.png")
    return fig
