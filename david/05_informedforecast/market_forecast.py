
"""
market_forecast.py  (FREE-tier friendly)

Alpha Vantage helpers + lightweight forecasting
- Primary: TIME_SERIES_DAILY (free)
- Optional fallback: WEEKLY or MONTHLY (free)
- Avoid TIME_SERIES_DAILY_ADJUSTED by default (often premium in 2025)
- Statsmodels Exponential Smoothing forecaster

Install:
  pip install requests pandas numpy statsmodels plotly
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

import requests
import pandas as pd
import numpy as np

# Forecasting
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except Exception:
    ExponentialSmoothing = None  # type: ignore


def _normalize_df_from_series(ts: dict, granularity: str) -> pd.DataFrame:
    rows = []
    # Keys differ: Daily -> "Time Series (Daily)", Weekly -> "Weekly Time Series", Monthly -> "Monthly Time Series"
    for d, v in ts.items():
        # Field names differ across endpoints; guard with get
        rows.append({
            "date": pd.to_datetime(d),
            "open": float(v.get("1. open") or v.get("1. Open")),
            "high": float(v.get("2. high") or v.get("2. High")),
            "low": float(v.get("3. low") or v.get("3. Low")),
            "close": float(v.get("4. close") or v.get("4. Close")),
            "volume": float(v.get("5. volume") or v.get("5. Volume") or 0.0),
        })
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df["adjusted_close"] = df["close"]  # for non-adjusted endpoints
    df["granularity"] = granularity
    return df


def fetch_alpha_vantage_free(symbol: str, api_key: str, granularity: str = "daily", outputsize: str = "compact") -> pd.DataFrame:
    """
    FREE-tier friendly fetcher. Uses non-adjusted endpoints.
      granularity: "daily" | "weekly" | "monthly"
      outputsize: "compact" | "full"  (daily only)
    Returns standardized columns: [date, open, high, low, close, adjusted_close, volume, granularity]
    """
    url = "https://www.alphavantage.co/query"
    if granularity == "daily":
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "apikey": api_key,
            "outputsize": outputsize,
        }
        key_name = "Time Series (Daily)"
    elif granularity == "weekly":
        params = {
            "function": "TIME_SERIES_WEEKLY",
            "symbol": symbol,
            "apikey": api_key,
        }
        key_name = "Weekly Time Series"
    elif granularity == "monthly":
        params = {
            "function": "TIME_SERIES_MONTHLY",
            "symbol": symbol,
            "apikey": api_key,
        }
        key_name = "Monthly Time Series"
    else:
        raise ValueError("granularity must be 'daily', 'weekly', or 'monthly'")

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    if "Error Message" in data:
        raise ValueError(data["Error Message"])
    if "Note" in data:
        # Rate limited; surface the message
        raise RuntimeError(data["Note"])
    if "Information" in data:
        # Premium endpoint or other info message
        raise RuntimeError(data["Information"])
    if key_name not in data:
        raise RuntimeError(f"Alpha Vantage response missing '{key_name}'. Raw keys: {list(data.keys())[:5]}")

    ts = data[key_name]
    df = _normalize_df_from_series(ts, granularity=granularity)
    return df


HORIZON_PAT = re.compile(r"(\d+)\s*(day|days|week|weeks|month|months|quarter|quarters|year|years)", re.I)

def parse_horizon(prompt: str, default_days: int = 20) -> int:
    """
    Parse natural language like 'forecast next 6 weeks' -> 30 business days.
    """
    if not prompt:
        return default_days
    m = HORIZON_PAT.search(prompt)
    if not m:
        return default_days
    n = int(m.group(1))
    unit = m.group(2).lower()
    if "day" in unit:
        return n
    if "week" in unit:
        return n * 5
    if "month" in unit:
        return n * 21
    if "quarter" in unit:
        return n * 63
    if "year" in unit:
        return n * 252
    return default_days


def exp_smoothing_forecast(df_prices: pd.DataFrame, value_col: str = "adjusted_close", periods: int = 20,
                           seasonal: Optional[str] = None, seasonal_periods: Optional[int] = None) -> pd.DataFrame:
    """
    Fit Exponential Smoothing with optional trend and seasonal components.
    Returns DataFrame with columns: date, yhat, yhat_lower, yhat_upper
    """
    if ExponentialSmoothing is None:
        raise ImportError("statsmodels is required. pip install statsmodels")

    s = df_prices.set_index("date")[value_col].asfreq("B")  # business days
    s = s.interpolate(limit_direction="both")  # fill any gaps

    model = ExponentialSmoothing(
        s,
        trend="add",
        seasonal=seasonal,                 # None by default
        seasonal_periods=seasonal_periods  # e.g., 5 for weekly pattern on business days
    )
    fit = model.fit(optimized=True, use_brute=True)

    resid = fit.resid.dropna()
    sigma = float(resid.std(ddof=1)) if len(resid) > 1 else 0.0
    z = 1.96

    future_idx = pd.bdate_range(start=s.index[-1] + pd.offsets.BDay(), periods=periods)
    yhat = fit.forecast(periods)
    yhat.index = future_idx

    out = pd.DataFrame({
        "date": yhat.index,
        "yhat": yhat.values,
        "yhat_lower": yhat.values - z * sigma,
        "yhat_upper": yhat.values + z * sigma,
    })
    return out




def garch_price_forecast(
    df_prices: pd.DataFrame,
    value_col: str = "adjusted_close",
    periods: int = 20,
    vol: str = "EGARCH",          # 'GARCH', 'EGARCH', 'GJR-GARCH', etc.
    p: int = 1,
    o: int = 0,                   # EGARCH/GJR needs o>=1 for asymmetry; keep 0 for plain GARCH
    q: int = 1,
    mean: str = "Constant",       # 'Zero', 'Constant', 'ARX'
    dist: str = "normal",         # 'normal', 't', 'skewt'
    alpha: float = 0.05,          # 95% interval by default
    scale_returns: float = 1.0,   # set to 100.0 if you prefer returns in %
    reindex_to_bdays: bool = True # create business-day future index like your ES function
) -> pd.DataFrame:
    """
    Fit an ARCH-family model on log returns and map the step-ahead return forecasts
    to price forecasts (median yhat) and a two-sided (1-alpha) interval under
    the model's conditional distribution.

    Returns: DataFrame[date, yhat, yhat_lower, yhat_upper] (prices)
    """

    if value_col not in df_prices.columns:
        raise ValueError(f"{value_col} not found in df_prices")
    if len(df_prices) < 50:
        raise ValueError("Need at least ~50 observations to fit a GARCH model reliably.")

    # 1) Prep series
    px = df_prices[["date", value_col]].dropna().copy()
    px["date"] = pd.to_datetime(px["date"])
    px = px.sort_values("date").reset_index(drop=True)
    last_date = px["date"].iloc[-1]
    last_price = float(px[value_col].iloc[-1])

    # log returns (natural); scale if you like (%)
    r = np.log(px[value_col]).diff().dropna()
    r = r * scale_returns
    r.index = px["date"].iloc[1:]  # align dates

    # 2) Fit ARCH-family model to returns
    am = arch_model(r, mean=mean, vol=vol, p=p, o=o, q=q, dist=dist, rescale=False)
    res = am.fit(disp="off")

    # 3) Step-ahead forecasts for mean and variance of *returns*
    #    (These are conditional on information at t.)
    fcast = res.forecast(horizon=periods, reindex=False)

    # Mean forecast of returns (shape: 1 x periods)
    # Some mean specs (e.g., 'Zero') will give 0s here, which is fine.
    mu_step = np.asarray(fcast.mean.values[-1, :], dtype=float)  # step-wise
    # Variance forecast of returns
    var_step = np.asarray(fcast.variance.values[-1, :], dtype=float)

    # 4) Undo scaling back to log-return units
    mu_step = mu_step / scale_returns
    var_step = var_step / (scale_returns ** 2)

    # 5) Convert step-wise (mu_t, var_t) into cumulative to map to prices
    #    Under conditional normality: sum of normals -> Normal(sum mu, sum var)
    cum_mu = np.cumsum(mu_step)
    cum_var = np.cumsum(var_step)

    # 6) Build future business-day index
    if reindex_to_bdays:
        future_idx = pd.bdate_range(start=last_date + pd.offsets.BDay(), periods=periods)
    else:
        # Just add 1..periods days
        future_idx = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq="D")

    # 7) Price mapping:
    #    log P_{t+h} ~ Normal( log P_t + cum_mu[h], cum_var[h] )
    #    Median price (yhat) = exp( E[log P] ) = P_t * exp(cum_mu)
    #    Mean price would be P_t * exp(cum_mu + 0.5*cum_var) — we use median for robustness.
    z = norm.ppf(1 - alpha/2.0)

    logP0 = np.log(last_price)
    log_med = logP0 + cum_mu
    log_lo  = logP0 + cum_mu - z * np.sqrt(cum_var)
    log_hi  = logP0 + cum_mu + z * np.sqrt(cum_var)

    yhat        = np.exp(log_med)
    yhat_lower  = np.exp(log_lo)
    yhat_upper  = np.exp(log_hi)

    out = pd.DataFrame({
        "date": future_idx,
        "yhat": yhat,
        "yhat_lower": yhat_lower,
        "yhat_upper": yhat_upper,
    })
    return out
