"""Synthetic chronological series for AeroPulse-Air — Hourly PM2.5 Forecasting; no external data is used."""

from __future__ import annotations
import numpy as np

FEATURES = ["hour_sin", "hour_cos", "week_sin", "week_cos", "trend", "temperature", "rainfall", "event_flag", "lag_1", "lag_24", "lag_168", "rolling_mean_24"]
BASELINE_FEATURE = "lag_24"


UNIT = 'uncalibrated PM2.5-like synthetic units'
def make_dataset(n=1200, seed=42):
    if n < 240:
        raise ValueError("n must be at least 240 to support a 168-hour history and holdout")
    rng = np.random.default_rng(seed)
    t=np.arange(168,168+n); full=np.arange(n+168); hour=full%24; dow=(full//24)%7
    wind=np.clip(rng.gamma(2,2,n+168),0,18); rain=np.clip(rng.gamma(1,1.5,n+168),0,18); temperature=21+10*np.sin(2*np.pi*(full/24-5)/365)+rng.normal(0,2,n+168)
    event=(rng.random(n+168)<.03).astype(float); noise=rng.normal(0,5,n+168)
    season=10*np.sin(2*np.pi*full/(24*365))
    series=np.maximum(1,28+season+5*np.sin(2*np.pi*(hour-7)/24)+4*np.sin(2*np.pi*dow/7)-1.3*wind-.6*rain+.4*(temperature<10)+19*event+noise)
    X=np.column_stack([np.sin(2*np.pi*hour[t]/24),np.cos(2*np.pi*hour[t]/24),np.sin(2*np.pi*dow[t]/7),np.cos(2*np.pi*dow[t]/7),t/(n+168),temperature[t],rain[t],event[t],series[t-1],series[t-24],series[t-168],np.array([series[k-24:k].mean() for k in t])])
    y=series[t]
    if X.shape != (n, len(FEATURES)) or y.shape != (n,) or not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("generator produced invalid time-series rows")
    return X.astype(float), y.astype(float)
