from __future__ import annotations
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_model(X, y, seed=42):
    model = HistGradientBoostingRegressor(max_iter=180, learning_rate=0.06, max_leaf_nodes=15,
                                          l2_regularization=1.0, random_state=seed)
    return model.fit(X, y)


def evaluate(y, pred):
    y = np.asarray(y, dtype=float); pred = np.asarray(pred, dtype=float)
    abs_error = np.abs(y-pred)
    denom = np.maximum(np.abs(y)+np.abs(pred), 1e-6)
    return {"n": int(len(y)), "mae": round(float(mean_absolute_error(y,pred)),4),
            "rmse": round(float(np.sqrt(mean_squared_error(y,pred))),4),
            "r2": round(float(r2_score(y,pred)),4),
            "smape_pct": round(float(np.mean(2*abs_error/denom)*100),4),
            "p90_absolute_error": round(float(np.quantile(abs_error,.90)),4),
            "mean_error": round(float(np.mean(pred-y)),4)}
