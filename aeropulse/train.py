"""Train/evaluate CLI for AeroPulse-Air — Hourly PM2.5 Forecasting."""
from __future__ import annotations
import argparse, json
import numpy as np
from .data import FEATURES, BASELINE_FEATURE, UNIT, make_dataset
from .model import evaluate, train_model
from .split import chronological_split

PROJECT = 'AeroPulse-Air — Hourly PM2.5 Forecasting'

def run(seed=42, samples=1200):
    X,y=make_dataset(n=samples,seed=seed)
    tr,te=chronological_split(len(y))
    model=train_model(X[tr],y[tr],seed=seed)
    baseline=X[te,FEATURES.index(BASELINE_FEATURE)]
    learned=model.predict(X[te])
    return {"project":PROJECT,"data":"synthetic demonstration data","seed":int(seed),"unit":UNIT,
            "split":"chronological 75/25 holdout; rolling one-step evaluation using observed prior targets",
            "n_train":int(len(tr)),"n_test":int(len(te)),
            "baseline_seasonal_naive":evaluate(y[te],baseline),
            "stronger_hist_gradient_boosting":evaluate(y[te],learned),
            "error_analysis":{"stronger_model_p90_absolute_error":round(float(np.quantile(np.abs(y[te]-learned),.90)),4),
                              "stronger_model_mean_bias":round(float(np.mean(learned-y[te])),4)}}

def main(argv=None):
    ap=argparse.ArgumentParser(description="Train and evaluate a synthetic forecasting prototype.")
    ap.add_argument("--seed",type=int,default=42); ap.add_argument("--samples",type=int,default=1200); ap.add_argument("--json",action="store_true")
    a=ap.parse_args(argv); print(json.dumps(run(a.seed,a.samples),indent=2))
if __name__=="__main__": main()
