import json, os, subprocess, sys
import numpy as np
from aeropulse.data import FEATURES, make_dataset
from aeropulse.split import chronological_split
from aeropulse.train import run

def test_series_is_reproducible_and_valid():
    X,y=make_dataset(480,seed=8); X2,y2=make_dataset(480,seed=8)
    assert X.shape==(480,len(FEATURES)) and y.shape==(480,)
    assert np.isfinite(X).all() and np.isfinite(y).all()
    assert np.array_equal(X,X2) and np.array_equal(y,y2)

def test_split_is_forward_only():
    tr,te=chronological_split(480)
    assert tr.max()<te.min() and len(tr)+len(te)==480

def test_models_metrics_and_cli():
    report=run(seed=9,samples=480)
    for key in ("baseline_seasonal_naive","stronger_hist_gradient_boosting"):
        assert report[key]["n"]==report["n_test"]
        assert report[key]["mae"]>=0 and report[key]["rmse"]>=0
    env=dict(os.environ); env["PYTHONPATH"]=os.pathsep.join([os.getcwd(),env.get("PYTHONPATH","")])
    out=subprocess.run([sys.executable,"-m","aeropulse.train","--samples","480","--json"],check=True,text=True,capture_output=True,env=env)
    assert json.loads(out.stdout)["project"]=='AeroPulse-Air — Hourly PM2.5 Forecasting'
