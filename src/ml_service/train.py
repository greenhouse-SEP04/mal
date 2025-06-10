from __future__ import annotations
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime,timedelta
from sklearn.ensemble import RandomForestRegressor
from . import config
from .data_fetcher import ApiClient
from .preprocessing import build_training_tensors

def main(now: datetime|None=None)->Path:
    if now is None: now=datetime.utcnow()
    client=ApiClient()
    raw=client.latest_telemetry(limit=50000)
    oldest=now-timedelta(days=config.LOOKBACK_DAYS)
    df=raw[raw["Timestamp"]>=oldest]
    if len(df)<500:
        raise RuntimeError("Not enough data.")
    Xs,Ys,sX,sY=build_training_tensors(df,config.MODEL_6H_STEPS)
    model=RandomForestRegressor(n_estimators=150,random_state=42)
    model.fit(Xs,Ys)
    stamp=now.strftime("%Y%m%dT%H%M%SZ")
    mf=config.MODEL_DIR/f"soil_moisture_rf_{stamp}.pkl"
    joblib.dump(model,mf)
    joblib.dump(sX,config.MODEL_DIR/"scaler_X.save")
    joblib.dump(sY,config.MODEL_DIR/"scaler_y.save")
    latest=config.MODEL_DIR/"latest.pkl"
    try:
        if latest.exists(): latest.unlink()
        latest.symlink_to(mf.name)
    except OSError:
        joblib.dump(model,latest)
    print(f"Trained → {mf.name}")
    return mf

if __name__=="__main__": main()