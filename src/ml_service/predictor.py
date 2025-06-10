from __future__ import annotations
import joblib
import numpy as np
from typing import Dict,List
from . import config

class SoilMoisturePredictor:
    _MODEL=None; _SX=None; _SY=None
    def _load(self):
        if self._MODEL: return
        arts={"model":config.MODEL_DIR/"latest.pkl",
              "sX":config.MODEL_DIR/"scaler_X.save",
              "sY":config.MODEL_DIR/"scaler_y.save"}
        miss=[k for k,p in arts.items() if not p.exists()]
        if miss: raise FileNotFoundError(f"Missing {miss}")
        self._MODEL=joblib.load(arts["model"])
        self._SX=joblib.load(arts["sX"])
        self._SY=joblib.load(arts["sY"])
    def forecast(self,snap:Dict[str,float])->List[float]:
        self._load()
        vec=np.array([[snap["Soil"],snap["Temperature"],snap["Humidity"]]])
        Xs=self._SX.transform(vec)
        y_s=self._MODEL.predict(Xs)
        y=self._SY.inverse_transform(y_s)[0]
        y=np.minimum(y,snap["Soil"]);
        if config.TOTAL_STEPS>config.MODEL_6H_STEPS:
            d=np.diff(y); w=np.arange(1,len(d)+1)
            drop=-np.dot(d,w)/w.sum() if len(d) else 0.5
            last=y[-1]
            extra=[max(last-drop*(i+1),0) for i in range(config.TOTAL_STEPS-config.MODEL_6H_STEPS)]
            y=np.concatenate([y,extra])
        return y.clip(min=0).tolist()