from __future__ import annotations
import json
import requests
from datetime import datetime
from .predictor import SoilMoisturePredictor
from .data_fetcher import ApiClient
from . import config

predictor=SoilMoisturePredictor()
api=ApiClient()

def _post_forecast(forecast:list[float]):
    url=f"{config.API_BASE}/telemetry"
    payload={"dev":config.DEVICE_MAC,"forecast":forecast,"generated":datetime.utcnow().isoformat()+"Z"}
    try: api._ensure_token()
    except: return
    headers={"Authorization":f"Bearer {api._token}","Content-Type":"application/json"}
    try: requests.post(url,json=payload,headers=headers,timeout=config.TIMEOUT,verify=config.VERIFY_SSL)
    except: pass

def lambda_handler(event,context):
    df=api.latest_telemetry(limit=1)
    if df.empty:
        return {"statusCode":200,"body":json.dumps({"forecast":[]})}
    snap=df.iloc[-1].to_dict()
    fc=predictor.forecast(snap)
    _post_forecast(fc)
    return {"statusCode":200,"body":json.dumps({"forecast":fc})}