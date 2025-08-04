 """
 Greenhouse ML Lambda
 --------------------
 A *single* Lambda function that can **train** a model on telemetry data
 (triggered by an EventBridge schedule) *and* **serve predictions** via an
 HTTP endpoint (API Gateway or direct Lambda invoke).

 Environment variables
 ---------------------
 S3_BUCKET          – the telemetry bucket (and where the model will live)
 MIN_SAMPLES        – abort training if we have fewer rows than this
 AWS_ENDPOINT_URL   – optional; points boto3 to LocalStack during local dev

 S3 layout
 ---------
 <bucket>/YYYY‑MM‑DD.csv               – daily raw telemetry appended by API      ↲
 <bucket>/ml/training.csv              – optional curated training set (CSV)     ↲
 <bucket>/ml/models/<gid>.joblib       – persisted scikit‑learn model            ↲

 The daily CSVs follow the exact column order produced by TelemetryController.
 The *training* handler will automatically derive target labels if
 `ml_water` / `ml_vent` are missing: `water := soil<40`, `vent := hum<50`.

 The model is a single `MultiOutputClassifier(RandomForestClassifier)` with
 two binary targets (water / vent). Hyperparameters are deliberately kept
 conservative so the model fits inside the 15 MB tmp space & 60 s timeout.
 """

 from __future__ import annotations

 import io
 import json
 import logging
 import os
 from typing import Any, Dict, Tuple

 import boto3
 import joblib
 import numpy as np
 import pandas as pd
 from sklearn.ensemble import RandomForestClassifier
 from sklearn.metrics import accuracy_score
 from sklearn.model_selection import train_test_split
 from sklearn.multioutput import MultiOutputClassifier

 # ──────────────────────────────────────────────────────────────────────────────
 # Config & globals
 # ──────────────────────────────────────────────────────────────────────────────

 _BUCKET = os.getenv("S3_BUCKET")
 _MIN_SAMPLES = int(os.getenv("MIN_SAMPLES", "10"))
 _MODEL_PREFIX = "ml/models"  # within bucket
 _LOG = logging.getLogger()
 _LOG.setLevel(logging.INFO)

 _s3 = boto3.client("s3", endpoint_url=os.getenv("AWS_ENDPOINT_URL"))

 # hot‑cache model between invocations to save cold‑start time
 _MODEL_CACHE: Dict[str, Any] = {}


 # ──────────────────────────────────────────────────────────────────────────────
 # Small helpers
 # ──────────────────────────────────────────────────────────────────────────────


 def _parse_s3_uri(uri: str) -> Tuple[str, str]:
     if not uri.startswith("s3://"):
         raise ValueError("s3_uri must start with s3://…")
     bucket, key = uri[5:].split("/", 1)
     return bucket, key


 def _get_object(bucket: str, key: str) -> bytes:
     return _s3.get_object(Bucket=bucket, Key=key)["Body"].read()


 def _put_object(bucket: str, key: str, body: bytes) -> None:
     _s3.put_object(Bucket=bucket, Key=key, Body=body)


 # ──────────────────────────────────────────────────────────────────────────────
 # Data wrangling
 # ──────────────────────────────────────────────────────────────────────────────


 _CSV_COLS = [
     "device_mac",
     "timestamp",
     "temp",
     "hum",
     "soil",
     "lux",
     "level",
     "motion",
     "tamper",
     "ax",
     "ay",
     "az",
     "ml_water",
     "ml_vent",
 ]


 def _load_csv(buf: bytes) -> pd.DataFrame:
     """Read a Greenhouse telemetry CSV into a *typed* DataFrame."""

     df = pd.read_csv(io.BytesIO(buf), header=None, names=_CSV_COLS)

     # Convert obvious boolean-ish fields
     df[["motion", "tamper"]] = df[["motion", "tamper"]].astype(int)
     return df


 def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
     """Return X with numeric features only & no NaNs."""

     X = df[[
         "temp",
         "hum",
         "soil",
         "lux",
         "level",
         "motion",
         "tamper",
         "ax",
         "ay",
         "az",
     ]].copy()
     return X.dropna()


 def _derive_targets(df: pd.DataFrame) -> pd.DataFrame:
     """Either use provided labels or derive simple heuristics."""

     if df["ml_water"].notna().any():
         y_water = df["ml_water"].fillna(0).astype(int)
     else:
         y_water = (df["soil"] < 40).astype(int)

     if df["ml_vent"].notna().any():
         y_vent = df["ml_vent"].fillna(0).astype(int)
     else:
         y_vent = (df["hum"] < 50).astype(int)

     return pd.concat([y_water, y_vent], axis=1).rename(columns={0: "water", 1: "vent"})


 # ──────────────────────────────────────────────────────────────────────────────
 # Model I/O
 # ──────────────────────────────────────────────────────────────────────────────


 def _model_key(gid: str) -> str:
     return f"{_MODEL_PREFIX}/{gid}.joblib"


 def _load_model(gid: str):
     """Fetch & unpickle a model; cached for warm invocations."""

     if gid in _MODEL_CACHE:
         return _MODEL_CACHE[gid]

     key = _model_key(gid)
     try:
         raw = _get_object(_BUCKET, key)
     except _s3.exceptions.NoSuchKey:
         _LOG.warning("No model at s3://%s/%s", _BUCKET, key)
         return None

     model = joblib.load(io.BytesIO(raw))
     _MODEL_CACHE[gid] = model
     return model


 def _save_model(gid: str, model) -> str:
     key = _model_key(gid)
     buf = io.BytesIO()
     joblib.dump(model, buf)
     _put_object(_BUCKET, key, buf.getvalue())
     return key


 # ──────────────────────────────────────────────────────────────────────────────
 # Training
 # ──────────────────────────────────────────────────────────────────────────────


 def _handle_train(event: Dict[str, Any]) -> Dict[str, Any]:
     s3_uri = event.get("s3_uri")
     gid = event.get("greenhouse_id", "default")

     if not s3_uri:
         raise KeyError("s3_uri missing in training event")

     bucket, key = _parse_s3_uri(s3_uri)
     _LOG.info("Training from %s/%s", bucket, key)

     df = _load_csv(_get_object(bucket, key))

     if len(df) < _MIN_SAMPLES:
         msg = f"{len(df)} samples < MIN_SAMPLES({_MIN_SAMPLES})"
         _LOG.error(msg)
         return {"ok": False, "reason": msg}

     X = _prepare_features(df)
     y = _derive_targets(df).loc[X.index]  # align indices

     X_tr, X_te, y_tr, y_te = train_test_split(
         X, y, test_size=0.2, random_state=42, stratify=y
     )

     base = RandomForestClassifier(
         n_estimators=200,
         min_samples_leaf=3,
         n_jobs=-1,
         class_weight="balanced",
         random_state=42,
     )
     model = MultiOutputClassifier(base)
     model.fit(X_tr, y_tr)

     # quick metrics
     y_pred = model.predict(X_te)
     acc = {
         "water": float(accuracy_score(y_te["water"], y_pred[:, 0])),
         "vent": float(accuracy_score(y_te["vent"], y_pred[:, 1])),
     }

     key_out = _save_model(gid, model)
     _LOG.info("Saved model → s3://%s/%s", _BUCKET, key_out)

     return {"ok": True, "samples": len(df), "model_key": key_out, "accuracy": acc}


 # ──────────────────────────────────────────────────────────────────────────────
 # Prediction (API Gateway v1/v2 or direct invoke)
 # ──────────────────────────────────────────────────────────────────────────────


 def _extract_features(payload: Dict[str, Any]) -> np.ndarray:
     """Convert JSON payload → 2‑D numpy array accepted by sklearn."""

     feats = [
         payload["temp"],
         payload["hum"],
         payload["soil"],
         payload["lux"],
         payload["level"],
         int(payload.get("motion", 0)),
         int(payload.get("tamper", 0)),
         payload.get("ax", 0),
         payload.get("ay", 0),
         payload.get("az", 0),
     ]
     return np.asarray(feats, dtype=float).reshape(1, -1)


 def _handle_predict(event: Dict[str, Any]):
     gid = (
         event.get("pathParameters", {}).get("id")
         or event.get("greenhouse_id")
         or "default"
     )

     body = event.get("body")
     if isinstance(body, str):
         body = json.loads(body)

     X = _extract_features(body)
     model = _load_model(gid)
     if model is None:
         return _resp(503, {"error": "model_not_trained"})

     pred_water, pred_vent = model.predict(X)[0]
     return _resp(200, {"water": bool(pred_water), "vent": bool(pred_vent)})


 def _resp(status: int, body: Dict[str, Any]) -> Dict[str, Any]:
     return {
         "statusCode": status,
         "headers": {
             "Content-Type": "application/json",
             "Access-Control-Allow-Origin": "*",
         },
         "body": json.dumps(body),
     }


 # ──────────────────────────────────────────────────────────────────────────────
 # Lambda entrypoint
 # ──────────────────────────────────────────────────────────────────────────────


 def handler(event: Dict[str, Any], context):  # noqa: D401, N802
     """Single entry‑point for both **training** and **prediction**."""

     _LOG.info("event=%s", json.dumps(event)[:400])

     # 1️⃣ Scheduled training: {"action":"train", ...}
     if event.get("action") == "train":
         return _handle_train(event)

     # 2️⃣ API Gateway prediction (any version)
     if event.get("httpMethod") or "temp" in event:
         # The second condition lets you test locally with plain feature JSON.
         return _handle_predict(event)

     raise RuntimeError("Unsupported invocation format")
