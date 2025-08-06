import os
import io
import json
import logging
import datetime as dt
from typing import Dict, Tuple, Optional, List

import boto3
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

# ---------- Logging ----------
logger = logging.getLogger()
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)

# ---------- Environment ----------
S3_BUCKET = os.getenv("S3_BUCKET", "")
MIN_SAMPLES = int(os.getenv("MIN_SAMPLES", "10"))
AWS_ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL")  # for LocalStack

_s3 = boto3.client("s3", endpoint_url=AWS_ENDPOINT_URL) if AWS_ENDPOINT_URL else boto3.client("s3")

# ---------- Schema ----------
# Expected columns from TelemetryController/S3Appender (CSV):
# DeviceMac, Timestamp (ISO), Temperature, Humidity, Soil, Lux, Level,
# Motion, Tamper, AccelX, AccelY, AccelZ, MlWater, MlVent
BASE_FEATURES: List[str] = [
    "Temperature", "Humidity", "Soil", "Lux", "Level",
    "Motion", "Tamper", "AccelX", "AccelY", "AccelZ"
]
LABELS = {
    "water": "MlWater",
    "vent": "MlVent",
}

# ---------- Utilities ----------
def parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
    assert s3_uri.startswith("s3://"), "s3_uri must start with s3://"
    _, _, rest = s3_uri.partition("s3://")
    bucket, _, key = rest.partition("/")
    return bucket, key

def s3_get_csv(s3_uri: Optional[str] = None, *, bucket: Optional[str] = None, key: Optional[str] = None) -> pd.DataFrame:
    if s3_uri:
        bucket, key = parse_s3_uri(s3_uri)
    if not bucket:
        bucket = S3_BUCKET
    if not bucket or not key:
        raise ValueError("No S3 location provided for training data.")
    logger.info(f"Loading CSV from s3://{bucket}/{key}")
    obj = _s3.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read()
    return pd.read_csv(io.BytesIO(body))

def s3_put_json(payload: dict, bucket: str, key: str) -> None:
    logger.info(f"Writing JSON to s3://{bucket}/{key}")
    _s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(payload).encode("utf-8"), ContentType="application/json")

def s3_put_joblib(obj, bucket: str, key: str) -> None:
    logger.info(f"Writing model to s3://{bucket}/{key}")
    buf = io.BytesIO()
    joblib.dump(obj, buf)
    buf.seek(0)
    _s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())

def s3_get_joblib(bucket: str, key: str):
    logger.info(f"Loading model from s3://{bucket}/{key}")
    obj = _s3.get_object(Bucket=bucket, Key=key)
    return joblib.load(io.BytesIO(obj["Body"].read()))

def today_str() -> str:
    return dt.datetime.utcnow().strftime("%Y-%m-%d")

def model_key(greenhouse_id: str, target: str) -> str:
    return f"ml/models/{greenhouse_id}/{target}_rf_cls.joblib"

def metrics_key(greenhouse_id: str, target: str) -> str:
    return f"ml/metrics/{greenhouse_id}/{target}_rf_cls_{today_str()}.json"

# ---------- Data Prep ----------
def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    # Basic cleansing / type coercion, safe for Lambda
    df = df.copy()
    # Coerce numeric
    for c in BASE_FEATURES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Booleans for Motion/Tamper may come as strings
    for c in ["Motion", "Tamper"]:
        if c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].astype(str).str.lower().isin(["1", "true", "yes"])
            df[c] = df[c].astype(int)  # 0/1
    # Timestamp-derived hour feature (optional, helps a bit for “night” behavior)
    if "Timestamp" in df.columns:
        ts = pd.to_datetime(df["Timestamp"], errors="coerce", utc=True)
        df["Hour"] = ts.dt.hour.fillna(0).astype(int)
    else:
        df["Hour"] = 0
    return df

def ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only the features we can handle; add Hour if present
    feats = BASE_FEATURES + (["Hour"] if "Hour" in df.columns else [])
    missing = [c for c in BASE_FEATURES if c not in df.columns]
    if missing:
        logger.warning(f"Missing feature columns in CSV: {missing}")
    present = [c for c in feats if c in df.columns]
    return df[present]

def derive_labels_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Heuristic labels only if ground truth is missing.
    # WATER: recommend ON if Soil <= 40 (%). This is a conservative, transparent rule.
    if LABELS["water"] not in df.columns:
        df[LABELS["water"]] = (df.get("Soil", pd.Series([np.nan]*len(df))).fillna(100) <= 40).astype(int)
        logger.info("Derived MlWater label from Soil<=40 heuristic.")
    # VENT: recommend ON (open) if Humidity <= 45 (%); OFF otherwise (close).
    if LABELS["vent"] not in df.columns:
        df[LABELS["vent"]] = (df.get("Humidity", pd.Series([np.nan]*len(df))).fillna(100) <= 45).astype(int)
        logger.info("Derived MlVent label from Humidity<=45 heuristic.")
    return df

def split_xy(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    y_col = LABELS[target]
    if y_col not in df.columns:
        raise ValueError(f"Target column '{y_col}' not in dataset.")
    X = ensure_features(df)
    y = df[y_col].astype(int)
    # Simple NA handling
    X = X.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(method="bfill").fillna(0)
    return X, y

# ---------- Training ----------
def train_one(df: pd.DataFrame, target: str, greenhouse_id: str) -> Dict:
    # Prepare
    df = coerce_types(df)
    df = derive_labels_if_missing(df)
    X, y = split_xy(df, target)

    if len(X) < max(MIN_SAMPLES, 5):
        msg = f"Not enough samples for '{target}' (have {len(X)}, need >= {MAX_SAMPLES})."
        logger.warning(msg)
        return {"trained": False, "reason": msg}

    # Split
    stratify = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify
    )

    # Model (RF is robust, no scaling required)
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )
    model.fit(X_train, y_train)

    # Metrics
    y_pred = model.predict(X_test)
    metrics = {
        "count": int(len(df)),
        "features": list(X.columns),
        "target": LABELS[target],
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    # Persist model + metrics
    mkey = model_key(greenhouse_id, target)
    s3_put_joblib(model, S3_BUCKET, mkey)

    metrics["model_s3"] = f"s3://{S3_BUCKET}/{mkey}"
    s3_put_json(metrics, S3_BUCKET, metrics_key(greenhouse_id, target))

    return {"trained": True, "metrics": metrics}

def train_all(df: pd.DataFrame, greenhouse_id: str, targets: Optional[List[str]] = None) -> Dict:
    targets = targets or ["water", "vent"]
    out = {}
    for t in targets:
        try:
            out[t] = train_one(df, t, greenhouse_id)
        except Exception as e:
            logger.exception(f"Training failed for target '{t}'")
            out[t] = {"trained": False, "error": str(e)}
    return out

# ---------- Prediction ----------
def load_model(greenhouse_id: str, target: str):
    key = model_key(greenhouse_id, target)
    return s3_get_joblib(S3_BUCKET, key)

def predict_one(features: Dict, greenhouse_id: str, target: str) -> Dict:
    # Minimal schema validation & ordering
    df = pd.DataFrame([features])
    df = coerce_types(df)
    X = ensure_features(df)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    model = load_model(greenhouse_id, target)
    proba = model.predict_proba(X)[0][1] if hasattr(model, "predict_proba") else float(model.predict(X)[0])
    pred = int(proba >= 0.5)
    return {"target": target, "prediction": pred, "probability": float(proba)}

# ---------- Lambda Entry ----------
def handler(event, context):
    """
    Actions:
      - {"action":"train", "s3_uri":"s3://bucket/ml/training.csv", "greenhouse_id":"gh-001", "target":"water|vent" }
        If 'target' omitted or "target", trains both 'water' and 'vent'.
      - {"action":"predict", "greenhouse_id":"gh-001", "target":"water|vent", "features": {...} }
        Features keys should be among BASE_FEATURES; extra keys are ignored.
      - {"action":"ping"}
    """
    try:
        action = (event or {}).get("action", "").lower()
        logger.info(f"Event action: {action} | event={json.dumps(event)}")

        if action == "ping":
            return _response(200, {"ok": True, "ts": dt.datetime.utcnow().isoformat() + "Z"})

        if action == "train":
            greenhouse_id = event.get("greenhouse_id") or "default"
            s3_uri = event.get("s3_uri")  # preferred, e.g., s3://bucket/ml/training.csv
            bucket = None
            key = None
            if not s3_uri:
                # Optional fallback: allow {"bucket": "...", "key": "..."}
                bucket = event.get("bucket") or S3_BUCKET
                key = event.get("key")

            df = s3_get_csv(s3_uri, bucket=bucket, key=key)
            targets = None
            t = (event.get("target") or "").lower()
            if t in ("water", "vent"):
                targets = [t]
            # Else: train both
            result = train_all(df, greenhouse_id, targets)
            return _response(200, {"ok": True, "result": result})

        if action == "predict":
            greenhouse_id = event.get("greenhouse_id") or "default"
            target = (event.get("target") or "").lower()
            if target not in ("water", "vent"):
                return _response(400, {"ok": False, "error": "target must be 'water' or 'vent'"})
            features = event.get("features") or {}
            # Keep only known features; allow graceful ignore of extras
            for k in list(features.keys()):
                if k not in BASE_FEATURES and k != "Hour":
                    features.pop(k, None)
            # Derive Hour if user passed Timestamp (optional) — not required
            if "Timestamp" in features and "Hour" not in features:
                try:
                    features["Hour"] = pd.to_datetime(features["Timestamp"], utc=True).hour
                except Exception:
                    features["Hour"] = 0
            result = predict_one(features, greenhouse_id, target)
            return _response(200, {"ok": True, "result": result})

        return _response(400, {"ok": False, "error": "Unknown or missing 'action'."})

    except Exception as e:
        logger.exception("Unhandled error")
        return _response(500, {"ok": False, "error": str(e)})

def _response(status: int, body: dict):
    # Works for both EventBridge & API Gateway lambda-proxy integrations
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }
