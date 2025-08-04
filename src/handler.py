"""
Greenhouse ML Service — unified handler
--------------------------------------
This version keeps **both** execution modes:
1. **Lambda event mode** (for training / batch prediction) — unchanged API:
   ``handler({"action":"train"|"predict", ...}, context)``.
2. **HTTP inference mode** (for the AVR firmware) — when the Lambda is fronted
   by API Gateway/ALB and receives a typical HTTP JSON request:

      POST /v1/predict  { ... }

   It answers with::

      {"status":"ok","recommendWater":bool,"openVent":bool}

Environment variables required **only for HTTP mode**:
  WATER_MODEL_URI  – S3 URI of the watering model  (e.g. s3://bucket/water.joblib)
  VENT_MODEL_URI   – S3 URI of the venting model   (e.g. s3://bucket/vent.joblib)
  WATER_THRESH     – probability threshold to recommend watering (default 0.5)
  VENT_THRESH      – probability threshold to advise opening the vent (default 0.5)

Both models are cached across warm Lambda invocations for speed.
"""
from __future__ import annotations
import io
import json
import os
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Dict, List, Tuple, Optional

import boto3
import joblib
import numpy as np  # noqa: F401  # kept for user models that may depend on numpy
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, SVR

# ---------------------------------------------------------------------------
# Configuration & constants
# ---------------------------------------------------------------------------
MIN_SAMPLES = int(os.getenv("MIN_SAMPLES", "10"))

# Model URIs & thresholds for HTTP inference mode
WATER_MODEL_URI: Optional[str] = os.getenv("WATER_MODEL_URI")
VENT_MODEL_URI: Optional[str] = os.getenv("VENT_MODEL_URI")
WATER_THRESH: float = float(os.getenv("WATER_THRESH", "0.5"))
VENT_THRESH: float = float(os.getenv("VENT_THRESH", "0.5"))

# Internal model caches (persist across warm invocations)
_water_model: Optional[Pipeline] = None
_vent_model: Optional[Pipeline] = None

# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _split_s3_uri(s3_uri: str) -> Tuple[str, str]:
    if not s3_uri.startswith("s3://"):
        raise ValueError("s3_uri must start with s3://")
    path = PurePosixPath(s3_uri[5:])
    return path.parts[0], "/".join(path.parts[1:])


def _download_s3_to_buffer(s3_uri: str) -> io.BytesIO:
    bucket, key = _split_s3_uri(s3_uri)
    buf = io.BytesIO()
    boto3.client("s3").download_fileobj(bucket, key, buf)
    buf.seek(0)
    return buf


def _upload_buffer_to_s3(buf: io.BytesIO, s3_uri: str) -> None:
    bucket, key = _split_s3_uri(s3_uri)
    buf.seek(0)
    boto3.client("s3").upload_fileobj(buf, bucket, key)

# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def _detect_feature_types(df: pd.DataFrame, target: str | None = None) -> Tuple[List[str], List[str]]:
    """Return ([numeric_cols], [categorical_cols]) excluding *target*."""
    num = [c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in df.columns if c != target and not pd.api.types.is_numeric_dtype(df[c])]
    return num, cat


@dataclass
class Preprocessor:
    numeric_strategy: str = "mean"

    def build(self, df: pd.DataFrame, target: str | None) -> ColumnTransformer:
        num_cols, cat_cols = _detect_feature_types(df, target)
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy=self.numeric_strategy)),
            ("scaler", StandardScaler()),
        ])
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        return ColumnTransformer([
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ])

# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------

MODEL_CATALOG: Dict[str, BaseEstimator] = {
    "logreg": LogisticRegression(max_iter=1000),
    "knn": KNeighborsClassifier(),
    "rf_cls": RandomForestClassifier(n_estimators=150),
    "svm": SVC(probability=True),
    "mlp": MLPClassifier(max_iter=500),
    "ridge": Ridge(),
    "elastic": ElasticNet(),
    "rf_reg": RandomForestRegressor(n_estimators=200),
    "svr": SVR(),
    "kmeans": KMeans(n_clusters=3),
}

REG_METRICS = {
    "rmse": lambda y, p: mean_squared_error(y, p, squared=False),
    "r2": r2_score,
}
CLS_METRICS = {
    "accuracy": accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
}

# ---------------------------------------------------------------------------
# Pipeline and evaluation helpers
# ---------------------------------------------------------------------------

def _build_pipeline(df: pd.DataFrame, target: str | None, model_key: str) -> Pipeline:
    if model_key not in MODEL_CATALOG:
        raise KeyError(f"Unknown model '{model_key}'")
    return Pipeline([
        ("prep", Preprocessor().build(df, target)),
        ("model", MODEL_CATALOG[model_key]),
    ])


def _evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    preds = model.predict(X)
    if y.dtype.kind in "biu" and y.nunique() <= 20:
        m = {n: fn(y, preds) for n, fn in CLS_METRICS.items()}
        m["confusion"] = confusion_matrix(y, preds).tolist()
    else:
        m = {n: fn(y, preds) for n, fn in REG_METRICS.items()}
    return m


def _fit_model(
    df: pd.DataFrame,
    target: str,
    model_key: str,
    hyper: Dict[str, List[Any]] | None = None,
) -> Tuple[Pipeline, Dict[str, Any]]:
    X, y = df.drop(columns=[target]), df[target]
    pipe = _build_pipeline(df, target, model_key)
    if hyper:
        grid = {f"model__{k}": v for k, v in hyper.items()}
        cv = GridSearchCV(pipe, grid, cv=5, n_jobs=-1,
                          scoring="f1" if y.nunique() == 2 else "accuracy")
        cv.fit(X, y)
        best = cv.best_estimator_
        metrics = _evaluate(best, X, y)
        metrics["cv_score"] = cv.best_score_
        return best, metrics
    else:
        pipe.fit(X, y)
        return pipe, _evaluate(pipe, X, y)

# ---------------------------------------------------------------------------
# Inference helpers for HTTP mode
# ---------------------------------------------------------------------------

def _load_model_cached(uri: Optional[str], cache_attr: str) -> Optional[Pipeline]:
    """Load the model from S3 once per cold start and cache it."""
    if not uri:
        return None
    mdl: Optional[Pipeline] = globals()[cache_attr]
    if mdl is None:
        buf = _download_s3_to_buffer(uri)
        mdl = joblib.load(buf)
        globals()[cache_attr] = mdl
    return mdl


def _predict_flags(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return the AVR‑friendly prediction flags."""
    ghid = payload.get("greenhouse_id")

    # -------- Water recommendation --------
    rec_water = False
    wm = _load_model_cached(WATER_MODEL_URI, "_water_model")
    if wm and "soil" in payload:
        Xw = pd.DataFrame([{**{k: payload.get(k) for k in ("soil",)}, "greenhouse_id": ghid}])
        if hasattr(wm[-1], "predict_proba"):
            prob = float(wm.predict_proba(Xw)[0][-1])
            rec_water = prob >= WATER_THRESH
        else:
            rec_water = bool(wm.predict(Xw)[0])
    else:
        # Fallback heuristic: soil below 35 → water
        if "soil" in payload:
            rec_water = int(payload["soil"]) < 35

    # -------- Vent recommendation (force‑open only) --------
    open_vent = False
    vm = _load_model_cached(VENT_MODEL_URI, "_vent_model")
    have_feats = all(k in payload for k in ("temp", "hum", "lux"))
    if vm and have_feats:
        Xv = pd.DataFrame([{**{k: payload.get(k) for k in ("temp", "hum", "lux")}, "greenhouse_id": ghid}])
        if hasattr(vm[-1], "predict_proba"):
            prob = float(vm.predict_proba(Xv)[0][-1])
            open_vent = prob >= VENT_THRESH
        else:
            open_vent = bool(vm.predict(Xv)[0])
    else:
        # Conservative fallback: open if humidity <= 45 %
        if "hum" in payload:
            open_vent = int(payload["hum"]) <= 45

    return {
        "status": "ok",
        "recommendWater": bool(rec_water),
        "openVent": bool(open_vent),
    }

# ---------------------------------------------------------------------------
# Main Lambda handler
# ---------------------------------------------------------------------------

def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:  # noqa: C901
    """Entry‑point supporting two invocation styles."""

    # ---------------------------------------------------
    # 1. HTTP inference mode (AVR firmware ➜ API Gateway)
    # ---------------------------------------------------
    if "body" in event:  # heuristic: API Gateway proxy integration adds this key
        try:
            body = event["body"]
            if event.get("isBase64Encoded"):
                import base64
                body = base64.b64decode(body)
            if isinstance(body, (bytes, bytearray)):
                body = body.decode()
            if isinstance(body, str):
                payload = json.loads(body)
            else:
                payload = body  # already dict
            result = _predict_flags(payload)
            return {
                "statusCode": 200,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps(result),
            }
        except Exception as exc:
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"status": "error", "msg": str(exc)}),
            }

    # ---------------------------------------------------
    # 2. Original event mode (training / batch prediction)
    # ---------------------------------------------------

    action: str = event.get("action", "predict")
    s3_uri: str | None = event.get("s3_uri")
    if not s3_uri:
        raise ValueError("'s3_uri' is required in non‑HTTP invocation mode")

    model_key: str = event.get("model", "rf_cls")
    greenhouse_id: str | None = event.get("greenhouse_id")

    if action == "train":
        target = event["target"]  # raises KeyError if missing
        buf = _download_s3_to_buffer(s3_uri)
        df = pd.read_csv(buf)

        if df.shape[0] < MIN_SAMPLES:
            return {
                "status": "cold_start",
                "model_uri": None,
                "metrics": {"note": f"Need at least {MIN_SAMPLES} rows."},
                "defaults": {"water_ml": 100, "fertilizer_ml": 10},
            }

        if greenhouse_id:
            df["greenhouse_id"] = greenhouse_id

        model, metrics = _fit_model(df, target, model_key, event.get("hyper"))

        # Persist model alongside the dataset (same prefix)
        buf2 = io.BytesIO()
        joblib.dump(model, buf2)

        bucket, key = _split_s3_uri(s3_uri)
        prefix = os.path.dirname(key)
        base = os.path.splitext(os.path.basename(key))[0]
        suffix = f"{greenhouse_id}_" if greenhouse_id else ""
        model_key_name = f"{prefix}/{suffix}{base}_model.joblib"
        out_uri = f"s3://{bucket}/{model_key_name}"
        _upload_buffer_to_s3(buf2, out_uri)

        return {"status": "trained", "model_uri": out_uri, "metrics": metrics}

    elif action == "predict":
        buf = _download_s3_to_buffer(s3_uri)
        model: Pipeline = joblib.load(buf)
        payload = event["payload"]
        X_new = pd.DataFrame(payload)
        if greenhouse_id and "greenhouse_id" not in X_new.columns:
            X_new["greenhouse_id"] = greenhouse_id
        preds = model.predict(X_new).tolist()
        probas = None
        if hasattr(model[-1], "predict_proba"):
            try:
                probas = model.predict_proba(X_new).tolist()
            except Exception:
                probas = None
        return {"status": "predicted", "predictions": preds, "probabilities": probas}

    else:
        raise ValueError("action must be 'train' or 'predict'")
