from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple

FeatureMatrix = Tuple[np.ndarray,np.ndarray,MinMaxScaler,MinMaxScaler]

def extract_drying_cycles(
    df: pd.DataFrame,
    moisture_col: str = "Soil",
    min_cycle_len: int = 2,
    spike_threshold: float = 3.0,
) -> list[list[int]]:
    dry_cycles, cur = [], []
    soil = df[moisture_col].to_numpy()
    for i in range(1,len(df)):
        prev,curr=soil[i-1],soil[i]
        if curr<=prev:
            cur.append(i)
        elif curr-prev>=spike_threshold:
            if len(cur)>=min_cycle_len: dry_cycles.append(cur.copy())
            cur=[i]
        else:
            if len(cur)>=min_cycle_len: dry_cycles.append(cur.copy())
            cur=[]
    if len(cur)>=min_cycle_len: dry_cycles.append(cur)
    return dry_cycles


def build_training_tensors(
    df: pd.DataFrame,
    model_steps: int,
    features: list[str]|None=None,
) -> FeatureMatrix:
    if features is None:
        features=["Soil","Temperature","Humidity"]
    cycles=extract_drying_cycles(df)
    X,y=[],[]
    for cyc in cycles:
        if len(cyc)<model_steps+1: continue
        start=cyc[0]
        fut=cyc[1:1+model_steps]
        X.append(df.iloc[start][features].to_numpy(dtype=float))
        y.append(df.iloc[fut]["Soil"].to_numpy(dtype=float))
    if not X or not y:
        raise ValueError("No valid cycles.")
    Xs=np.array(X); ys=np.array(y)
    sX=MinMaxScaler().fit(Xs); Xs=sX.transform(Xs)
    sY=MinMaxScaler().fit(ys); ys=sY.transform(ys)
    return Xs, ys, sX, sY