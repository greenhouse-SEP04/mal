# data_ingestion.py

import os
import requests
import pandas as pd

def fetch_telemetry(
    api_url: str,
    device_mac: str,
    limit: int = 1000,
    csv_path: str = None
) -> pd.DataFrame:
    """
    Fetch sensor telemetry data from the cloud API or load from a local CSV if csv_path is provided.

    If csv_path points to an existing file, this function will read it directly;
    otherwise, it performs an HTTP GET to pull live data.
    """
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=['timestamp'])
        return df

    params = {"dev": device_mac, "limit": limit}
    resp = requests.get(f"{api_url}/v1/telemetry", params=params)
    resp.raise_for_status()

    data = resp.json()
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df
