from __future__ import annotations
import datetime as dt
import requests
import pandas as pd
from . import config

class ApiClient:
    def __init__(self):
        self._token: str | None = None

    def latest_telemetry(self, limit: int = 10000) -> pd.DataFrame:
        self._ensure_token()
        url = f"{config.API_BASE}/telemetry"
        resp = requests.get(
            url,
            params={"dev": config.DEVICE_MAC, "limit": limit},
            headers={"Authorization": f"Bearer {self._token}"},
            timeout=config.TIMEOUT,
            verify=config.VERIFY_SSL,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            raise ValueError("API returned 0 telemetry rows – cannot train model.")
        df = pd.DataFrame(data)
        df.rename(
            columns={
                "temperature": "Temperature",
                "humidity": "Humidity",
                "soil": "Soil",
                "lux": "Lux",
                "level": "Level",
                "tamper": "Tamper",
                "timestamp": "Timestamp",
            },
            inplace=True,
        )
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
        df.sort_values("Timestamp", inplace=True)
        return df

    def _ensure_token(self) -> None:
        if self._token and not self._token_expired:
            return
        if not config.API_USER or not config.API_PASS:
            raise RuntimeError("API_USER / API_PASS must be set.")
        url = f"{config.API_BASE}/auth/login"
        payload = {"username": config.API_USER, "password": config.API_PASS}
        resp = requests.post(url, json=payload, timeout=config.TIMEOUT, verify=config.VERIFY_SSL)
        resp.raise_for_status()
        self._token = resp.json()["token"]
        self._expires = dt.datetime.utcnow() + dt.timedelta(minutes=55)

    @property
    def _token_expired(self) -> bool:
        return dt.datetime.utcnow() + dt.timedelta(seconds=60) >= self._expires