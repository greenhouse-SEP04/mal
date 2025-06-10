from __future__ import annotations
import os
from pathlib import Path

API_BASE: str = os.getenv("API_BASE", "https://api.example.com/v1")
API_USER: str | None = os.getenv("API_USER")
API_PASS: str | None = os.getenv("API_PASS")
DEVICE_MAC: str = os.getenv("DEVICE_MAC", "AA:BB:CC:DD:EE:FF")

MODEL_6H_STEPS: int = int(os.getenv("MODEL_6H_STEPS", 5))
TOTAL_STEPS: int = int(os.getenv("TOTAL_STEPS", 28))
LOOKBACK_DAYS: int = int(os.getenv("LOOKBACK_DAYS", 30))

_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR: Path = Path(os.getenv("MODEL_DIR", _ROOT / "models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

VERIFY_SSL: bool = os.getenv("VERIFY_SSL", "true").lower() == "true"
TIMEOUT: float = float(os.getenv("HTTP_TIMEOUT", 10))