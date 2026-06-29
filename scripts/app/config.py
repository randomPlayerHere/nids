from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def _env(key: str, default: str) -> str:
    return os.getenv(f"NIDS_{key}", default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(f"NIDS_{key}", str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(f"NIDS_{key}", str(default)))
    except ValueError:
        return default


def _env_list(key: str, default: list[str]) -> list[str]:
    raw = os.getenv(f"NIDS_{key}")
    if not raw:
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


class Settings:
    def __init__(self) -> None:
        self.MODEL_PATH = Path(_env("MODEL_PATH", str(ROOT / "models" / "new" / "nids_model.h5")))
        self.SCALER_PATH = Path(_env("SCALER_PATH", str(ROOT / "models" / "new" / "cicids_scaler.pkl")))
        self.LABELS_PATH = Path(_env("LABELS_PATH", str(ROOT / "models" / "new" / "class_labels.json")))
        self.BACKGROUND_PATH = Path(_env("BACKGROUND_PATH", str(ROOT / "data" / "processed" / "X_dcnn.npy")))
        # committed sample of scaled flows, used when the full datasets aren't shipped
        self.DEMO_FLOWS_PATH = Path(_env("DEMO_FLOWS_PATH", str(ROOT / "data" / "demo_flows.npy")))

        self.BACKGROUND_SIZE = _env_int("BACKGROUND_SIZE", 100)
        self.TOP_K_DEFAULT = _env_int("TOP_K_DEFAULT", 10)

        geoip = os.getenv("NIDS_GEOIP_DB_PATH")
        self.GEOIP_DB_PATH = Path(geoip) if geoip else None

        self.CORS_ORIGINS = _env_list(
            "CORS_ORIGINS",
            ["http://localhost:8080", "http://127.0.0.1:8080", "http://localhost:5173", "http://127.0.0.1:5173"],
        )
        self.STREAM_RATE_HZ = _env_float("STREAM_RATE_HZ", 1.0)

        self.MAX_UPLOAD_ROWS = _env_int("MAX_UPLOAD_ROWS", 5000)
        self.ANALYZE_SHAP_CAP = _env_int("ANALYZE_SHAP_CAP", 12)

        self.LOG_LEVEL = _env("LOG_LEVEL", "INFO").upper()


settings = Settings()

MODEL_PATH = settings.MODEL_PATH
SCALER_PATH = settings.SCALER_PATH
LABELS_PATH = settings.LABELS_PATH
BACKGROUND_PATH = settings.BACKGROUND_PATH
BACKGROUND_SIZE = settings.BACKGROUND_SIZE
TOP_K_DEFAULT = settings.TOP_K_DEFAULT
