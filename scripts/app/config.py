from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

MODEL_PATH = ROOT / "models" / "new" / "nids_model.h5"
SCALER_PATH = ROOT / "models" / "new" / "cicids_scaler.pkl"
LABELS_PATH = ROOT / "models" / "new" / "class_labels.json"
BACKGROUND_PATH = ROOT / "data" / "processed" / "X_dcnn.npy"

BACKGROUND_SIZE = 100
TOP_K_DEFAULT = 10
