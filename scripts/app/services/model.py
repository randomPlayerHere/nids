from __future__ import annotations

import json

import joblib
import numpy as np
import shap
import tensorflow as tf
from fastapi import HTTPException

from ..config import (
    BACKGROUND_PATH,
    BACKGROUND_SIZE,
    LABELS_PATH,
    MODEL_PATH,
    SCALER_PATH,
)

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)

with open(LABELS_PATH) as f:
    _label_info = json.load(f)

LABELS: list[str] = _label_info["labels"]
N_FEATURES: int = int(_label_info["n_features"])
FEATURE_NAMES: list[str] = list(
    getattr(scaler, "feature_names_in_", [f"f{i}" for i in range(N_FEATURES)])
)

if len(FEATURE_NAMES) != N_FEATURES:
    raise RuntimeError(
        f"Feature-name count ({len(FEATURE_NAMES)}) does not match model input width ({N_FEATURES})."
    )

_bg_all = np.load(BACKGROUND_PATH)
_rng = np.random.default_rng(42)
_idx = _rng.choice(_bg_all.shape[0], size=min(BACKGROUND_SIZE, _bg_all.shape[0]), replace=False)
background = _bg_all[_idx].astype(np.float32)

explainer = shap.GradientExplainer(model, background)


def vectorize(features: dict[str, float] | list[float]) -> np.ndarray:
    if isinstance(features, list):
        if len(features) != N_FEATURES:
            raise HTTPException(400, f"Expected {N_FEATURES} feature values, got {len(features)}")
        return np.asarray(features, dtype=np.float32)
    missing = [name for name in FEATURE_NAMES if name not in features]
    if missing:
        raise HTTPException(400, f"Missing features: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    return np.asarray([features[name] for name in FEATURE_NAMES], dtype=np.float32)
