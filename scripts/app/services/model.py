from __future__ import annotations

import json
import logging
import time

import joblib
import numpy as np
from fastapi import HTTPException

from ..config import settings

logger = logging.getLogger("nids.model")

# light metadata loads now (no tensorflow needed)
scaler = joblib.load(settings.SCALER_PATH)

with open(settings.LABELS_PATH) as f:
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

# background rows for SHAP (already scaled)
_bg_all = np.load(settings.BACKGROUND_PATH)
_rng = np.random.default_rng(42)
_idx = _rng.choice(_bg_all.shape[0], size=min(settings.BACKGROUND_SIZE, _bg_all.shape[0]), replace=False)
background = _bg_all[_idx].astype(np.float32)

# heavy model + explainer load lazily (see load_models)
model = None
explainer = None


def load_models() -> None:
    global model, explainer
    if model is not None and explainer is not None:
        return

    import shap
    import tensorflow as tf

    t0 = time.perf_counter()
    model = tf.keras.models.load_model(settings.MODEL_PATH, compile=False)
    explainer = shap.GradientExplainer(model, background)
    model.predict(background[:1].reshape(1, N_FEATURES, 1), verbose=0)  # warm up
    logger.info("model + explainer ready in %.2fs", time.perf_counter() - t0)


def get_model():
    if model is None:
        load_models()
    return model


def get_explainer():
    if explainer is None:
        load_models()
    return explainer


def vectorize(features: dict[str, float] | list[float]) -> np.ndarray:
    if isinstance(features, list):
        if len(features) != N_FEATURES:
            raise HTTPException(400, f"Expected {N_FEATURES} feature values, got {len(features)}")
        return np.asarray(features, dtype=np.float32)
    missing = [name for name in FEATURE_NAMES if name not in features]
    if missing:
        raise HTTPException(400, f"Missing features: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    return np.asarray([features[name] for name in FEATURE_NAMES], dtype=np.float32)
