from __future__ import annotations

import json
import logging
import time

import joblib
import numpy as np
from fastapi import HTTPException

from ..config import settings

logger = logging.getLogger("nids.model")

# metadata only here; tensorflow stays out of import time
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

# SHAP background, already scaled to [0, 1]. Use the processed sample if present,
# else demo_flows, else a synthetic background so a clean clone still runs.
_rng = np.random.default_rng(42)
_bg_source = (
    settings.BACKGROUND_PATH if settings.BACKGROUND_PATH.exists()
    else settings.DEMO_FLOWS_PATH if settings.DEMO_FLOWS_PATH.exists()
    else None
)
if _bg_source is not None:
    _bg_all = np.load(_bg_source)
    _idx = _rng.choice(_bg_all.shape[0], size=min(settings.BACKGROUND_SIZE, _bg_all.shape[0]), replace=False)
    background = _bg_all[_idx].astype(np.float32)
    logger.info("SHAP background loaded from %s (%d rows)", _bg_source.name, background.shape[0])
else:
    logger.warning(
        "No SHAP background file found, using a synthetic [0,1] background of %d rows.",
        settings.BACKGROUND_SIZE,
    )
    background = _rng.random((settings.BACKGROUND_SIZE, N_FEATURES, 1), dtype=np.float32)

# model + explainer load lazily, see load_models
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
    model.predict(background[:1].reshape(1, N_FEATURES, 1), verbose=0)  # warmup
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
