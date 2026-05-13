"""FastAPI service exposing the NIDS model with SHAP explanations.

Run from project root:
    uvicorn scripts.api:app --reload --port 8000

Endpoints:
    GET  /health      Liveness + model metadata
    GET  /features    Ordered list of feature names the model expects
    POST /predict     Predict + SHAP feature attributions for a single flow
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any



import joblib
import numpy as np
import shap
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "new" / "nids_model.h5"
SCALER_PATH = ROOT / "models" / "new" / "cicids_scaler.pkl"
LABELS_PATH = ROOT / "models" / "new" / "class_labels.json"
BACKGROUND_PATH = ROOT / "data" / "processed" / "X_dcnn.npy"

BACKGROUND_SIZE = 100
TOP_K_DEFAULT = 10


model = tf.keras.models.load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)
with open(LABELS_PATH) as f:
    label_info = json.load(f)

LABELS= label_info["labels"]
N_FEATURES = int(label_info["n_features"])
FEATURE_NAMES = list(getattr(scaler, "feature_names_in_", [f"f{i}" for i in range(N_FEATURES)]))

if len(FEATURE_NAMES) != N_FEATURES:
    raise RuntimeError(
        f"Feature-name count ({len(FEATURE_NAMES)}) does not match model input width ({N_FEATURES})."
    )

_bg_all = np.load(BACKGROUND_PATH)
_rng = np.random.default_rng(42)
_idx = _rng.choice(_bg_all.shape[0], size=min(BACKGROUND_SIZE, _bg_all.shape[0]), replace=False)
background = _bg_all[_idx].astype(np.float32)

explainer = shap.GradientExplainer(model, background)


class PredictRequest(BaseModel):
    features: dict[str, float] | list[float]
    top_k: int = TOP_K_DEFAULT


class ShapContribution(BaseModel):
    feature: str
    value: float
    raw_input: float


class PredictResponse(BaseModel):
    predicted_class: str
    predicted_index: int
    confidence: float
    probabilities: dict[str, float]
    top_contributions: list[ShapContribution]
    all_contributions: list[ShapContribution]


app = FastAPI(title="NIDS API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _vectorize(features: dict[str, float] | list[float]) -> np.ndarray:
    if isinstance(features, list):
        if len(features) != N_FEATURES:
            raise HTTPException(400, f"Expected {N_FEATURES} feature values, got {len(features)}")
        return np.asarray(features, dtype=np.float32)
    missing = [name for name in FEATURE_NAMES if name not in features]
    if missing:
        raise HTTPException(400, f"Missing features: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    return np.asarray([features[name] for name in FEATURE_NAMES], dtype=np.float32)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "n_features": N_FEATURES,
        "n_classes": len(LABELS),
        "labels": LABELS,
        "shap_background_size": int(background.shape[0]),
    }


@app.get("/features")
def features() -> dict[str, list[str]]:
    return {"features": FEATURE_NAMES}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    raw = _vectorize(req.features)
    scaled = scaler.transform(raw.reshape(1, 78)).astype(np.float32)
    x = scaled.reshape(1, N_FEATURES, 1)

    probs = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = LABELS[pred_idx]

    # GradientExplainer returns a list (one tensor per output class) of shape (n_samples, n_features, 1).
    shap_values = explainer.shap_values(x)
    if isinstance(shap_values, list):
        sv = np.asarray(shap_values[pred_idx])
    else:
        sv = np.asarray(shap_values)[..., pred_idx]
    contributions = sv.reshape(-1)
    if contributions.shape[0] != N_FEATURES:
        contributions = contributions[:N_FEATURES]

    all_contribs = []
    for i,(name,val) in enumerate(zip(FEATURE_NAMES, contributions)):
        all_contribs.append(
            ShapContribution(feature=name, value=float(val), raw_input=float(raw[i]))
        )
    top = sorted(all_contribs, key=lambda c: abs(c.value), reverse=True)[: max(1, req.top_k)]

    return PredictResponse(
        predicted_class=pred_label,
        predicted_index=pred_idx,
        confidence=float(probs[pred_idx]),
        probabilities={LABELS[i]: float(p) for i, p in enumerate(probs)},
        top_contributions=top,
        all_contributions=all_contribs,
    )
