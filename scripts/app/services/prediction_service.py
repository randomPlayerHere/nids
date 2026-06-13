from __future__ import annotations

import numpy as np

from ..schemas import (
    PredictBatchRequest,
    PredictBatchResponse,
    PredictRequest,
    PredictResponse,
    PredictResponseFast,
    ShapContribution,
)
from .model import (
    FEATURE_NAMES,
    LABELS,
    N_FEATURES,
    get_explainer,
    get_model,
    scaler,
    vectorize,
)


def _to_model_input(raw: np.ndarray, already_scaled: bool) -> np.ndarray:
    if already_scaled:
        return raw.reshape(1, N_FEATURES, 1).astype(np.float32)
    scaled = scaler.transform(raw.reshape(1, N_FEATURES)).astype(np.float32)
    return scaled.reshape(1, N_FEATURES, 1)


def infer_fast(raw: np.ndarray, already_scaled: bool = False) -> tuple[str, int, float, np.ndarray]:
    x = _to_model_input(raw, already_scaled)
    probs = get_model().predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    return LABELS[pred_idx], pred_idx, float(probs[pred_idx]), probs


def infer_explained(raw: np.ndarray, already_scaled: bool = False) -> tuple[str, int, float, np.ndarray, np.ndarray]:
    x = _to_model_input(raw, already_scaled)
    probs = get_model().predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))

    shap_values = get_explainer().shap_values(x)
    if isinstance(shap_values, list):
        sv = np.asarray(shap_values[pred_idx])
    else:
        sv = np.asarray(shap_values)[..., pred_idx]
    contributions = sv.reshape(-1)
    if contributions.shape[0] != N_FEATURES:
        contributions = contributions[:N_FEATURES]

    return LABELS[pred_idx], pred_idx, float(probs[pred_idx]), probs, contributions


def predict_fast(req: PredictRequest) -> PredictResponseFast:
    raw = vectorize(req.features)
    label, idx, confidence, _ = infer_fast(raw)
    return PredictResponseFast(predicted_class=label, predicted_index=idx, confidence=confidence)


def predict_explained(req: PredictRequest) -> PredictResponse:
    raw = vectorize(req.features)
    label, idx, confidence, probs, contributions = infer_explained(raw)

    all_contribs = [
        ShapContribution(feature=name, value=float(val), raw_input=float(raw[i]))
        for i, (name, val) in enumerate(zip(FEATURE_NAMES, contributions))
    ]
    top = sorted(all_contribs, key=lambda c: abs(c.value), reverse=True)[: max(1, req.top_k)]

    return PredictResponse(
        predicted_class=label,
        predicted_index=idx,
        confidence=confidence,
        probabilities={LABELS[i]: float(p) for i, p in enumerate(probs)},
        top_contributions=top,
        all_contributions=all_contribs,
    )


def predict_batch(req: PredictBatchRequest) -> PredictBatchResponse:
    result = [predict_fast(PredictRequest(features=flow, top_k=req.top_k)) for flow in req.flows]
    return PredictBatchResponse(flow_result=result)
