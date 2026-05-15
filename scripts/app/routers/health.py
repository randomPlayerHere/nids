from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from ..services.model import FEATURE_NAMES, LABELS, N_FEATURES, background

router = APIRouter()


@router.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "n_features": N_FEATURES,
        "n_classes": len(LABELS),
        "labels": LABELS,
        "shap_background_size": int(background.shape[0]),
    }


@router.get("/features")
def features() -> dict[str, list[str]]:
    return {"features": FEATURE_NAMES}
