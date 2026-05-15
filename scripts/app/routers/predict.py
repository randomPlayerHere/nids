from __future__ import annotations

from fastapi import APIRouter

from ..schemas import PredictRequest, PredictResponse, PredictResponseFast

from ..services.prediction_service import predict_explained, predict_fast

router = APIRouter()


@router.post("/predict", response_model=PredictResponseFast)
def predict(req: PredictRequest) -> PredictResponseFast:
    return predict_fast(req)


@router.post("/predict/explain", response_model=PredictResponse)
def predict_with_shap(req: PredictRequest) -> PredictResponse:
    return predict_explained(req)
