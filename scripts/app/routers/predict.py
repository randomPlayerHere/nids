from __future__ import annotations

from fastapi import APIRouter, UploadFile

from ..config import TOP_K_DEFAULT

from ..schemas import PredictRequest, PredictResponse, PredictResponseFast, PredictBatchRequest, PredictBatchResponse

from ..services.prediction_service import predict_explained, predict_fast, predict_batch
from ..services.csv_loader import load_csv

router = APIRouter()


@router.post("/predict", response_model=PredictResponseFast)
def predict(req: PredictRequest) -> PredictResponseFast:
    return predict_fast(req)


@router.post("/predict/explain", response_model=PredictResponse)
def predict_with_shap(req: PredictRequest) -> PredictResponse:
    return predict_explained(req)

@router.post("/predict/batch", response_model=PredictBatchResponse)
def predict_csv(csv: UploadFile) -> PredictBatchResponse:
    flows =load_csv(csv)
    req = PredictBatchRequest(
        flows=flows,
        top_k=TOP_K_DEFAULT
    )
    return predict_batch(req)