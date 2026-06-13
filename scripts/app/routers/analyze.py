from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Form, UploadFile

from ..config import settings
from ..schemas import AnalyzeResponse, AnalyzeSummary
from ..services.csv_loader import load_csv, synth_meta
from ..services.flow_to_alert import flow_to_alert
from ..services.model import FEATURE_NAMES

router = APIRouter()


@router.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(csv: UploadFile, include_shap: bool = Form(False)) -> AnalyzeResponse:
    rows = load_csv(csv)

    alerts = []
    by_class: dict[str, int] = {}
    malicious = 0
    shap_used = 0

    for row in rows:
        features = np.asarray([row[name] for name in FEATURE_NAMES], dtype=np.float32)
        meta = synth_meta(row)

        # SHAP is slow, so only do it for the first few flows
        want_shap = include_shap and shap_used < settings.ANALYZE_SHAP_CAP
        if want_shap:
            shap_used += 1

        alert = flow_to_alert(features, meta, include_shap=want_shap, already_scaled=False)
        alerts.append(alert)

        by_class[alert.type] = by_class.get(alert.type, 0) + 1
        if alert.type != "BENIGN":
            malicious += 1

    summary = AnalyzeSummary(
        total=len(alerts),
        benign=len(alerts) - malicious,
        malicious=malicious,
        by_class=by_class,
    )
    return AnalyzeResponse(alerts=alerts, summary=summary)
