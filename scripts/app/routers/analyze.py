from __future__ import annotations

import uuid
from datetime import datetime

import numpy as np
from fastapi import APIRouter, Form, UploadFile

from ..config import settings
from ..schemas import Alert, AnalyzeResponse, AnalyzeSummary, Geo, ShapContribution
from ..services import geoip
from ..services.csv_loader import load_csv, synth_meta
from ..services.label_map import to_severity
from ..services.model import FEATURE_NAMES, LABELS
from ..services.prediction_service import infer_batch, shap_batch

router = APIRouter()


@router.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(csv: UploadFile, include_shap: bool = Form(False)) -> AnalyzeResponse:
    rows = load_csv(csv)

    # predict the whole file in one batch
    X = np.asarray([[row[name] for name in FEATURE_NAMES] for row in rows], dtype=np.float32)
    probs = infer_batch(X)
    pred_idx = probs.argmax(axis=1)
    confidences = probs[np.arange(len(rows)), pred_idx]

    # SHAP is slow, so cap it to the first few flows
    cap = min(settings.ANALYZE_SHAP_CAP, len(rows)) if include_shap else 0
    shap_contribs = shap_batch(X[:cap], pred_idx[:cap]) if cap > 0 else None

    alerts: list[Alert] = []
    by_class: dict[str, int] = {}
    malicious = 0

    for i, row in enumerate(rows):
        label = LABELS[int(pred_idx[i])]
        meta = synth_meta(row)

        geo = None
        if label != "BENIGN":
            geo_data = geoip.lookup(meta.src_ip)
            if geo_data is not None:
                geo = Geo(**geo_data)

        shap_list = None
        if shap_contribs is not None and i < cap:
            shap_list = [
                ShapContribution(feature=name, value=float(shap_contribs[i][j]), raw_input=float(X[i][j]))
                for j, name in enumerate(FEATURE_NAMES)
            ]
            shap_list.sort(key=lambda c: abs(c.value), reverse=True)

        alerts.append(
            Alert(
                id=str(uuid.uuid4()),
                timestamp=meta.timestamp or datetime.utcnow(),
                type=label,
                severity=to_severity(label),
                confidence=float(confidences[i]),
                srcIP=meta.src_ip,
                dstIP=meta.dst_ip,
                protocol=meta.protocol,
                flowDuration=meta.flow_duration,
                fwdPackets=meta.fwd_packets,
                geo=geo,
                shapValues=shap_list,
            )
        )

        by_class[label] = by_class.get(label, 0) + 1
        if label != "BENIGN":
            malicious += 1

    summary = AnalyzeSummary(
        total=len(alerts),
        benign=len(alerts) - malicious,
        malicious=malicious,
        by_class=by_class,
    )
    return AnalyzeResponse(alerts=alerts, summary=summary)
