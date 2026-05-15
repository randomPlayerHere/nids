from __future__ import annotations

import uuid
from datetime import datetime

import numpy as np

from ..schemas import Alert, FlowMeta, Geo, ShapContribution
from . import geoip
from .label_map import to_severity
from .model import FEATURE_NAMES
from .prediction_service import infer_explained, infer_fast


def flow_to_alert(
    features: np.ndarray,
    meta: FlowMeta,
    include_shap: bool = False,
    already_scaled: bool = False,
) -> Alert:
    if include_shap:
        label, _, confidence, _, contributions = infer_explained(features, already_scaled=already_scaled)
        shap_list = [
            ShapContribution(feature=name, value=float(val), raw_input=float(features.flatten()[i]))
            for i, (name, val) in enumerate(zip(FEATURE_NAMES, contributions))
        ]
        shap_list.sort(key=lambda c: abs(c.value), reverse=True)
    else:
        label, _, confidence, _ = infer_fast(features, already_scaled=already_scaled)
        shap_list = None

    severity = to_severity(label)

    geo = None
    if label != "BENIGN":
        geo_data = geoip.lookup(meta.src_ip)
        if geo_data is not None:
            geo = Geo(**geo_data)

    return Alert(
        id=str(uuid.uuid4()),
        timestamp=meta.timestamp or datetime.utcnow(),
        type=label,
        severity=severity,
        confidence=confidence,
        srcIP=meta.src_ip,
        dstIP=meta.dst_ip,
        protocol=meta.protocol,
        flowDuration=meta.flow_duration,
        fwdPackets=meta.fwd_packets,
        geo=geo,
        shapValues=shap_list,
    )
