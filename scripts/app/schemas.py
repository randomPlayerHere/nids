from __future__ import annotations

from pydantic import BaseModel
from datetime import datetime
from typing import Literal

from .config import TOP_K_DEFAULT
from .services.label_map import Severity


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

class PredictResponseFast(BaseModel):
    predicted_class: str
    predicted_index: int
    confidence: float

class Geo(BaseModel):
    lat: float
    lng: float
    city: str


class FlowMeta(BaseModel):
    src_ip: str
    dst_ip: str
    protocol: str
    flow_duration: int
    fwd_packets: int
    timestamp: datetime | None = None

class Alert(BaseModel):
    id: str
    timestamp: datetime | None = None
    type: str
    severity: Severity
    confidence: float
    srcIP: str
    dstIP: str
    protocol: str
    flowDuration: int
    fwdPackets: int
    geo: Geo | None = None
    shapValues: list[ShapContribution] | None = None
