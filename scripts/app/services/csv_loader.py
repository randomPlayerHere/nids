from __future__ import annotations

import random

import numpy as np
import pandas as pd
from fastapi import HTTPException, UploadFile

from ..config import settings
from ..schemas import FlowMeta
from .model import FEATURE_NAMES

_PROTOCOLS = ["TCP", "UDP", "ICMP"]


def load_csv(csv: UploadFile) -> list[dict[str, float]]:
    try:
        df = pd.read_csv(csv.file)
    except Exception as e:
        raise HTTPException(400, f"ERROR, CSV not parsed: {e}")
    if df.empty:
        raise HTTPException(400, "CSV is empty")

    df.columns = df.columns.str.strip()

    missing = [name for name in FEATURE_NAMES if name not in df.columns]
    if missing:
        raise HTTPException(400, f"Missing feature columns: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    # column order the scaler/model expect
    df = df[FEATURE_NAMES]

    # flow CSVs carry inf/NaN (e.g. Flow Bytes/s div-by-zero); zero them out
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if len(df) > settings.MAX_UPLOAD_ROWS:
        raise HTTPException(400, f"Too many rows ({len(df)}). Max is {settings.MAX_UPLOAD_ROWS}.")

    return df.to_dict(orient="records")


def _rand_public_ip() -> str:
    return f"{random.randint(1, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}"


def synth_meta(row: dict[str, float]) -> FlowMeta:
    # CSV has no IP/protocol columns; fabricate them for display
    return FlowMeta(
        src_ip=_rand_public_ip(),
        dst_ip=f"10.0.0.{random.randint(1, 254)}",
        protocol=random.choice(_PROTOCOLS),
        flow_duration=int(row.get("Flow Duration", 0)),
        fwd_packets=int(row.get("Total Fwd Packets", 0)),
    )
