from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import HTTPException, UploadFile

from .model import FEATURE_NAMES

MAX_ROWS = 5000

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
        raise HTTPException(
            400,
            f"Missing feature columns: {missing[:5]}{'...' if len(missing) > 5 else ''}",
        )
    df = df[FEATURE_NAMES]

    # --- Step 4: clean the values -----------------------------------------
    # NIDS flow CSVs are full of inf / NaN (divide-by-zero in rate features
    # like Flow Bytes/s). Force numeric, then map inf/NaN -> 0 so the scaler
    # and model never see non-finite input.
    df = df.apply(pd.to_numeric, errors="coerce")        # force numeric
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)  # inf/NaN -> 0

    # --- Step 5: enforce row cap ------------------------------------------
    if len(df) > MAX_ROWS:
        raise HTTPException(400, f"Too many rows ({len(df)}). Max is {MAX_ROWS}.")

    # --- Step 6: convert to the row format predict_batch expects ----------
    # list[dict[str, float]] — one dict per flow, keyed by feature name.
    # vectorize() will validate + order these, so this slots straight into
    # PredictBatchRequest(flows=...).
    return df.to_dict(orient="records")
