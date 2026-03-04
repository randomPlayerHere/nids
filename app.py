"""
Network Intrusion Detection System (NIDS) - FastAPI Backend
Replaces the Streamlit interface with a REST API + static HTML frontend.
"""

import os
import io
import traceback

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ─── Constants ───────────────────────────────────────────────────────────────
MODEL_PATH = "models/nids_dcnn_model.h5"
SCALER_PATH = "models/cicids_scaler.pkl"
LABEL_MAP = {0: "BENIGN", 1: "ATTACK"}

# ─── App setup ───────────────────────────────────────────────────────────────
app = FastAPI(title="Sentinel AI – NIDS Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Load model + scaler at startup ─────────────────────────────────────────
model = None
scaler = None
load_error = None

def load_assets():
    global model, scaler, load_error
    try:
        # Lazy-import tensorflow so the module loads fast if tf isn't needed yet
        from tensorflow.keras.models import load_model as keras_load

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")

        model = keras_load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        load_error = None
        print("✅ Model and scaler loaded successfully")
    except Exception as exc:
        load_error = str(exc)
        print(f"❌ Failed to load ML assets: {load_error}")

load_assets()

# ─── Preprocessing (mirrors streamlit_app.py) ───────────────────────────────
def preprocess_inference(df: pd.DataFrame) -> pd.DataFrame:
    """Drop identifiers, handle inf/NaN – same logic as the Streamlit version."""
    to_drop = ["Flow ID", "Source IP", "Destination IP", "Timestamp"]
    df = df.drop(columns=to_drop, errors="ignore")
    df = df.drop(columns=["Label"], errors="ignore")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df

# ─── Risk assessment (mirrors streamlit_app.py) ─────────────────────────────
def compute_risk(attack_pct: float):
    if attack_pct > 75:
        return "CRITICAL", "Severe attack detected. Immediate action required."
    elif attack_pct > 50:
        return "HIGH", "High volume of malicious traffic detected."
    elif attack_pct > 25:
        return "MEDIUM", "Moderate attack activity detected."
    elif attack_pct > 5:
        return "LOW", "Minor attack activity detected."
    else:
        return "MINIMAL", "Network appears secure with minimal threats."

# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main dashboard HTML."""
    html_path = os.path.join(os.path.dirname(__file__), "code.html")
    with open(html_path, "r") as f:
        return HTMLResponse(content=f.read())


@app.get("/health")
async def health():
    """Report whether the ML model is loaded and ready."""
    if load_error:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "detail": load_error},
        )
    return {
        "status": "ready",
        "expected_features": int(scaler.n_features_in_),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept a CSV upload, run the full NIDS pipeline, and return results as JSON.
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {load_error}")

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    try:
        # 1. Read CSV ─────────────────────────────────────────────────────
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        df.columns = df.columns.str.strip()
        total_rows = len(df)

        # Keep a copy of the original data before preprocessing for display
        original_df = df.copy()

        # 2. Preprocess ───────────────────────────────────────────────────
        processed = preprocess_inference(df)

        expected_features = scaler.feature_names_in_
        if len(processed.columns) != len(expected_features):
            raise ValueError(
                f"Feature mismatch: expected {len(expected_features)}, "
                f"got {len(processed.columns)}"
            )

        missing = set(expected_features) - set(processed.columns)
        if missing:
            raise ValueError(f"Missing columns: {', '.join(list(missing)[:10])}")

        processed = processed[expected_features]

        # 3. Scale + reshape for Conv1D ───────────────────────────────────
        scaled = scaler.transform(processed)
        X_input = scaled.reshape(scaled.shape[0], scaled.shape[1], 1)

        # 4. Predict ──────────────────────────────────────────────────────
        predictions = model.predict(X_input, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        confidence_scores = np.max(predictions, axis=1)

        # 5. Summary statistics ───────────────────────────────────────────
        n_benign = int((predicted_classes == 0).sum())
        n_attack = int((predicted_classes == 1).sum())
        attack_pct = round((n_attack / total_rows) * 100, 2) if total_rows else 0
        avg_confidence = round(float(confidence_scores.mean()) * 100, 2)
        min_confidence = round(float(confidence_scores.min()) * 100, 2)
        max_confidence = round(float(confidence_scores.max()) * 100, 2)

        risk_level, risk_msg = compute_risk(attack_pct)

        # 6. Confidence histogram buckets ─────────────────────────────────
        conf_histogram = {
            "90-100%": int(((confidence_scores >= 0.9) & (confidence_scores <= 1.0)).sum()),
            "80-90%":  int(((confidence_scores >= 0.8) & (confidence_scores < 0.9)).sum()),
            "70-80%":  int(((confidence_scores >= 0.7) & (confidence_scores < 0.8)).sum()),
            "60-70%":  int(((confidence_scores >= 0.6) & (confidence_scores < 0.7)).sum()),
            "<60%":    int((confidence_scores < 0.6).sum()),
        }

        # 7. Pick the most useful columns for the results table ───────────
        # Prefer network-identifiable columns, fall back to whatever exists
        preferred_cols = [
            "Source IP", "Destination IP", "Timestamp", "Protocol",
            "Destination Port", "Flow Duration", "Total Fwd Packets",
            "Total Backward Packets", "Flow Bytes/s", "Flow Packets/s",
            "Fwd Packet Length Mean", "Bwd Packet Length Mean",
            "Flow IAT Mean", "Fwd IAT Mean",
        ]
        display_columns = [c for c in preferred_cols if c in original_df.columns]
        # If none matched, just take the first few columns
        if not display_columns:
            display_columns = list(original_df.columns[:4])
        # Cap at 4 columns for readability
        display_columns = display_columns[:4]

        # 8. Build sample results (top 100) ──────────────────────────────
        limit = min(100, total_rows)
        results_rows = []
        for i in range(limit):
            row = {}
            for col in display_columns:
                val = original_df.iloc[i][col]
                # Format numbers nicely
                if isinstance(val, float):
                    row[col] = round(val, 2) if abs(val) < 1e6 else f"{val:.2e}"
                else:
                    row[col] = str(val)
            row["Label"] = LABEL_MAP[int(predicted_classes[i])]
            row["Confidence"] = round(float(confidence_scores[i]), 4)
            results_rows.append(row)

        # 9. Build CSV string for download ────────────────────────────────
        download_data = {}
        for col in display_columns:
            download_data[col] = original_df[col].tolist()
        download_data["Prediction"] = [LABEL_MAP[c] for c in predicted_classes]
        download_data["Confidence"] = [round(float(s), 4) for s in confidence_scores]
        download_df = pd.DataFrame(download_data)

        return {
            "total_flows": total_rows,
            "n_benign": n_benign,
            "n_attack": n_attack,
            "attack_pct": attack_pct,
            "avg_confidence": avg_confidence,
            "min_confidence": min_confidence,
            "max_confidence": max_confidence,
            "risk_level": risk_level,
            "risk_message": risk_msg,
            "conf_histogram": conf_histogram,
            "columns": display_columns,
            "results": results_rows,
            "csv_download": download_df.to_csv(index=False),
        }

    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


# ─── Run with: python app.py ─────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
