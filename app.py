import os
import io
import traceback

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.run_functions_eagerly(False)
except:
    pass

TFLITE_MODEL_PATH = "models/nids_dcnn_model.tflite"
H5_MODEL_PATH = "models/nids_dcnn_model.h5"
SCALER_PATH = "models/cicids_scaler.pkl"
LABEL_MAP = {0: "BENIGN", 1: "ATTACK"}

MAX_ROWS_PER_REQUEST = 10000   # Hard limit for low-memory tiers
PREDICT_BATCH_SIZE = 64        # Smaller batches = less peak memory
MAX_DOWNLOAD_ROWS = 500        # Cap CSV response size
MAX_DISPLAY_ROWS = 30          # Rows to show in results table

app = FastAPI(title="Sentinel AI - NIDS Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
tflite_interpreter = None
scaler = None
load_error = None
use_tflite = False

def load_assets():
    global model, tflite_interpreter, scaler, load_error, use_tflite
    try:
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")
        scaler = joblib.load(SCALER_PATH)

        if os.path.exists(TFLITE_MODEL_PATH):
            import tensorflow as tf
            tflite_interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
            tflite_interpreter.allocate_tensors()
            use_tflite = True
            print("TFLite model and scaler loaded successfully (low-memory mode)")
        elif os.path.exists(H5_MODEL_PATH):
            from tensorflow.keras.models import load_model as keras_load
            model = keras_load(H5_MODEL_PATH)
            use_tflite = False
            print("Keras H5 model and scaler loaded successfully")
        else:
            raise FileNotFoundError(f"No model found at {TFLITE_MODEL_PATH} or {H5_MODEL_PATH}")

        load_error = None
    except Exception as exc:
        load_error = str(exc)
        print(f"Failed to load ML assets: {load_error}")

load_assets()

def preprocess_inference(df: pd.DataFrame) -> pd.DataFrame:
    to_drop = ["Flow ID", "Source IP", "Destination IP", "Timestamp"]
    df = df.drop(columns=to_drop, errors="ignore")
    df = df.drop(columns=["Label"], errors="ignore")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df

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

@app.head("/")
async def head_root():
    return HTMLResponse(content="")


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = os.path.join(os.path.dirname(__file__), "code.html")
    with open(html_path, "r") as f:
        return HTMLResponse(content=f.read())


@app.get("/sample-dataset")
async def download_sample():
    csv_path = os.path.join(os.path.dirname(__file__), "test.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="Sample dataset not found.")
    return FileResponse(
        csv_path,
        media_type="text/csv",
        filename="sample_network_data.csv",
    )


@app.get("/health")
async def health():
    if load_error:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "detail": load_error},
        )
    return {
        "status": "ready",
        "expected_features": int(scaler.n_features_in_),
        "model_type": "tflite" if use_tflite else "keras",
        "max_rows": MAX_ROWS_PER_REQUEST,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if (model is None and tflite_interpreter is None) or scaler is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {load_error}")

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        df.columns = df.columns.str.strip()
        total_rows = len(df)
        if total_rows == 0:
            raise ValueError("Uploaded CSV is empty.")
        if total_rows > MAX_ROWS_PER_REQUEST:
            raise ValueError(
                f"Too many rows ({total_rows}). Max allowed is {MAX_ROWS_PER_REQUEST} rows."
            )
        original_df = df.copy()
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
        predicted_chunks = []
        confidence_chunks = []

        if use_tflite:
            input_details = tflite_interpreter.get_input_details()
            output_details = tflite_interpreter.get_output_details()

            for i in range(0, total_rows, PREDICT_BATCH_SIZE):
                batch_df = processed.iloc[i:i + PREDICT_BATCH_SIZE]
                scaled_batch = scaler.transform(batch_df).astype(np.float32)
                x_batch = scaled_batch.reshape(scaled_batch.shape[0], scaled_batch.shape[1], 1)

                batch_preds = []
                for row_idx in range(x_batch.shape[0]):
                    single_input = x_batch[row_idx:row_idx+1]
                    tflite_interpreter.resize_tensor_input(input_details[0]['index'], single_input.shape)
                    tflite_interpreter.allocate_tensors()
                    tflite_interpreter.set_tensor(input_details[0]['index'], single_input)
                    tflite_interpreter.invoke()
                    output = tflite_interpreter.get_tensor(output_details[0]['index'])
                    batch_preds.append(output[0])

                batch_preds = np.array(batch_preds)
                predicted_chunks.append(np.argmax(batch_preds, axis=1))
                confidence_chunks.append(np.max(batch_preds, axis=1))
        else:
            for i in range(0, total_rows, PREDICT_BATCH_SIZE):
                batch_df = processed.iloc[i:i + PREDICT_BATCH_SIZE]
                scaled_batch = scaler.transform(batch_df).astype(np.float32)
                x_batch = scaled_batch.reshape(scaled_batch.shape[0], scaled_batch.shape[1], 1)
                batch_preds = model.predict(x_batch, verbose=0)
                predicted_chunks.append(np.argmax(batch_preds, axis=1))
                confidence_chunks.append(np.max(batch_preds, axis=1))

        predicted_classes = np.concatenate(predicted_chunks)
        confidence_scores = np.concatenate(confidence_chunks)

        n_benign = int((predicted_classes == 0).sum())
        n_attack = int((predicted_classes == 1).sum())
        attack_pct = round((n_attack / total_rows) * 100, 2) if total_rows else 0
        avg_confidence = round(float(confidence_scores.mean()) * 100, 2)
        min_confidence = round(float(confidence_scores.min()) * 100, 2)
        max_confidence = round(float(confidence_scores.max()) * 100, 2)

        risk_level, risk_msg = compute_risk(attack_pct)

        conf_histogram = {
            "90-100%": int(((confidence_scores >= 0.9) & (confidence_scores <= 1.0)).sum()),
            "80-90%":  int(((confidence_scores >= 0.8) & (confidence_scores < 0.9)).sum()),
            "70-80%":  int(((confidence_scores >= 0.7) & (confidence_scores < 0.8)).sum()),
            "60-70%":  int(((confidence_scores >= 0.6) & (confidence_scores < 0.7)).sum()),
            "<60%":    int((confidence_scores < 0.6).sum()),
        }

        preferred_cols = [
            "Source IP", "Destination IP", "Timestamp", "Protocol",
            "Destination Port", "Flow Duration", "Total Fwd Packets",
            "Total Backward Packets", "Flow Bytes/s", "Flow Packets/s",
            "Fwd Packet Length Mean", "Bwd Packet Length Mean",
            "Flow IAT Mean", "Fwd IAT Mean",
        ]
        display_columns = [c for c in preferred_cols if c in original_df.columns]
        if not display_columns:
            display_columns = list(original_df.columns[:4])
        display_columns = display_columns[:4]

        limit = min(MAX_DISPLAY_ROWS, total_rows)
        results_rows = []
        for i in range(limit):
            row = {}
            for col in display_columns:
                val = original_df.iloc[i][col]
                if isinstance(val, float):
                    row[col] = round(val, 2) if abs(val) < 1e6 else f"{val:.2e}"
                else:
                    row[col] = str(val)
            row["Label"] = LABEL_MAP[int(predicted_classes[i])]
            row["Confidence"] = round(float(confidence_scores[i]), 4)
            results_rows.append(row)

        download_limit = min(MAX_DOWNLOAD_ROWS, total_rows)
        download_slice = original_df.iloc[:download_limit]
        download_data = {}
        for col in display_columns:
            download_data[col] = download_slice[col].tolist()
        download_data["Prediction"] = [LABEL_MAP[int(c)] for c in predicted_classes[:download_limit]]
        download_data["Confidence"] = [round(float(s), 4) for s in confidence_scores[:download_limit]]
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
            "download_rows": download_limit,
            "csv_download": download_df.to_csv(index=False),
        }

    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
