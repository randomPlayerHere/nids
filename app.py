from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import io
import os

# ---------------------------
# Lifespan Event (Modern FastAPI way)
# ---------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.model = load_model("models/nids_dcnn_model.h5")
        app.state.scaler = joblib.load("models/cicids_scaler.pkl")
        print("✅ Model and Scaler loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load assets: {e}")
        app.state.model = None
        app.state.scaler = None
    yield

app = FastAPI(
    title="NIDS Prediction API",
    version="1.0.0",
    lifespan=lifespan
)

# ---------------------------
# Prediction Endpoint
# ---------------------------
@app.post("/predict")
async def predict_network_traffic(file: UploadFile = File(...)):
    
    # 1️⃣ Check model availability
    if app.state.model is None or app.state.scaler is None:
        raise HTTPException(status_code=500, detail="Model or scaler not loaded.")

    # 2️⃣ Validate file type
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        df = pd.read_csv(io.BytesIO(contents))

        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file contains no data.")

        # 3️⃣ Drop non-feature columns safely
        columns_to_drop = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp', 'Label']
        df_features = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

        # 4️⃣ Ensure feature alignment (CRITICAL for ML models)
        expected_features = app.state.scaler.feature_names_in_

        missing_features = set(expected_features) - set(df_features.columns)
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required features: {missing_features}"
            )

        df_features = df_features[expected_features]

        # 5️⃣ Scale
        scaled_features = app.state.scaler.transform(df_features)

        # 6️⃣ Reshape for Conv1D
        X_input = scaled_features.reshape(
            scaled_features.shape[0],
            scaled_features.shape[1],
            1
        )

        # 7️⃣ Predict
        predictions = app.state.model.predict(X_input, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)

        # 8️⃣ Map labels
        label_mapping = {
            0: "BENIGN",
            1: "🚨 ATTACK DETECTED"
        }

        human_readable_preds = [
            label_mapping.get(val, "UNKNOWN")
            for val in predicted_classes
        ]

        return {
            "num_samples": len(human_readable_preds),
            "predictions": human_readable_preds
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Feature mismatch error: {str(ve)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    