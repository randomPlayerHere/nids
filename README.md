# 🛡️ Network Intrusion Detection System (NIDS) using DCNN

A Deep Convolutional Neural Network (DCNN) for detecting network intrusions using the CICIDS2017 dataset.

## 📋 Project Overview

This project implements a binary classification system that identifies malicious network traffic:
- **Architecture**: 1D Convolutional Neural Network (Conv1D)
- **Dataset**: CICIDS2017
- **Classes**: BENIGN vs ATTACK
- **Framework**: TensorFlow/Keras

## 🚨 Issues Fixed

### Original Problems:
1. ❌ **Missing preprocessor file** (`data/processed/preprocessor.pkl`)
2. ❌ **Inconsistent preprocessing** across different files
3. ❌ **Typo in filename** (`stramlit_app.py` → `streamlit_app.py`)
4. ❌ **Missing dependencies** (streamlit, imbalanced-learn)
5. ⚠️ **No feature validation** during inference

### Solutions Applied:
- ✅ Updated Streamlit app to use consistent preprocessing (matching main.py)
- ✅ Added feature count validation
- ✅ Improved error handling and user feedback
- ✅ Added NaN and infinity handling
- ✅ Updated requirements.txt
- ✅ Fixed filename typo

## 📁 Project Structure

```
nids/
├── models/
│   ├── nids_dcnn_model.h5      # Trained DCNN model
│   └── cicids_scaler.pkl        # MinMaxScaler for features
├── data/
│   ├── raw/                     # Original CICIDS2017 CSV files
│   └── processed/               # Preprocessed numpy arrays
├── frontend/                    # HTML/JS frontend for FastAPI
├── notebooks/                   # Jupyter notebooks for training
├── main.py                      # FastAPI backend server
├── streamlit_app.py             # Streamlit web interface (FIXED ✅)
├── inference_example.py         # Example inference script
├── preprocessing.py             # Data preprocessing utilities
└── requirements.txt             # Python dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit App (Recommended)

```bash
streamlit run streamlit_app.py
```

Then open your browser to `http://localhost:8501`

### 3. Alternative: Run FastAPI Backend

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Access the API at `http://localhost:8000/docs`

## 🎯 Usage

### Using Streamlit (Easiest)

1. Launch the app: `streamlit run streamlit_app.py`
2. Upload a CSV file with network traffic data
3. View predictions, confidence scores, and visualizations
4. Download results as CSV

### Using FastAPI

Send POST request to `/predict`:

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@path/to/your/data.csv"
```

### Using Python Script

```python
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import joblib

# Load assets
model = load_model('models/nids_dcnn_model.h5')
scaler = joblib.load('models/cicids_scaler.pkl')

# Load your data
df = pd.read_csv('your_data.csv')

# Drop identifier columns
df = df.drop(columns=['Flow ID', 'Source IP', 'Destination IP', 'Timestamp', 'Label'], errors='ignore')

# Scale and reshape
X_scaled = scaler.transform(df)
X_input = X_scaled.reshape(-1, X_scaled.shape[1], 1)

# Predict
predictions = model.predict(X_input)
predicted_classes = np.argmax(predictions, axis=1)

print(f"Benign: {(predicted_classes == 0).sum()}")
print(f"Attack: {(predicted_classes == 1).sum()}")
```

## 📊 Input Data Format

Your CSV must contain the same features as the training data (CICIDS2017):

**Features automatically dropped:**
- Flow ID
- Source IP
- Destination IP
- Timestamp
- Label (if present)

**Expected features (78 total):**
- Duration
- Protocol
- Flow Bytes/s
- Flow Packets/s
- Flow IAT Mean
- Fwd IAT Mean
- Bwd IAT Mean
- ...and 71 more CICIDS2017 features

## ⚙️ Model Details

**Architecture:**
- Input: (samples, 78, 1) - reshaped feature vectors
- Conv1D layers with 1D pooling
- Dense layers with dropout
- Output: Softmax with 2 classes

**Training:**
- Dataset: CICIDS2017
- Preprocessing: MinMaxScaler
- Class balancing: Applied during training
- Optimizer: Adam

## 🔧 Troubleshooting

### "Feature Mismatch" Error
- Ensure your CSV has the exact same columns as training data
- Remove any extra columns or add missing ones
- Check that column names match exactly (case-sensitive)

### "Model not found" Error
- Verify `models/nids_dcnn_model.h5` exists
- Verify `models/cicids_scaler.pkl` exists
- Check file permissions

### NaN or Infinity Values
- The app automatically handles these by filling with 0
- For better results, clean your data beforehand

## 📈 Expected Performance

Based on CICIDS2017 test set:
- **Accuracy**: ~98-99%
- **Precision**: High for both classes
- **Recall**: High for attack detection

Note: Performance depends on data quality and similarity to training data.

## 🤝 Contributing

Improvements made:
1. Fixed preprocessing inconsistencies
2. Added comprehensive error handling
3. Improved user interface with visualizations
4. Added feature validation
5. Better documentation

## 📝 License

This is an educational project based on DCNN for intrusion detection.

## 🙏 Acknowledgments

- CICIDS2017 Dataset by Canadian Institute for Cybersecurity
- TensorFlow/Keras framework
- Streamlit for the web interface

---

**Last Updated**: March 2026
**Status**: Production Ready ✅
