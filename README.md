<div align="center">

![Network Intrusion Detection System](https://img.shields.io/badge/NIDS-Deep%20Learning-blue?style=for-the-badge&logo=tensorflow)
![Python](https://img.shields.io/badge/Python-3.8%2B-green?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen?style=for-the-badge)

# Network Intrusion Detection System

**A Deep Convolutional Neural Network approach to detecting malicious network traffic**

[Getting Started](#getting-started) | [Documentation](#usage) | [Research Background](#research-background) | [Contributing](#contributing)

</div>

---

## Overview

This project implements a network intrusion detection system using a one-dimensional Deep Convolutional Neural Network (DCNN). The model is trained on the CICIDS2017 dataset to perform **multi-class classification** across 11 categories — `BENIGN` plus 10 attack families (DDoS, PortScan, Botnet, the DoS variants, the Patators, and Web Attacks).

The approach draws from recent advances in deep learning for cybersecurity applications, where convolutional architectures have shown remarkable success in extracting spatial features from network flow data.

**Key Features:**
- 11-class classifier (BENIGN + 10 attack families) over 78 CICIDS flow features
- 1D Convolutional Neural Network architecture
- FastAPI backend with SHAP explanations and a live WebSocket alert stream
- React/Vite real-time dashboard (alert feed, threat map, per-flow explanations)
- One-command full-stack deployment via Docker Compose
- TensorFlow Lite export for resource-constrained environments

---

## Research Background

This implementation is inspired by the growing body of research applying deep learning to network intrusion detection. The architectural decisions and preprocessing pipeline are informed by several key publications in the field:

### Foundational Work

**Deep Learning for Network Intrusion Detection:**
The use of convolutional neural networks for intrusion detection follows the approach outlined by researchers who demonstrated that CNNs can effectively learn hierarchical feature representations from raw network traffic data. Unlike traditional machine learning methods that rely heavily on manual feature engineering, deep learning models can automatically extract relevant patterns.

Key references that informed this implementation:

1. **Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018).** "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization." *Proceedings of the 4th International Conference on Information Systems Security and Privacy (ICISSP)*, pp. 108-116.
   - This paper introduces the CICIDS2017 dataset used in this project, which addresses limitations of older datasets like KDD Cup 99 and NSL-KDD.

2. **Kim, J., Kim, J., Thu, H. L. T., & Kim, H. (2016).** "Long Short Term Memory Recurrent Neural Network Classifier for Intrusion Detection." *International Conference on Platform Technology and Service (PlatCon)*, pp. 1-5. IEEE.
   - Demonstrates the effectiveness of deep learning architectures for network-based intrusion detection.

3. **Vinayakumar, R., Alazab, M., Soman, K. P., Poornachandran, P., Al-Nemrat, A., & Venkatraman, S. (2019).** "Deep Learning Approach for Intelligent Intrusion Detection System." *IEEE Access*, vol. 7, pp. 41525-41550.
   - Comprehensive study comparing various deep learning architectures including CNNs for intrusion detection, establishing benchmarks on CICIDS2017.

4. **Zhang, Y., Chen, X., Jin, L., Wang, X., & Guo, D. (2019).** "Network Intrusion Detection: Based on Deep Hierarchical Network and Original Flow Data." *IEEE Access*, vol. 7, pp. 37004-37016.
   - Explores hierarchical deep network structures for feature extraction from network flows.

### Architecture Rationale

The model architecture follows the Conv1D approach for sequential network flow features:

- **1D Convolutions** are chosen over 2D because network flow features represent sequential measurements rather than spatial image data
- **Two convolutional layers** (128 and 256 filters) progressively extract higher-level feature abstractions
- **L1/L2 regularization and dropout** prevent overfitting on the training data
- **Softmax output with categorical cross-entropy** aligns with standard multi-class classification practices in the literature

### Dataset: CICIDS2017

The Canadian Institute for Cybersecurity Intrusion Detection System 2017 (CICIDS2017) dataset was selected because:

- Contains realistic modern network traffic captured over 5 days
- Includes both benign traffic and common attack types (DDoS, PortScan, Web Attacks, Infiltration, etc.)
- Provides 78 network flow features extracted using CICFlowMeter
- Addresses class imbalance through proper labeling and sampling strategies
- Widely adopted as a benchmark in recent intrusion detection research

---

## Project Structure

```
nids/
├── models/
│   └── new/
│       ├── nids_model.h5         # Trained 11-class Keras model
│       ├── nids_model.tflite     # TensorFlow Lite version
│       ├── cicids_scaler.pkl     # MinMaxScaler for feature normalization
│       └── class_labels.json     # Index → label mapping
├── scripts/
│   ├── api.py                    # FastAPI entrypoint (uvicorn scripts.api:app)
│   ├── app/
│   │   ├── config.py             # Env-driven settings (NIDS_* overrides)
│   │   ├── schemas.py            # Pydantic request/response models
│   │   ├── routers/              # health, predict, analyze, stream (WS)
│   │   └── services/             # model, SHAP, csv_loader, demo_stream, geoip
│   ├── convert_to_tflite.py      # Model conversion utility
│   └── preprocessing.py          # Data preprocessing utilities
├── frontend/                     # React + Vite dashboard (nginx-served in prod)
│   ├── src/lib/api.ts            # Backend client (REST + WebSocket)
│   ├── Dockerfile                # Build → nginx image
│   └── nginx.conf                # Static serve + reverse proxy to backend
├── tests/                        # Backend pytest suite
├── Dockerfile                    # Backend container
├── docker-compose.yml            # Full stack: backend + frontend
└── requirements.txt              # Python dependencies
```

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/nids.git
cd nids
pip install -r requirements.txt
```

### Running the Application

**Option 1: Full stack with Docker Compose (Recommended)**

Builds the backend API and the nginx-served React dashboard, wired together on
one network:

```bash
cp .env.example .env        # optional — sane defaults work out of the box
docker compose up --build
```

Then open the dashboard at **http://localhost:8080**. The frontend's nginx
reverse-proxies `/api`, `/ws`, `/health`, etc. to the backend, so there is no
CORS or URL configuration to do.

**Option 2: Run the two services locally for development**

```bash
# Terminal 1 — backend (hot reload)
uvicorn scripts.api:app --reload --port 8000

# Terminal 2 — frontend (Vite dev server on :8080)
cd frontend
echo "VITE_API_BASE=http://localhost:8000" > .env   # point the UI at the API
npm install && npm run dev
```

Interactive API docs are always available at `http://localhost:8000/docs`.

**Option 3: Backend container only**

```bash
docker build -t nids-backend .
docker run -p 8000:8000 nids-backend
```

---

## Usage

### Web Interface

The web interface provides a straightforward way to analyze network traffic:

1. Navigate to `http://localhost:8000` in your browser
2. Upload a CSV file containing network flow data
3. The system automatically preprocesses the data and runs inference
4. View prediction results with confidence scores
5. Download the results as a CSV file

### REST API

Run the backend from the project root:

```bash
uvicorn scripts.api:app --reload --port 8000
# interactive docs at http://localhost:8000/docs
```

**Endpoints**

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness + model metadata (feature/class counts) |
| GET | `/features` | Ordered list of the 78 feature names |
| POST | `/predict` | One flow → predicted class + confidence |
| POST | `/predict/explain` | One flow → prediction + SHAP feature contributions |
| POST | `/predict/batch` | JSON list of flows **or** a CSV upload → per-flow predictions |
| POST | `/api/analyze` | CSV upload → list of `Alert` objects + summary |
| WS | `/ws/alerts` | Live demo stream: one `Alert`/sec from real CICIDS rows |

**Examples**

```bash
# Health
curl http://localhost:8000/health

# Single prediction (78 values, in feature order — see GET /features)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.0, 0.0, /* …78 total… */ 0.0]}'

# Prediction with SHAP explanation
curl -X POST http://localhost:8000/predict/explain \
  -H "Content-Type: application/json" \
  -d '{"features": {"Flow Duration": 1200, "Total Fwd Packets": 5, "...": 0}}'

# Analyze an uploaded CSV (CICIDS feature columns)
curl -X POST http://localhost:8000/api/analyze \
  -F "csv=@test.csv" -F "include_shap=true"
```

**Configuration** — every setting has a default and an `NIDS_`-prefixed env var
override, e.g. `NIDS_CORS_ORIGINS`, `NIDS_STREAM_RATE_HZ`, `NIDS_GEOIP_DB_PATH`,
`NIDS_MAX_UPLOAD_ROWS` (see `scripts/app/config.py`).

> For a full walkthrough of how every component works and why, see
> [`explain.md`](explain.md).

### Python Integration

For direct integration into your Python applications:

```python
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import joblib

# Load the trained model and scaler
model = load_model('models/nids_dcnn_model.h5')
scaler = joblib.load('models/cicids_scaler.pkl')

# Load and preprocess your data
df = pd.read_csv('network_traffic.csv')
df = df.drop(columns=['Flow ID', 'Source IP', 'Destination IP', 
                       'Timestamp', 'Label'], errors='ignore')

# Handle missing values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# Scale and reshape for Conv1D input
X_scaled = scaler.transform(df)
X_input = X_scaled.reshape(-1, X_scaled.shape[1], 1)

# Run inference
predictions = model.predict(X_input)
predicted_classes = np.argmax(predictions, axis=1)

# Interpret results
benign_count = (predicted_classes == 0).sum()
attack_count = (predicted_classes == 1).sum()
print(f"Benign flows: {benign_count}")
print(f"Attack flows: {attack_count}")
```

---

## Input Data Format

The model expects network flow data matching the CICIDS2017 feature set.

**Automatically Removed Columns:**
- Flow ID
- Source IP
- Destination IP
- Timestamp
- Label (if present)

**Required Features (78 total):**

The remaining 78 features are standard network flow metrics including:
- Flow duration and byte/packet counts
- Forward and backward packet statistics
- Inter-arrival times (IAT) for flow, forward, and backward directions
- Flag counts (PSH, URG, FIN, SYN, etc.)
- Header lengths and segment sizes
- Active/idle time statistics

For a complete feature list, refer to the CICFlowMeter documentation or examine the training data columns.

---

## Model Architecture

The DCNN architecture is designed to extract meaningful patterns from sequential network flow features:

| Layer | Configuration | Output Shape |
|-------|--------------|--------------|
| Input | - | (samples, 78, 1) |
| Conv1D | 128 filters, kernel size 3, ReLU | (samples, 76, 128) |
| Conv1D | 256 filters, kernel size 3, ReLU | (samples, 74, 256) |
| Flatten | - | (samples, 18944) |
| Dense | 256 units, ReLU, L1/L2 regularization | (samples, 256) |
| Dropout | 0.1 | (samples, 256) |
| Dense | 11 units, Softmax | (samples, 11) |

**Training Configuration:**
- Optimizer: Adam (learning rate: 0.0001)
- Loss function: Sparse categorical cross-entropy
- Batch size: 256
- Epochs: 30
- Train/test split: 80/20 with stratification

---

## Performance

Evaluation on the CICIDS2017 test set yields the following results:

| Metric | Score |
|--------|-------|
| Accuracy | 98-99% |
| Precision (BENIGN) | High |
| Precision (ATTACK) | High |
| Recall (ATTACK) | High |

Performance may vary depending on the similarity between your data and the training distribution. The model performs best on traffic patterns that resemble those captured in the CICIDS2017 dataset.

---

## Deployment

The stack is two stateless services — a FastAPI/TensorFlow backend and a static
React bundle — so it deploys cleanly anywhere.

- **Local / demo:** `docker compose up --build` → dashboard at
  `http://localhost:8080`. Zero config; best for a presentation or viva.
- **Free public link:** host the **backend on Hugging Face Spaces** (Docker, free
  16 GB RAM, WebSocket support) and the **frontend on Vercel**. Step-by-step
  instructions are in **[DEPLOYMENT.md](DEPLOYMENT.md)**.

The repo is pre-configured for this: the backend honors `$PORT`, synthesizes its
SHAP background when the large `X_dcnn.npy` isn't present, and the frontend's API
base is set with the `VITE_API_BASE` build variable.

If you later need production scale, the same images run on any container platform
(Cloud Run, Fly.io, Railway) or Kubernetes — just terminate TLS, set
`NIDS_CORS_ORIGINS` to your exact frontend origin, and add auth in front of the
unauthenticated endpoints.

---

## Troubleshooting

**Feature Mismatch Error**

This occurs when your input CSV has a different number of columns than expected. Ensure your data contains the same 78 features used during training. Column names are case-sensitive.

**Model Not Found Error**

Verify that the following files exist in the `models/` directory:
- `nids_dcnn_model.h5` (or `nids_dcnn_model.tflite`)
- `cicids_scaler.pkl`

**NaN or Infinity Values**

The preprocessing pipeline automatically handles these by replacing infinity values with NaN and filling NaN with zeros. For more accurate results, consider cleaning your data before submission.

---

## Contributing

Contributions are welcome. If you would like to improve the project, please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with appropriate tests
4. Submit a pull request

Areas where contributions would be particularly valuable:
- Multi-class attack categorization
- Real-time packet capture integration
- Performance optimization for edge deployment
- Additional preprocessing options


---

## Acknowledgments

This project builds upon the work of many researchers and open-source contributors:

- **Canadian Institute for Cybersecurity** for creating and maintaining the CICIDS2017 dataset
- **TensorFlow and Keras teams** for the deep learning framework
- **FastAPI** and **React/Vite** for the application stack
- The broader research community working on network security and machine learning

---




## License

This project is released under the MIT License. See the LICENSE file for details.

---


<div align="center">

**Last Updated:** March 2026

[Report Bug](https://github.com/yourusername/nids/issues) | [Request Feature](https://github.com/yourusername/nids/issues)

</div>
