# NIDS — Project Explainer

A full walkthrough of the project: what it is, what every file does, what was
added to finish the backend, and how the whole thing fits together. Read this and
you'll understand the project end to end.

---

## 1. What the project is

A **Network Intrusion Detection System (NIDS)**. It takes network "flows"
(connections, each described by 78 numbers) and uses a deep learning model to say
whether a flow is **normal (BENIGN)** or one of **10 attack types**
(DDoS, PortScan, DoS Hulk, SSH-Patator, Botnet, Web Attacks, etc.).

Three parts:
1. **Model** — a 1D CNN trained on the CICIDS2017 dataset (already trained).
2. **Backend** — a FastAPI server that runs the model and explains its predictions with SHAP.
3. **Frontend** — a React dashboard with a live alert feed, a world map, and a
   "why was this flagged" panel.

---

## 2. The model and its files

The model expects **78 numeric features per flow** and outputs probabilities over
**11 classes** (BENIGN + 10 attacks). We take the highest probability as the
prediction and its value as the confidence.

**Architecture:** `Input(78,1) → Conv1D(128) → Conv1D(256) → Flatten → Dense(256)+Dropout → Dense(11, softmax)`.
A 1D CNN is used because flow features are a sequence of measurements (not a 2D
image), and convolutions let the model learn combinations of features by itself.

**Training pipeline** lives in [scripts/preprocessing.py](scripts/preprocessing.py)
and the notebook. Key steps and why:
- Merge the 5 daily CSVs, drop ID columns (Flow ID, IPs, Timestamp) so the model
  can't "cheat" off identifiers.
- Replace inf/NaN (rate features like `Flow Bytes/s` divide by zero) with safe values.
- Clip outliers at the 99th percentile.
- **MinMax scale** every feature to [0,1] and **save the scaler** — at prediction
  time we must apply the exact same scaling.
- Balance classes (CICIDS is ~80% benign; without balancing the model would just
  always guess "benign").
- Reshape to `(N, 78, 1)` for the CNN and save `X_dcnn.npy`.

**Saved artifacts:**

| File | What it is |
|------|-----------|
| `models/new/nids_model.h5` | the trained Keras model |
| `models/new/cicids_scaler.pkl` | the fitted MinMaxScaler (also gives the feature order) |
| `models/new/class_labels.json` | the 11 label names + feature/class counts |
| `data/processed/X_dcnn.npy` | scaled training rows, reused as the SHAP "background" |

> **Important detail:** training used `df.columns.difference(['Label'])`, which
> returns columns **sorted alphabetically**. So the scaler (and therefore
> `FEATURE_NAMES`) is in alphabetical order. Any CSV we load must be **reordered
> to that same order** before scaling, or every feature gets scaled wrong.

---

## 3. The backend, file by file

Everything lives under `scripts/`. The app is built in
[scripts/api.py](scripts/api.py) and run with `uvicorn scripts.api:app`.

### Entry point
- **[scripts/api.py](scripts/api.py)** — creates the FastAPI app, sets up logging,
  CORS, a request logger, and an error handler, loads the model at startup
  (lifespan), and registers all the routers.

### Config & schemas
- **[scripts/app/config.py](scripts/app/config.py)** — all settings (file paths,
  CORS origins, stream rate, upload limits). Each has a default but can be
  overridden with an `NIDS_`-prefixed environment variable. Reads env vars with
  plain `os.getenv` so it has no extra dependencies.
- **[scripts/app/schemas.py](scripts/app/schemas.py)** — the Pydantic models that
  define request/response shapes (`PredictRequest`, `PredictResponse`, `Alert`,
  `AnalyzeResponse`, …). FastAPI uses these for validation and the auto docs.

### Services (the logic)
- **[scripts/app/services/model.py](scripts/app/services/model.py)** — loads the
  artifacts. Splits loading in two: the light metadata (scaler, labels, feature
  names, SHAP background) loads immediately and needs **no TensorFlow**; the heavy
  Keras model + SHAP explainer load **lazily** via `load_models()`. Also has
  `vectorize()` (turn a request payload into an ordered, validated array) and
  `get_model()` / `get_explainer()`.
- **[scripts/app/services/prediction_service.py](scripts/app/services/prediction_service.py)**
  — the actual prediction logic. `infer_fast` (scale → predict → argmax),
  `infer_explained` (same + SHAP), and the request wrappers `predict_fast`,
  `predict_explained`, `predict_batch`.
- **[scripts/app/services/flow_to_alert.py](scripts/app/services/flow_to_alert.py)**
  — converts a flow + metadata into the `Alert` object the frontend shows (adds
  severity and a geo lookup).
- **[scripts/app/services/label_map.py](scripts/app/services/label_map.py)** —
  maps an attack label to a severity (critical/high/medium/low).
- **[scripts/app/services/geoip.py](scripts/app/services/geoip.py)** — looks up an
  IP's location using a MaxMind database if one is configured; otherwise returns
  `None`. Caches results and skips private IPs.
- **[scripts/app/services/csv_loader.py](scripts/app/services/csv_loader.py)** —
  parses an uploaded CSV: validates the feature columns, reorders them, turns
  inf/NaN into 0, caps the row count. `synth_meta()` makes up IP/protocol for a
  row (the CSV doesn't contain them).
- **[scripts/app/services/demo_stream.py](scripts/app/services/demo_stream.py)** —
  for the live demo: samples real CICIDS rows at startup and `next_demo_alert()`
  builds one alert. Falls back to the scaled background rows if the raw CSVs
  aren't present.

### Routers (the HTTP/WS endpoints)
- **[scripts/app/routers/health.py](scripts/app/routers/health.py)** — `GET /health`, `GET /features`.
- **[scripts/app/routers/predict.py](scripts/app/routers/predict.py)** — `POST /predict`, `/predict/explain`, `/predict/batch`.
- **[scripts/app/routers/stream.py](scripts/app/routers/stream.py)** — `WS /ws/alerts` live stream.
- **[scripts/app/routers/analyze.py](scripts/app/routers/analyze.py)** — `POST /api/analyze` (CSV upload → alerts).

### Endpoint summary

| Method | Path | What it does |
|--------|------|--------------|
| GET | `/health` | status + model metadata |
| GET | `/features` | the 78 feature names, in order |
| POST | `/predict` | one flow → class + confidence |
| POST | `/predict/explain` | one flow → class + SHAP contributions |
| POST | `/predict/batch` | CSV upload or JSON list → prediction per flow |
| POST | `/api/analyze` | CSV upload → list of Alerts + a summary |
| WS | `/ws/alerts` | one alert per second from real CICIDS rows |

---

## 4. The frontend files (the ones connected to the backend)

- **[frontend/src/lib/api.ts](frontend/src/lib/api.ts)** — the only file that
  knows about the backend: `connectAlertStream()` (live websocket),
  `analyzeFile()` (upload a CSV), and a small mapper that converts a backend alert
  into the UI's `Alert` type.
- **[frontend/src/pages/Index.tsx](frontend/src/pages/Index.tsx)** — the dashboard
  page. Subscribes to the websocket for the live feed and merges uploaded results
  into the same feed.
- **[frontend/src/components/dashboard/UploadModal.tsx](frontend/src/components/dashboard/UploadModal.tsx)**
  — calls `/api/analyze` and shows a summary.
- **[frontend/src/components/dashboard/ExplanationDrawer.tsx](frontend/src/components/dashboard/ExplanationDrawer.tsx)**
  — shows the SHAP bars; uses real values when the alert has them.

---

## 5. What was changed/added to finish the backend

This is the changelog of the work done in this session.

**Fixed**
- `prediction_service.py` had a half-written `predict_batch` causing a syntax
  error (the app wouldn't start). Finished it.

**New features**
- `POST /api/analyze` — upload a CSV, get back analyzed alerts + a summary.
  Added [routers/analyze.py](scripts/app/routers/analyze.py), the `synth_meta`
  helper and column-reordering in [csv_loader.py](scripts/app/services/csv_loader.py),
  and `AnalyzeResponse`/`AnalyzeSummary` in [schemas.py](scripts/app/schemas.py).
- Real **GeoIP** in [geoip.py](scripts/app/services/geoip.py) (MaxMind lookup,
  caching, skips private IPs, gracefully off when no DB).

**Refactors / robustness**
- **Model loading split in two** in [model.py](scripts/app/services/model.py): light
  metadata at import (no TensorFlow), heavy model loaded once at startup. This is
  what lets the tests and most of the code import without TensorFlow installed.
- **Env-driven config** in [config.py](scripts/app/config.py) replacing the hard
  coded constants (old constant names kept so nothing breaks).
- **Startup lifespan + logging + error handling** in [api.py](scripts/api.py):
  loads the model once at startup, logs each request, returns clean JSON on errors,
  and reads CORS origins from config.
- **Configurable stream rate** in [stream.py](scripts/app/routers/stream.py).
- **Demo stream fallback** so it still works without the raw CSVs.

**Tests** (in [tests/](tests/), plain `unittest`)
- `test_label_map.py`, `test_vectorize.py`, `test_csv_loader.py`, `test_schemas.py`
  — run without TensorFlow.
- `test_api.py` — full `/predict` + websocket tests that skip automatically if
  TensorFlow/httpx aren't installed.

**Deploy / docs**
- [Dockerfile](Dockerfile) (multi-stage), [docker-compose.yml](docker-compose.yml),
  [.dockerignore](.dockerignore), [requirements-dev.txt](requirements-dev.txt).
- Updated [README.md](README.md) with the endpoint table + curl examples, and this file.

**Frontend wiring**
- Added [api.ts](frontend/src/lib/api.ts); connected the live feed
  ([Index.tsx](frontend/src/pages/Index.tsx)), the upload
  ([UploadModal.tsx](frontend/src/components/dashboard/UploadModal.tsx)), and real
  SHAP in the drawer ([ExplanationDrawer.tsx](frontend/src/components/dashboard/ExplanationDrawer.tsx)).

---

## 6. How a request flows

**`POST /predict`**
1. FastAPI validates the body against `PredictRequest`.
2. `vectorize()` builds an ordered array of 78 values (errors if any are missing).
3. `_to_model_input()` scales with the saved scaler and reshapes to `(1,78,1)`.
4. The model predicts; `argmax` gives the class and confidence.
5. `/predict/explain` additionally runs SHAP and returns per-feature contributions.

**`POST /api/analyze`**
- `load_csv()` validates + cleans the file → for each row, `synth_meta()` +
  `flow_to_alert()` build an Alert → returns the list + a summary. SHAP is capped
  to keep it fast.

**`WS /ws/alerts`**
- Every second, `next_demo_alert()` picks a real row, builds an Alert, and sends it.

---

## 7. Running and testing

```bash
# backend
pip install -r requirements.txt
uvicorn scripts.api:app --reload --port 8000   # docs at /docs

# frontend
cd frontend && npm install && npm run dev       # or bun

# tests (no TensorFlow needed for the logic tests)
python -m unittest discover -s tests

# docker
docker compose up --build
```

---

## 8. Known limitations (and honest answers)

- **Batch predict loops one flow at a time** instead of predicting all rows in one
  call. Fine for demo-sized files; the obvious next optimization.
- **No PCAP support yet** — only CSVs. PCAP would need a flow extractor (CICFlowMeter).
- **Uploaded flows have synthetic IPs** because the CICIDS CSVs don't include
  IP/protocol columns. The *prediction* is real; the IPs shown are for display.
- The model only works well on traffic similar to the CICIDS2017 distribution.

---

## 9. Quick interview answers

- **Why scale at prediction time?** The model was trained on [0,1] inputs, so we
  must apply the same saved scaler to new data.
- **Why a 1D CNN?** Flow features are a 1D sequence; convolutions learn feature
  combinations automatically.
- **What does SHAP do?** It assigns each feature a contribution to a single
  prediction, so we can show *why* a flow was flagged.
- **Why load the model lazily?** Keeps imports fast and TensorFlow-free (so tests
  run anywhere) and moves the heavy load into a single startup step.
- **How to handle class imbalance?** Undersample the majority class during training.
