# 13 · Cheat Sheets

> One-glance review. Read this the night before. Everything here is expanded elsewhere in the book.

---

## 🎯 The pitch (memorize)
Real-time NIDS. **1D-CNN** classifies **CICFlowMeter flows** (**78 features**, CICIDS2017) into **11 classes**
(BENIGN + 10 attack families) at **99.48% accuracy / 97.67% macro-F1**. **FastAPI** backend: fast predict,
SHAP-explained predict, batch CSV, **WebSocket** live stream. **React** dashboard. **Stateless** (no DB,
no auth), multi-stage **Docker**, deployed on **HF Spaces + Vercel**.

---

## 🏛️ Architecture cheat sheet
```
Browser (React, api.ts) ──REST/WS──> nginx (same-origin proxy)
   └─> FastAPI (api.py: lifespan + CORS + timing log + global 500 handler)
         ├─ routers/  (transport)   health · predict · analyze · stream(WS)
         ├─ services/ (business/ML) prediction_service · flow_to_alert · csv_loader · demo_stream · geoip · label_map
         └─ model.py  (singletons)  scaler · Keras model · SHAP explainer · FEATURE_NAMES · LABELS
               └─ artifacts: nids_model.h5 · cicids_scaler.pkl · class_labels.json · demo_flows.npy
```
**Dependency rule:** routers → services → model. Nothing in services imports routers.
**Patterns:** singleton, lazy-init + eager warm-up, settings object, DTO/schemas, mapper, composition root.
**No:** database, repository layer, auth, LLM/RAG/vector DB, Redis, queues.

---

## 🔌 API cheat sheet
| Method | Path | In | Out | SHAP |
|---|---|---|---|---|
| GET | `/health` | — | status, n_features=78, n_classes=11, labels, bg size | — |
| GET | `/features` | — | 78 feature names (model order) | — |
| POST | `/predict` | `{features: dict|list[78], top_k}` | class, index, confidence | no |
| POST | `/predict/explain` | same | + probabilities{11}, top/all contributions | yes |
| POST | `/predict/batch` | CSV upload | `{flow_result: [...]}` | no |
| POST | `/api/analyze` | CSV + `include_shap` | `{alerts[], summary}` | ≤50 |
| WS | `/ws/alerts` | — | `Alert` every 1/STREAM_RATE_HZ s | no |

**Validation:** Pydantic (422) → `vectorize` count/names (400) → `load_csv` columns/rows/inf (400) → global (500 generic).

---

## 🧠 ML cheat sheet
```
Conv1D(128,k3,same)+BN+MaxPool → Conv1D(256)+BN+MaxPool → Conv1D(256)+BN
→ Flatten → Dense(512)+Drop0.3 → Dense(256)+Drop0.2 → Dense(11,softmax)
Adam(1e-4) · sparse_categorical_crossentropy · 30 epochs · batch 256 · seed 42
EarlyStopping(val_loss,patience5,restore_best) · ReduceLROnPlateau(0.5,patience3)
Split: 80/20 test, then 10% val · stratified
Input (batch,78,1)
```
**Preprocessing:** merge days → strip cols → drop Flow ID/Src IP/Dst IP/Timestamp → merge label variants →
drop rare(<100) → inf/NaN drop → clip 99th pct → MinMax[0,1] → LabelEncode → undersample(cap 50k) → reshape.
**Serving transform:** raw → `scaler.transform` → reshape(1,78,1) → predict → argmax. (`already_scaled` skips scaling.)

---

## 📊 Metrics cheat sheet
| | Value |
|---|---|
| Accuracy | 99.48% |
| Macro F1 | 97.67% |
| Weighted F1 | 99.47% |
| Macro Precision / Recall | 98.29% / 97.20% |
| Test set | 47,912 flows |
| **Weakest:** Botnet | recall **77.75%** (mimics benign) |
| 2nd weakest: Web Attacks | F1 93.44% |
**Quote macro-F1, not just accuracy.** Imbalance hides minority performance.

---

## 🏷️ Classes & severity
`BENIGN(low) · DDoS/DoS GoldenEye/DoS Hulk(critical) · DoS Slowhttptest/DoS slowloris/FTP-Patator/SSH-Patator(high) · Botnet/PortScan/Web Attacks(medium)`
Unknown label → `low` (safe default). Severity is policy (`label_map.py`), not a model output → change without retraining.

---

## 🔍 SHAP cheat sheet
- **GradientExplainer** (gradient-based, cheap for NNs) vs KernelSHAP (slow, model-agnostic).
- **Background** = 100 rows; fallback **real X_dcnn → demo_flows.npy → synthetic**.
- Output: signed per-feature contribution vs baseline; sort by **|value|**; return `top_k`.
- Handles list-per-class **or** stacked ndarray (version-robust).
- Explains the **model**, not causation; not "hallucination" (no LLM). Risk = overconfident OOD → add confidence floor.

---

## 🔐 Security cheat sheet
| Have | Don't have |
|---|---|
| CORS allow-list, strong input validation, upload caps, generic 500, non-root container, TLS at edge | auth, RBAC, API keys, rate limiting, secrets (none needed) |
- CORS ≠ auth (browser-only; curl bypasses).
- N/A: SQL/NoSQL injection (no DB), prompt injection (no LLM).
- Biggest risk: **unauthenticated SHAP DoS**.
- Hardening order: **auth → rate limit → timeouts/pool → stream-parse uploads → confidence floor → audit log → headers → dep scan.**
- ML-specific threat: **adversarial evasion / model extraction.**

---

## ⚡ Performance & scale cheat sheet
- Cost: **SHAP ≫ forward pass**; `predict_batch` not vectorized (top quick win).
- In code: singleton+warm-up, `compile=False`, lazy TF import, fast/explain split, SHAP cap, row cap, geoip LRU, reservoir sampling, async WS, static cache.
- Concurrency: WS async; sync routes run in threadpool (blocking predict is safe). Sharp edge: WS loop's blocking predict → threadpool it.
- Scale: **stateless → clone behind LB → queue+Redis (decouple SHAP) → GPU/ONNX + dynamic batch + pub/sub WS + TSDB + observability.**
- 100→1container · 10k→replicas+LB(+batch fix) · 100k→queue+Redis+autoscale · 1M→GPU fleet+CDN+bus+TSDB.

---

## 🗄️ "Database" cheat sheet (there isn't one)
- Persistent state = **artifact files** (model/scaler/labels/demo). Scaler's `feature_names_in_` **is the schema**.
- Runtime state = module singletons + geoip LRU; per-process, rebuilt on restart.
- "Migration" = swap the **artifact trio atomically** (`old/` vs `new/` = versions).
- If you add a DB → **Postgres** (alerts append-only, partition by time, index ts/type/severity/src_ip; Timescale/Clickhouse at scale). Schema: api_key→run→alert→shap_contrib/feedback, model_version.

---

## 🚀 Deployment cheat sheet
- **Local:** `docker compose up --build` → frontend :8080 (nginx) proxies to backend (no host port). Same-origin.
- **Hosted:** backend → HF Spaces (Docker, 16GB RAM, `app_port 8000`, honors `$PORT`); frontend → Vercel (`VITE_API_BASE` = Space URL). Set `NIDS_CORS_ORIGINS` = Vercel URL.
- **Dockerfile:** multi-stage; non-root uid 1000; caches → `/tmp`; copies only serving files; excludes 149MB dataset.
- No CI/CD in repo (manual/auto-push). Would add: test → build → push, gated on green + eval-gate.

---

## 🐛 Debug cheat sheet
| Symptom | First suspect |
|---|---|
| wrong/all-BENIGN predictions | feature **order** (list input) / scaling / mismatched artifact trio |
| explain slow/timeout | SHAP cost; shrink background; raise proxy timeout |
| 400 on upload | read message: missing columns / too many rows / empty / parse |
| CORS error in browser | origin not in `NIDS_CORS_ORIGINS` (curl won't reveal it) |
| WS flapping | proxy Upgrade headers / read timeout / backend restart |
| container dies on boot | artifact load failed or cache dir not writable (need /tmp caches) |
| empty map | no GeoLite2 DB (expected); frontend synthesizes a point |
**Start every diagnosis at the request-log line:** `METHOD path -> status (ms)`.

---

## 🔧 Key files cheat sheet
| File | Why it matters |
|---|---|
| `scripts/api.py` | composition root: lifespan, middleware, routers |
| `scripts/app/config.py` | all `NIDS_*` tunables + paths |
| `scripts/app/schemas.py` | the API contract |
| `scripts/app/services/model.py` | singletons, fallbacks, `vectorize`, the schema (FEATURE_NAMES) |
| `scripts/app/services/prediction_service.py` | `infer_fast` / `infer_explained` — the inference core |
| `scripts/app/services/flow_to_alert.py` | prediction → Alert mapper |
| `scripts/app/services/csv_loader.py` | upload validation + sanitization |
| `scripts/app/services/demo_stream.py` | reservoir sampling + WS data source + fallbacks |
| `scripts/app/services/label_map.py` | label → severity policy |
| `scripts/preprocessing.py` | offline data pipeline |
| `scripts/NIDS_new_training.ipynb` | model definition + training |
| `scripts/EVAL_REPORT.md` | the metrics you quote |
| `Dockerfile` / `docker-compose.yml` / `frontend/nginx.conf` | how it ships |

---

## 🧩 Numbers to never forget
**78** features · **11** classes · **99.48%** acc · **97.67%** macro-F1 · **47,912** test flows ·
**Botnet 77.75%** recall · input **(b,78,1)** · **100**-row SHAP background · **5000** row cap ·
**50** SHAP cap · **1 Hz** stream · **Adam 1e-4** · **30** epochs · batch **256** · drop **Flow ID/IPs/Timestamp**.
