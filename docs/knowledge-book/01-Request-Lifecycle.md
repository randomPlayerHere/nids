# 01 · Request Lifecycle

> The interviewer's favorite question is "walk me through what happens when…". This chapter is
> the answer for every important path. Learn the **generic skeleton** first, then the per-feature
> specifics.

---

## The generic skeleton (every REST request)

The template you were given lists: *User → Frontend → API Route → Middleware → Auth → Controller →
Service → Repository → Database → AI Service → Response → Frontend.*

Here is how this project maps onto it — including the layers that **don't exist** (say this out loud
in an interview, it shows you know what you skipped and why):

```mermaid
flowchart LR
    A["User action\n(click / upload)"] --> B["Frontend\nsrc/lib/api.ts (fetch/WS)"]
    B --> C["nginx reverse proxy\n(prod only)"]
    C --> D["CORS middleware"]
    D --> E["Request-timing logger\n(@app.middleware http)"]
    E --> F["Router / path operation\n(routers/*.py)"]
    F --> G["Pydantic validation\n(schemas.py)"]
    G --> H["Service layer\n(services/*.py)"]
    H --> I["model.py singletons\nscaler · model · explainer"]
    I --> J["NumPy / TF / SHAP math"]
    J --> K["Pydantic response model"]
    K --> L["JSON → back through middleware → client"]
    L --> M["Frontend normalizes & renders"]
```

| Template layer | In this project? | What plays its role |
|----------------|------------------|---------------------|
| Frontend request | ✅ | `frontend/src/lib/api.ts` — thin `fetch`/`WebSocket` client |
| API Route | ✅ | FastAPI `APIRouter`s in `scripts/app/routers/` |
| Middleware | ✅ | CORS + a custom request-timing logger + a global exception handler ([api.py](../../scripts/api.py)) |
| **Authentication** | ❌ | **None.** Open API. See [07-Security.md](07-Security.md) for why & what you'd add |
| **Authorization** | ❌ | **None.** No roles, no per-resource checks |
| Controller | ✅ (merged) | FastAPI "path operation functions" *are* the controllers; they're thin and delegate immediately |
| Service | ✅ | `prediction_service`, `csv_loader`, `demo_stream`, `flow_to_alert` |
| **Repository** | ❌ | **None.** No DB rows to fetch. `model.py` is the closest analog: it "loads" artifacts once |
| **Database** | ❌ | **None.** Persistence = static model/scaler/label files. See [04](04-Data-and-Model-Store.md) |
| AI Service | ✅ | The Keras model + SHAP explainer, invoked through `prediction_service` |
| Response | ✅ | Pydantic models serialized to JSON (or WS frame) |

> **Why no controller/service/repository split into three?** With no database, a repository layer
> would be an empty pass-through — a classic **over-engineering anti-pattern**. The honest design
> is **router → service → model**. Three layers, each earning its keep.

---

## Cross-cutting concerns (run on *every* request)

These live in [scripts/api.py](../../scripts/api.py) and wrap all routes:

1. **CORS middleware** — rejects browser calls from origins not in `settings.CORS_ORIGINS`. Methods/headers are `*`; origins are an explicit allow-list (default includes `localhost:8080/5173`). In prod you add your Vercel URL via `NIDS_CORS_ORIGINS`.
2. **Request-timing logger** — `@app.middleware("http")` records `method path -> status (X.Xms)` for every request. This is your cheapest observability.
3. **Global exception handler** — `@app.exception_handler(Exception)` catches anything uncaught, logs the stack trace, and returns a generic `500 {"error": "Internal server error"}` so internal details never leak to the client.
4. **Startup lifespan** — *not per-request*, but it's why requests are fast: `load_models()` runs once before the server accepts traffic, so the first real request doesn't eat the multi-second TF load.

```mermaid
sequenceDiagram
    participant U as Uvicorn boot
    participant L as lifespan()
    participant M as model.py
    Note over U,M: Happens ONCE, before serving
    U->>L: enter lifespan
    L->>M: load_models()
    M->>M: tf.keras.load_model(h5)
    M->>M: shap.GradientExplainer(model, background)
    M->>M: model.predict(warm-up row)
    M-->>L: ready (11 classes, 78 features)
    L-->>U: yield → server now accepts requests
```

---

## Feature 1 — Fast prediction (`POST /predict`)

**Use case:** a machine client (or the dashboard) wants the label for one flow, as fast as possible.

```mermaid
sequenceDiagram
    participant C as Client
    participant R as predict.py
    participant V as model.vectorize()
    participant P as prediction_service.predict_fast
    participant S as scaler (MinMax)
    participant K as Keras model

    C->>R: POST /predict {features, top_k}
    R->>R: Pydantic validates PredictRequest
    R->>P: predict_fast(req)
    P->>V: vectorize(req.features)
    V-->>P: np.float32[78]  (validates length/missing keys)
    P->>S: scaler.transform(raw.reshape(1,78))
    S-->>P: scaled[1,78]
    P->>K: model.predict(scaled.reshape(1,78,1))
    K-->>P: probs[11]
    P->>P: argmax → label, index, confidence
    P-->>R: PredictResponseFast
    R-->>C: 200 JSON {predicted_class, predicted_index, confidence}
```

**Key points to narrate:**
- Input may be a **dict** (`{"Flow Duration": ..., ...}`) *or* a **list** of 78 floats. `vectorize()` handles both; dicts get reordered to `FEATURE_NAMES`.
- Scaling happens **server-side** (`already_scaled=False`) — the client sends raw flow values, not pre-scaled ones. This keeps the client dumb and the contract honest.
- No SHAP → one forward pass → low latency.

---

## Feature 2 — Explained prediction (`POST /predict/explain`)

Same as above until after the forward pass, then it **also** runs SHAP:

```mermaid
sequenceDiagram
    participant C as Client
    participant P as predict_explained
    participant K as Keras model
    participant E as SHAP GradientExplainer

    C->>P: POST /predict/explain
    P->>K: model.predict(x) → probs[11]
    P->>P: pred_idx = argmax(probs)
    P->>E: explainer.shap_values(x)
    E-->>P: shap values per class
    P->>P: pick sv for pred_idx → 78 contributions
    P->>P: build all_contributions, sort by |value|, take top_k
    P-->>C: 200 {class, confidence, probabilities{11}, top_contributions, all_contributions}
```

**Why two endpoints instead of a flag?** Clearer contract (`PredictResponseFast` vs `PredictResponse`),
self-documenting in `/docs`, and the fast path never accidentally pays the SHAP cost. (A `?explain=true`
flag would also work — that's a legitimate alternative to mention.)

---

## Feature 3 — Batch CSV analysis (`POST /api/analyze`)

**Use case:** analyst uploads a CSV of many flows for offline triage; wants alerts + a summary.

```mermaid
sequenceDiagram
    participant C as Browser (Upload Mode)
    participant R as analyze.py
    participant L as csv_loader.load_csv
    participant F as flow_to_alert
    participant P as prediction_service
    participant G as geoip

    C->>R: POST /api/analyze (multipart: csv, include_shap)
    R->>L: load_csv(csv)
    L->>L: read_csv, strip cols, check 78 present, reorder,\n coerce numeric, inf/NaN→0, cap rows
    L-->>R: list[dict] rows
    loop each row
        R->>R: synth_meta(row)  # fake src/dst IP, protocol
        R->>F: flow_to_alert(features, meta, include_shap?)
        Note over R,F: SHAP only for first ANALYZE_SHAP_CAP rows
        F->>P: infer_fast OR infer_explained
        P-->>F: label, confidence, (contributions)
        F->>G: geoip.lookup(src_ip)  # only if not BENIGN
        G-->>F: geo or None
        F-->>R: Alert
    end
    R->>R: build AnalyzeSummary (total/benign/malicious/by_class)
    R-->>C: 200 {alerts[], summary}
```

**Narrate the safeguards:** row cap (`MAX_UPLOAD_ROWS=5000`), SHAP cap (`ANALYZE_SHAP_CAP=50`),
inf/NaN sanitization, and the fact that IPs/protocol are **synthesized** because the CICIDS feature
CSV doesn't carry them (they were dropped during preprocessing).

---

## Feature 4 — Live alert stream (`GET /ws/alerts`, WebSocket)

```mermaid
sequenceDiagram
    participant C as Browser
    participant W as stream.py (WS)
    participant D as demo_stream.next_demo_alert
    participant F as flow_to_alert
    participant K as model

    C->>W: WS handshake /ws/alerts
    W->>C: accept
    loop forever (until disconnect)
        W->>D: next_demo_alert()
        D->>D: random.choice(all_flows)  # sampled real flows
        D->>F: flow_to_alert(features, synth meta, no SHAP, already_scaled?)
        F->>K: infer_fast → label, confidence
        F-->>D: Alert
        D-->>W: Alert
        W->>C: send_json(alert)
        W->>W: await asyncio.sleep(1/STREAM_RATE_HZ)
    end
    Note over C,W: client auto-reconnects with exp backoff on close
```

**The client side** ([api.ts](../../frontend/src/lib/api.ts) `connectAlertStream`): opens the socket,
on each message `normalizeAlert(JSON.parse(...))` (converts timestamp → `Date`, fills geo fallback),
and on close reconnects with `min(1000 * 2**retry, 15000)` ms backoff unless the caller closed it.

---

## Response lifecycle (the way back out)

1. Service returns a **Pydantic model** instance (e.g. `PredictResponse`).
2. FastAPI serializes it to JSON using the route's `response_model` — this also **filters** fields, so even if the object had extras, only declared fields ship.
3. The timing middleware logs `... -> 200 (X.Xms)` on the way out.
4. nginx (prod) forwards the bytes to the browser.
5. Frontend `api.ts` normalizes (e.g. `normalizeAlert`) and React renders.

> **WHAT IF something throws mid-service?** It bubbles to the global exception handler → logged with
> stack trace → client gets `500 {"error": "Internal server error"}`. Validation errors (wrong length,
> missing feature) are raised as `HTTPException(400, ...)` *inside* the service, so the client gets a
> specific 400 with a helpful message instead of a generic 500.
