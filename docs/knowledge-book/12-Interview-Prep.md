# 12 · Interview Preparation — 150+ Questions

> Drill these. Format: **Q → Excellent Answer → Why it's right → Common Mistake → Follow-ups.**
> Grouped by topic. The first time, cover the answer and try to produce it yourself.

Legend for brevity: **CM** = Common Mistake, **FU** = Follow-up(s).

---

## A. Project & Architecture (1–20)

**1. What is this project in one sentence?**
A real-time network intrusion detection system: a 1D-CNN classifies CICFlowMeter flows into 11 classes (BENIGN + 10 attack families) at 99.48% accuracy, served by a FastAPI backend with SHAP explanations and a WebSocket alert stream, consumed by a React dashboard.
*Why:* hits problem, model, metric, interface. **CM:** rambling without the metric or the "flow" unit. **FU:** what's a flow? what's CICFlowMeter?

**2. Walk me through the architecture.**
Layered FastAPI: routers (transport) → services (business/ML) → `model.py` singletons (scaler/model/explainer) → TF/SHAP. Stateless, no DB. Frontend talks REST + WebSocket through nginx (same-origin).
*Why:* names layers + the stateless property. **CM:** describing files instead of layers. **FU:** why no DB? why singletons?

**3. Why FastAPI?** *(see [09](09-Design-Decisions.md))* Async WebSockets, Pydantic validation, OpenAPI docs, Python-native for TF. **CM:** "it's fast" with no specifics. **FU:** Flask vs FastAPI? async vs sync routes?

**4. Why no database?** Pure inference function of input + read-only model; nothing to persist; a DB would be accidental complexity. I designed the schema for when persistence is required. **CM:** treating it as a gap to apologize for. **FU:** when would you add one? what schema?

**5. What are the main components?** Routers (health/predict/analyze/stream), services (prediction_service, model, flow_to_alert, csv_loader, demo_stream, geoip, label_map), artifacts (h5/scaler/labels/demo). **FU:** which service is reused most? (`prediction_service` — 4 entry points).

**6. What's the unit of analysis?** A network **flow** — a summarized conversation between endpoints, 78 numeric features from CICFlowMeter. **CM:** saying "a packet." **FU:** packet vs flow detection tradeoffs?

**7. Signature-based vs ML-based NIDS?** Signature = human rules, great for known attacks, brittle to novel ones; ML = learns statistical patterns, generalizes better, but is a black box (hence SHAP) and can be evaded. **FU:** which is this? can you combine them?

**8. Why is statelessness important?** Replicas are interchangeable → horizontal scaling by cloning containers behind an LB; no shared-state coordination. **FU:** what breaks statelessness if you add features?

**9. How is the model loaded so requests are fast?** `lifespan` calls `load_models()` once at startup (load + warm-up predict), kept as module singletons. **CM:** "loaded per request." **FU:** what does warm-up do?

**10. What's the request lifecycle for `/predict`?** validate (Pydantic) → `vectorize` → scale → reshape (1,78,1) → predict → argmax → response model → JSON. **FU:** where does scaling happen, client or server?

**11. Why separate routers from services?** Transport vs business separation → the inference core is reused by 4 routers and unit-testable without HTTP. **CM:** calling it boilerplate. **FU:** show a function tested without a server.

**12. Is there a repository layer?** No — no DB means a repository would be an empty pass-through; `model.py` is the closest analog (loads read-only artifacts once). **FU:** is that an anti-pattern? (avoiding it is the right call.)

**13. What design patterns are used?** Process singleton (model), lazy init + eager warm-up, Settings/config object, DTO/schemas, mapper (flow_to_alert), composition root (api.py). **FU:** is it dependency injection?

**14. Frontend → backend data flow?** `src/lib/api.ts` does `fetch` (REST) and `WebSocket`; nginx proxies same-origin; responses normalized (`normalizeAlert`) and rendered. **FU:** how does the WS reconnect?

**15. What are the 5 core features?** fast predict, explained predict, batch CSV, WebSocket stream, introspection (health/features). **FU:** which is the demo experience? (the stream.)

**16. Two "worlds" in the repo?** Offline training world (preprocessing, notebook, tflite) produces artifacts; online serving world (`scripts/app`) consumes them; they meet only at the saved files. **FU:** how do you version the handoff?

**17. Biggest engineering challenges?** Heavy model load, slow SHAP, messy input, feature-order correctness, clean-clone deploy, real-time delivery — each with a concrete mitigation. **FU:** pick one and go deep.

**18. Why nginx?** Serves SPA + reverse-proxies API/WS → same-origin (kills CORS), handles WS `Upgrade`, caches assets, sets upload/timeout. **FU:** what would break without same-origin?

**19. How is config managed?** 12-factor: `NIDS_*` env vars via a typed `Settings` object with defaults; one image, env-driven behavior. **FU:** what's secret? (nothing today.)

**20. What's the test strategy?** unittest + FastAPI `TestClient`; pure functions (vectorize, load_csv, to_severity, schemas) tested without TF; the full-stack test loads the model via lifespan and skips if TF/httpx absent. **FU:** what's missing? (no perf/load tests, limited integration.)

---

## B. API Design (21–38)

**21. Why two predict endpoints?** Cost + contract isolation: `/predict` (fast) vs `/predict/explain` (SHAP); response shapes differ and the fast path never pays SHAP. **CM:** "they're the same." **FU:** would a `?explain` flag be better?

**22. How do you validate input?** Three gates: Pydantic (types/shape→422), `vectorize` (count/names→400), `load_csv` (columns/rows/numeric/inf→400). **FU:** what gets to TF? (only clean vectors.)

**23. Dict vs list input — how handled?** `vectorize` reorders dicts to `FEATURE_NAMES`; lists are length-checked only. **CM:** missing that **lists aren't order-validated** (a footgun). **FU:** how would you harden list input?

**24. What happens on 77 features?** `HTTPException(400, "Expected 78 feature values, got 77")`. **FU:** on a missing dict key? (400 listing first 5.)

**25. What's in the explain response?** class, index, confidence, full 11-class probabilities, top_k contributions (sorted by |value|), all 78 contributions. **FU:** why sort by absolute value?

**26. `/predict/batch` vs `/api/analyze`?** batch = raw fast predictions list; analyze = rich `Alert`s + summary + optional capped SHAP + synthetic meta + geo. **FU:** why two? (machine vs dashboard consumers.)

**27. Why cap SHAP in analyze?** SHAP is the dominant cost; `ANALYZE_SHAP_CAP=50` bounds tail latency while still explaining the first flows. **FU:** rows capped too? (yes, `MAX_UPLOAD_ROWS`.)

**28. How does the WebSocket work?** accept → loop: sample flow → classify → `send_json` → `asyncio.sleep(1/rate)`; disconnect ends it; client reconnects with backoff. **FU:** SSE instead? (valid — it's push-only.)

**29. Error contract?** 400 = your input wrong (specific msg); 422 = Pydantic shape; 500 = generic (details only in logs). **FU:** why generic 500? (don't leak internals.)

**30. How are responses shaped/filtered?** `response_model=` per route both validates and drops undeclared fields. **FU:** could internal fields leak? (no — filtered.)

**31. Where are OpenAPI docs?** `/docs` (Swagger), auto-generated from type hints + Pydantic. **FU:** value of that? (self-documenting, client testing.)

**32. Is the batch endpoint efficient?** No — loops per-row; batching into one tensor is the #1 optimization. **CM:** claiming it's optimized. **FU:** how to fix?

**33. How do you bound upload size?** `MAX_UPLOAD_ROWS=5000` + nginx `client_max_body_size 64m`; caveat: pandas buffers the file before the row check. **FU:** how to harden? (stream-parse/`nrows`.)

**34. Why camelCase in `Alert`?** Shaped to match the TS frontend (`srcIP`, `flowDuration`) to minimize client remapping. **FU:** how does the client still normalize? (Date + geo fallback.)

**35. How does the client handle WS drops?** exponential backoff `min(1000*2**retry, 15000)`, no reconnect if caller closed. **FU:** thundering herd at scale?

**36. Idempotency?** All endpoints are read-only/pure → naturally idempotent; no writes to coordinate. **FU:** changes if you add persistence? (alert insert isn't idempotent.)

**37. Versioning the API?** Not versioned today; I'd add `/v1` prefix before breaking changes. **FU:** how to evolve schemas safely? (additive fields, response_model.)

**38. Why `include_shap` as a form field, not query?** It's a multipart upload (file + flag), so the flag rides in the form. **FU:** default? (False — cheap by default.)

---

## C. Machine Learning / Model (39–66)

**39. Describe the model architecture.** 3×Conv1D (128,256,256, k=3) with BatchNorm + MaxPool, Flatten, Dense(512,0.3 dropout)→Dense(256,0.2)→Dense(11 softmax); Adam 1e-4; sparse categorical cross-entropy. **FU:** why each layer?

**40. Why 1D and not 2D convolution?** Flows are 1D ordered feature vectors, not images; 1D kernels slide over the feature axis. **CM:** treating tabular as an image. **FU:** does feature order matter for conv? (neighbors should be related; CICFlowMeter groups them.)

**41. Why a CNN over XGBoost for tabular?** Honest: to learn local feature interactions and follow the CICIDS DL literature, but XGBoost is often equal/cheaper — I'd benchmark it. **CM:** claiming CNN is obviously best. **FU:** how would the serving change if XGBoost won? (only `model.py`.)

**42. What's sparse categorical cross-entropy and why?** Multi-class loss with **integer** labels (no one-hot) → less memory; pairs with softmax. **FU:** vs categorical CCE? (one-hot targets.)

**43. Why Adam at 1e-4?** Adaptive, robust default; small LR for stable convergence; `ReduceLROnPlateau` lowers it further on plateaus. **FU:** SGD alternative tradeoffs?

**44. What does BatchNorm do here?** Normalizes layer activations → faster/more stable training + mild regularization. **FU:** train vs inference behavior? (uses running stats at inference.)

**45. Role of dropout?** Randomly zeroes activations in the dense head (0.3/0.2) to reduce overfitting. **FU:** active at inference? (no.)

**46. How handle class imbalance?** RandomUnderSampler caps majority classes at 50k; monitor macro-F1 + per-class recall. **FU:** alternatives? (class weights, focal loss, SMOTE.)

**47. Accuracy is 99.48% — is that suspicious?** It's plausible: stratified held-out test (47,912), EarlyStopping, dropped IP/timestamp to prevent leakage. But I quote **macro-F1 (97.67%)** because accuracy hides minority performance. **CM:** quoting only accuracy. **FU:** weakest class?

**48. Which class is weakest and why?** **Botnet (recall 77.75%)** — botnet flows statistically mimic benign traffic; 87/391 misclassed as BENIGN. **FU:** how to fix? (class weights/focal loss/more data.)

**49. Macro vs weighted F1?** Macro = unweighted class average (exposes minorities); weighted = support-weighted (≈ accuracy). Gap reveals imbalance impact. **FU:** which matters for security? (macro/recall — missing attacks is costly.)

**50. Precision vs recall for a NIDS — which matters more?** Usually **recall** on attack classes (a missed intrusion is worse than a false alarm), but too-low precision causes alert fatigue. It's a tunable threshold tradeoff. **FU:** how would you tune it? (per-class thresholds, cost-sensitive.)

**51. Why MinMaxScaler?** Bounds inputs to [0,1] for the NN; with 99th-pct clipping it tames heavy tails; the fitted scaler is reused at serve time → no train/serve skew. **FU:** StandardScaler tradeoff?

**52. What is train/serve skew and how avoided?** Different preprocessing in training vs serving → wrong inputs. Avoided by persisting and reusing the *exact* fitted scaler (`cicids_scaler.pkl`) and pinning feature order via `feature_names_in_`. **FU:** how could it still happen? (mismatched artifact trio.)

**53. Why clip at the 99th percentile?** Flow stats are heavy-tailed; extreme outliers distort MinMax scaling and destabilize training. **FU:** why not drop outliers? (clipping keeps the row's other signal.)

**54. Why drop Source IP/Timestamp before training?** They're identifiers, not behavioral signal; keeping them invites memorization/leakage and hurts generalization. **Consequence:** the API synthesizes IPs for display. **FU:** any leakage risk left?

**55. What's the input tensor shape and why?** `(batch, 78, 1)` — Conv1D needs `(steps, channels)`; one channel. **FU:** what's the trailing 1?

**56. How does inference scale a single request?** `scaler.transform(raw.reshape(1,78))` → `reshape(1,78,1)` → `predict` → argmax. **FU:** the `already_scaled` flag?

**57. What's `already_scaled` for?** Demo flows from `demo_flows.npy` are pre-scaled; the flag skips re-scaling to avoid distortion. **CM:** not knowing it → double-scaling bug. **FU:** what if set wrong?

**58. How would you detect model drift in prod?** Monitor the prediction-class distribution and confidence over time vs a baseline; alert on shift; sample + relabel for retraining. **FU:** needs persistence — which schema? (alerts + model_version.)

**59. How would you retrain/deploy a new model?** Re-run preprocessing + notebook → export the trio (h5/scaler/labels) → swap atomically → eval gate blocks regression. **FU:** why atomic? (scaler/labels must match.)

**60. What's the TFLite export for?** Edge/sensor deployment; quantization (fp16/int8/dynamic) shrinks the model 2–4×. Not used by the API server. **FU:** int8 needs what? (a representative calibration dataset.)

**61. Quantization types?** fp16 (half weights, tiny accuracy hit), int8 (full integer, fastest edge, needs calibration), dynamic (int8 weights, float activations at runtime). **FU:** accuracy/size tradeoff?

**62. Is the model overfit?** EarlyStopping on val loss + dropout + BatchNorm + held-out test all argue no; the small train/test metric gap (implied by 99%+ test) supports it. **FU:** how would you prove it? (learning curves, CV.)

**63. Why stratified splits?** Preserve class ratios across train/val/test so minority classes are represented and metrics are meaningful. **FU:** without stratify? (a rare class could vanish from a split.)

**64. Could you use an autoencoder/anomaly model instead?** Yes for *unknown* attacks (train on benign, flag reconstruction error). This project is supervised multi-class (known families). A hybrid (classifier + anomaly score) is the best of both. **FU:** when does anomaly win? (zero-day.)

**65. What features matter most?** Per-prediction, SHAP tells you; globally, flow duration, packet/byte rates, IAT stats, and TCP-flag counts dominate attack signatures. **FU:** how to get global importance? (aggregate SHAP / permutation importance.)

**66. What's the inference latency profile?** Fast predict = a few ms (one forward pass post-warm-up); explain = 10–100× due to SHAP. **FU:** how to speed explain? (smaller background, cache, batch.)

---

## D. SHAP / Explainability (67–78)

**67. What is SHAP?** Shapley-value-based attribution: signed per-feature contribution to a prediction relative to a background baseline, with consistency guarantees. **FU:** why for security? (analysts must justify alerts.)

**68. Why GradientExplainer specifically?** Designed for differentiable models; uses gradients (expected/integrated gradients) → far cheaper than KernelSHAP's many perturbations. **FU:** KernelSHAP tradeoff? (model-agnostic but slow.)

**69. What's the SHAP background and why does it matter?** A reference sample (100 rows) the explainer compares against; its quality affects attribution fidelity. **FU:** fallback chain? (real → demo_flows → synthetic.)

**70. Why sort contributions by absolute value?** The most *influential* features matter regardless of push direction; top_k surfaces them. **FU:** does sign matter? (yes — toward/away from the class.)

**71. How is SHAP's variable output shape handled?** `infer_explained` handles list-per-class vs stacked-ndarray (`[...,class]`) and trims to 78 — defensive against SHAP versions. **FU:** why defensive? (version drift.)

**72. SHAP vs LIME?** SHAP has theoretical consistency/additivity; LIME fits a local surrogate (faster but less stable). **FU:** which here? (SHAP.)

**73. Is SHAP causal?** No — it explains the *model's* behavior, not real-world causation. **CM:** claiming causal. **FU:** implication for analysts? (interpret as model reasoning.)

**74. How do you make explanations cheaper at scale?** Cap count (done), shrink background, cache by input hash, or precompute for common patterns. **FU:** acceptable accuracy loss?

**75. Could you explain a wrong prediction with SHAP?** Yes — SHAP shows which features drove the (wrong) call, which is exactly how you debug misclassifications. **FU:** Botnet→BENIGN, what would you look at?

**76. Hallucination prevention — relevant here?** No — that's an LLM concept. The analog is overconfident OOD predictions; mitigate with a confidence floor / anomaly score. **CM:** answering as if there's an LLM. **FU:** how add a guard?

**77. What does `raw_input` in a contribution mean?** The unscaled feature value shown alongside its SHAP value so analysts see the actual measurement. **FU:** why show raw not scaled? (human-readable.)

**78. Limitations of SHAP here?** Cost, background-dependence, correlated-feature attribution ambiguity, not causal. **FU:** how would correlated features mislead? (credit split arbitrarily.)

---

## E. Data Engineering & Preprocessing (79–90)

**79. Walk through the preprocessing pipeline.** merge days → strip cols → drop IDs → merge label variants → drop rare(<100) → inf/NaN→drop → clip 99th pct → MinMax → label-encode → undersample 50k → reshape → save. **FU:** why that order?

**80. Why does order matter?** Clip before scale (so outliers don't blow up MinMax); scale before undersample; fit scaler on the train distribution; reshape last. **FU:** what if you scaled before clipping? (outliers dominate range.)

**81. How are inf/NaN handled in training vs serving?** Training: drop rows. Serving (`load_csv`): replace with 0.0 (can't drop a user's row silently per-request without telling them; 0 maps to feature min post-scale). **FU:** better serving choice? (median impute / reject.)

**82. Why merge label variants (Web Attack *, Bot)?** Collapse sparse sub-labels into coherent families with enough samples to learn/evaluate. **FU:** downside? (lose sub-type granularity.)

**83. Why drop classes < 100 samples?** Too few to train or evaluate reliably; they'd add noise and unstable metrics. **FU:** alternative? (few-shot/anomaly handling.)

**84. Explain reservoir sampling in `demo_stream`.** One-pass uniform k-sample without knowing length: fill reservoir, then for item i replace a random slot with prob k/i. O(1) extra memory. **FU:** prove uniformity? (induction on k/i.)

**85. Why reservoir sampling here?** Raw CICIDS CSVs are huge; can't load fully to sample. **FU:** seeded — why? (reproducible demo.)

**86. How is feature order guaranteed end-to-end?** `FEATURE_NAMES = scaler.feature_names_in_`; dict inputs and CSV both reorder to it; boot asserts `len==N_FEATURES`. **FU:** what it doesn't catch? (a list in wrong order; a scaler fit on different order.)

**87. Why `CAP_PER_CLASS=50_000`?** Enough to learn majority classes well while shrinking the dominant ones so gradients aren't swamped; also speeds training. **FU:** why not balance fully? (would discard too much; minorities stay small anyway.)

**88. How big is the training data and why isn't it committed?** Processed `X_dcnn.npy` is ~149 MB → too big for git; the app falls back to `demo_flows.npy` (~150 KB) for SHAP background + stream. **FU:** deploy impact? (clean-clone works.)

**89. What's `LabelEncoder` doing and where's the inverse at serve time?** Maps class names→0..10 for training; at serve time `class_labels.json` (`index_to_label`) maps predicted index back to name. **FU:** mismatch risk? (ship labels with model.)

**90. How would you build a streaming feature pipeline for true real-time?** Tap packets → CICFlowMeter (or a flow exporter) → feature vector → scale → model → alert; buffer/window flows; backpressure. **FU:** where's the latency? (flow completion + feature extraction.)

---

## F. Concurrency, Performance, Scalability (91–110)

**91. Sync vs async routes here?** WS handler is async (cheap idle sockets); predict routes are sync `def` → run in FastAPI's threadpool so blocking `predict` doesn't stall the loop. **CM:** thinking all routes must be async. **FU:** what if you made predict `async def` with blocking predict? (blocks the loop.)

**92. Is the model thread-safe?** Yes — read-only weights; `predict` is safe concurrently; singleton is immutable after load. **FU:** the load race? (lifespan loads once, single-threaded.)

**93. Where's the perf bottleneck?** SHAP (`/explain`, batch) and the unbatched `predict_batch` loop. **FU:** fix order? (vectorize batch first.)

**94. How do you scale to 1M users?** stateless → clone behind LB → queue+Redis to decouple SHAP/batch → GPU/ONNX + dynamic batching + pub/sub WS fan-out + TSDB for history + observability. **FU:** first stateful piece to hurt? (the history DB.)

**95. What caching exists?** geoip LRU (10k), nginx static asset cache, model-in-memory singleton. No result cache yet. **FU:** what to cache next? (identical-flow results.)

**96. Would Redis help and where?** Rate limiting, per-key quotas, a result cache, and WS pub/sub fan-out. Not needed at current scale. **FU:** result cache key? (hash of the 78-vector.)

**97. What's the WS scaling concern?** Sticky sessions / connection-aware LB; the blocking predict in the loop; fan-out across nodes needs pub/sub. **FU:** threadpool fix?

**98. How does horizontal scaling "just work" here?** No shared mutable state → replicas are interchangeable behind round-robin. **FU:** exception? (WS connections are stateful per node.)

**99. How would you add backpressure?** Queue the slow path (analyze/SHAP), return job ids, cap queue depth, shed/429 when full. **FU:** UX impact? (async polling.)

**100. p99 latency strategy?** Separate fast vs slow paths, cap SHAP, autoscale on queue depth, timeouts + circuit breakers, dynamic batching. **FU:** what metric to alert on?

**101. Memory risks?** CSV buffered before row-cap; large background; model RAM. **FU:** mitigation? (stream-parse, smaller background.)

**102. Why warm up the model at startup?** First predict triggers TF graph tracing/XLA; doing it pre-traffic avoids a slow first user request. **FU:** measurable? (first-request latency drop.)

**103. Compression?** Not enabled; add GZipMiddleware/nginx gzip for large analyze JSON. **FU:** CPU tradeoff? (small.)

**104. How would dynamic batching help?** Coalesce concurrent single-flow requests into one tensor → higher GPU/CPU throughput. **FU:** latency cost? (small batching window.)

**105. Connection pooling — relevant?** Not now (no DB). Becomes relevant when you add Postgres (use an async pool). **FU:** pool size guidance?

**106. Observability today vs needed?** Today: request-timing logs, healthcheck. Needed: Prometheus metrics, OTel traces, structured logs, drift monitor. **FU:** key SLO?

**107. Circuit breakers — where?** Around the model/queue/DB calls once those exist, to fail fast and shed load. **FU:** half-open behavior?

**108. How to load-test this?** Locust/k6 against `/predict` and `/api/analyze`; measure p50/p99, throughput, error rate; separate fast vs explain. **FU:** what would saturate first?

**109. Cold start on HF Spaces?** Free Spaces sleep after ~48h idle; first hit wakes (~30s, includes model load). **FU:** mitigation for a demo? (warm it before presenting.)

**110. Why `tensorflow-cpu` not full TF?** No GPU on the target host; the CPU build is lighter to install and deploy. **FU:** when switch to GPU build? (latency/throughput at scale.)

---

## G. Security (111–124)

**111. Is the API authenticated?** No — open API; for a demo it's fine, but auth/rate limiting are my top production additions. **CM:** pretending it's secure. **FU:** how add auth minimally?

**112. Does CORS secure the API?** No — browser-enforced only; `curl` bypasses it. Not authentication. **CM:** conflating CORS with auth. **FU:** what is CORS for then?

**113. SQL injection risk?** None — no DB/SQL; inputs never reach a query engine. If added: parameterized queries/ORM. **FU:** NoSQL injection? (also N/A.)

**114. Prompt injection?** N/A — no LLM/prompts; numeric vectors to a CNN. **CM:** answering as if generative. **FU:** what's the ML-specific attack instead? (evasion.)

**115. Biggest DoS vector?** Unauthenticated, unbounded **SHAP**; mitigate with rate limits, quotas, timeouts, worker pool. **FU:** memory DoS? (big CSV before row-cap.)

**116. How would you add API keys?** Hashed keys in a store, checked by a FastAPI `Depends` on routers; service layer untouched (auth is transport). **FU:** where store hashes? (DB/secret store.)

**117. RBAC design?** analyst (read/run) vs admin (rotate keys, swap model) as a claim on the key/JWT, enforced by a dependency. **FU:** per-resource checks?

**118. XSS exposure?** Backend returns JSON not HTML; risk is frontend rendering — React escapes by default; avoid `dangerouslySetInnerHTML`. **FU:** which field to watch? (geo.city / any string.)

**119. CSRF?** Low — stateless, token-less, no session cookie to ride. If you add cookie sessions, add CSRF/SameSite. With bearer keys, moot. **FU:** why bearer avoids CSRF?

**120. Secrets management?** None needed today (no DB pass/API key); `.env.example` documents non-secret config; real `.env` gitignored. **FU:** when secrets appear? (DB, third-party.)

**121. Container security wins?** Non-root uid 1000, multi-stage slim image, caches in /tmp, TLS at the edge, healthcheck. **FU:** why non-root matters?

**122. Adversarial ML threat?** Feature-space evasion (shape traffic to look BENIGN), model extraction via queries. Defenses: adversarial training, ensembling, anomaly score, rate limiting. **FU:** is the model itself a target? (yes.)

**123. Error handling and info leakage?** Global handler logs the trace, returns generic 500 — no internals leak; validation errors are specific 400s. **FU:** what to log safely? (no PII.)

**124. Hardening priority order?** auth → rate limiting → timeouts/worker pool → stream-parse uploads → confidence floor → audit logging → security headers → dep scanning. **FU:** why auth first?

---

## H. Deployment & Ops (125–135)

**125. How is it deployed?** Multi-stage Docker; locally via compose (frontend:8080 + backend), hosted via HF Spaces (backend, 16GB RAM) + Vercel (frontend). **FU:** why HF? (RAM for TF + WS + Docker.)

**126. Why multi-stage Docker?** Build deps isolated → slim non-root runtime → smaller image/attack surface. **FU:** image build cost? (~10–15 min, TF.)

**127. How does same-origin work in prod?** nginx serves the SPA and proxies /api,/ws,/health,/predict to the backend → browser sees one origin. **FU:** the split (Vercel+HF) case? (cross-origin → set CORS + VITE_API_BASE.)

**128. How are model "migrations" done?** Swap the artifact trio atomically; `old/` vs `new/` are versions; an eval gate should block regressions. **FU:** atomic why?

**129. CI/CD present?** Not in-repo; deploys are manual/auto-on-push. I'd add Actions: test → build → push to Space, gated on green. **FU:** what tests gate? (unittest + eval report.)

**130. Why `${PORT:-8000}`?** Portability — HF doesn't inject PORT (stays 8000), Cloud Run/Render/Railway do. **FU:** healthcheck uses it? (yes.)

**131. Why caches in /tmp in Docker?** HF runs as uid 1000 with read-only HOME; TF/SHAP/matplotlib need a writable cache dir. **FU:** symptom if missing? (startup crash / permission error.)

**132. What's NOT in the image and why?** The 149 MB `X_dcnn.npy` — too big; `demo_flows.npy` (~150 KB) substitutes for SHAP background + stream. **FU:** effect on demo? (synthetic-but-realistic stream.)

**133. How do health checks work?** Docker `HEALTHCHECK` curls `/health`; compose gates the frontend on backend health. **FU:** what does /health prove? (artifacts loaded.)

**134. Rolling out a config change (e.g., CORS)?** Set env var in HF Settings → Space restarts; no code change. **FU:** zero-downtime? (replicas + LB drain.)

**135. How to debug a wedged container?** Check startup logs (model load), healthcheck status, cache-dir permissions; reproduce locally with the same env. **FU:** common cause? (missing/corrupt artifact.)

---

## I. Algorithms & Data Structures in the project (136–144)

**136. Reservoir sampling — describe and analyze.** One-pass uniform sample; O(n) time, O(k) space; item i kept with prob k/i maintained inductively. Used in `demo_stream`. **FU:** parallel/distributed version?

**137. Argmax — where and why?** `np.argmax(probs)` picks the predicted class from softmax; O(C). **FU:** ties? (first index.)

**138. LRU cache — where and why?** `@lru_cache(10000)` on geoip — recurring IPs avoid repeat lookups; bounded memory, O(1) amortized. **FU:** eviction policy? (least-recently-used.)

**139. Sorting by |SHAP|.** `sorted(..., key=abs(value), reverse=True)[:k]` — O(F log F) over 78 features; surfaces top contributors. **FU:** partial sort? (`heapq.nlargest` for big F.)

**140. Hash map as schema/contract.** `FEATURE_NAMES` order + dict reorder = O(F) vectorization; `SEVERITY_MAP` O(1) lookup. **FU:** why dict over list scan?

**141. Softmax — what/why.** Converts logits to a probability distribution over 11 classes; confidence = max prob. **FU:** numerical stability? (max-subtraction internally.)

**142. MinMax scaling math.** `(x - min)/(max - min)` per feature from train stats; serve-time reuse. O(F). **FU:** out-of-range serve values? (can exceed [0,1]; model still runs.)

**143. Convolution as an operation.** Sliding dot-product of a kernel over the feature axis → feature maps; weight sharing → few params, local pattern detection. **FU:** receptive field growth with depth?

**144. Complexity of a forward pass.** Roughly linear in params/activations; fixed per input → O(1) wrt request count, constant per flow. **FU:** vs SHAP? (many passes.)

---

## J. Behavioral / Project-ownership (145–155)

**145. What was the hardest part?** Honest pick: getting a clean-clone deploy on a non-root, read-only-HOME host without the 149 MB dataset — solved with ordered artifact fallbacks + /tmp caches. **FU:** how did you discover it? (HF permission errors.)

**146. A bug you found and fixed?** e.g., SHAP output shape varying by version → added list-vs-ndarray handling; or double-scaling caught via the `already_scaled` flag. **FU:** how prevent class of bug? (tests + invariants.)

**147. A tradeoff you made and would revisit?** Per-row batch loop (chose simplicity, would vectorize); undersampling (would try class weights for Botnet recall). **FU:** why ship the simpler version first?

**148. What would you do differently?** Add a confidence guard and batch vectorization from day one; benchmark XGBoost before committing to a CNN. **FU:** why didn't you?

**149. How did you validate correctness?** unittest for pure functions + a TestClient integration test; the eval report (per-class metrics, confusion matrix) for the model. **FU:** coverage gaps?

**150. How would you explain this to a non-technical stakeholder?** "It watches network traffic, recognizes attack patterns it learned from real data, flags the dangerous ones with a severity, and shows analysts *why* it flagged each one." **FU:** business value? (faster triage, fewer missed attacks.)

**151. What metric defines success for the product?** Detection recall on attacks (don't miss intrusions) balanced against false-positive rate (avoid alert fatigue) — plus analyst trust via explanations. **FU:** how measure trust?

**152. Why should we trust the model?** Held-out test (47,912 flows), per-class metrics + confusion matrix published, SHAP explanations per alert, and known weak spots documented (Botnet). **FU:** what about unknown attacks? (anomaly add-on.)

**153. If predictions started degrading in prod, your steps?** Check drift (class distribution/confidence), validate the artifact trio + feature order, inspect recent inputs for skew, compare against the eval baseline, retrain if drift confirmed. **FU:** automate it?

**154. What did you learn?** Stateless ML serving, explainability as a product requirement, deploy-anywhere constraints (non-root, RAM), and the discipline of train/serve consistency. **FU:** apply where next?

**155. Sell me on this project in 30 seconds.** *(Use the pitch from the README.)* End-to-end ML system: data pipeline → 99.48% CNN → explainable, real-time, containerized, deployed — with a clear-eyed list of what I'd productionize next. **FU:** the single thing you're proudest of?

---

## Rapid-fire one-liners (bonus drills)

- *Feature count?* 78. *Classes?* 11. *Accuracy?* 99.48%. *Macro-F1?* 97.67%. *Weakest class?* Botnet (recall 77.75%).
- *Input shape?* (batch, 78, 1). *Loss?* sparse categorical cross-entropy. *Optimizer?* Adam 1e-4.
- *Explainer?* SHAP GradientExplainer. *Background size?* 100. *Stream rate?* 1 Hz.
- *Upload cap?* 5000 rows / 64 MB. *SHAP cap in batch?* 50. *Severity levels?* critical/high/medium/low.
- *Dataset?* CICIDS2017. *Flow tool?* CICFlowMeter. *Dropped columns?* Flow ID, Source/Dest IP, Timestamp.
- *Backend?* FastAPI/Uvicorn. *Model?* Keras 1D-CNN (.h5). *Edge?* TFLite. *Proxy?* nginx. *Hosting?* HF Spaces + Vercel.
