---
title: NIDS API
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
license: mit
---

# NIDS API

FastAPI backend for the Network Intrusion Detection System demo — an 11-class
1D-CNN over CICIDS2017 flow features, with SHAP explanations and a live
WebSocket alert stream.

This Space hosts only the API. The dashboard frontend is deployed separately
(e.g. on Vercel) and points at this Space via `VITE_API_BASE`.

## Endpoints
- `GET  /health` — liveness + model metadata
- `GET  /docs` — interactive API docs
- `POST /api/analyze` — upload a CICIDS CSV → per-flow alerts + summary
- `WS   /ws/alerts` — live demo alert stream

## Configuration
Set in **Settings → Variables and secrets**:
- `NIDS_CORS_ORIGINS` — your frontend's origin, e.g. `https://your-app.vercel.app`
