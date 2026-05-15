"""FastAPI service exposing the NIDS model with SHAP explanations.

Run from project root:
    uvicorn scripts.api:app --reload --port 8000
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .app.routers import health, predict, stream

app = FastAPI(title="NIDS API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(stream.router)
