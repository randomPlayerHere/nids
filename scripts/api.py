"""NIDS API. Run from project root:  uvicorn scripts.api:app --reload --port 8000"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .app.config import settings
from .app.routers import analyze, health, predict, stream

logging.basicConfig(level=settings.LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("nids")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load the heavy model once at startup instead of on the first request
    logger.info("loading model...")
    from .app.services import model

    model.load_models()
    logger.info("startup done (%d classes, %d features)", len(model.LABELS), model.N_FEATURES)
    yield


app = FastAPI(title="NIDS API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000
    logger.info("%s %s -> %s (%.1fms)", request.method, request.url.path, response.status_code, ms)
    return response


@app.exception_handler(Exception)
async def on_error(request: Request, exc: Exception):
    logger.exception("error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


app.include_router(health.router)
app.include_router(predict.router)
app.include_router(stream.router)
app.include_router(analyze.router)
