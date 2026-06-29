# build deps in a venv, then copy into a slim runtime image
FROM python:3.11-slim AS builder
WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt


FROM python:3.11-slim AS final
# HF Spaces runs as uid 1000 with a read-only HOME, so point the caches at /tmp.
ENV PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8000 \
    HOME=/home/user \
    XDG_CACHE_HOME=/tmp/cache \
    MPLCONFIGDIR=/tmp/cache/mpl
RUN useradd -m -u 1000 user
WORKDIR /app

# venv stays root-owned but world-readable, so uid 1000 imports from it without a chown
COPY --from=builder /opt/venv /opt/venv
COPY scripts/ ./scripts/
COPY models/ ./models/
# small flow sample for the demo stream and SHAP background; X_dcnn.npy is not bundled
COPY data/demo_flows.npy ./data/demo_flows.npy

USER user

# HF doesn't inject $PORT (it routes to app_port=8000); override $PORT elsewhere
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:%s/health' % os.getenv('PORT','8000')).status==200 else 1)"

CMD ["sh", "-c", "uvicorn scripts.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
