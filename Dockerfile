# build deps into a venv, then copy into a slim runtime image
FROM python:3.11-slim AS builder
WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt


FROM python:3.11-slim AS final
ENV PYTHONUNBUFFERED=1 PATH="/opt/venv/bin:$PATH"
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY scripts/ ./scripts/
COPY models/ ./models/
# Small (~150 KB) real-flow sample: powers the live demo stream and the SHAP
# background. The 149 MB data/processed/X_dcnn.npy is intentionally NOT bundled;
# mount it in if you want the exact reference distribution.
COPY data/demo_flows.npy ./data/demo_flows.npy

# Honor $PORT so the same image runs on Hugging Face Spaces, Render, Railway,
# Cloud Run, etc. Defaults to 8000 for local use / docker-compose.
ENV PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:%s/health' % os.getenv('PORT','8000')).status==200 else 1)"

CMD ["sh", "-c", "uvicorn scripts.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
