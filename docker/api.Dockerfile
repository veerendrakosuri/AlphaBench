FROM python:3.12-slim AS base
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 curl && rm -rf /var/lib/apt/lists/*
# libgomp1 is required by LightGBM/XGBoost at runtime. Omitting it produces a
# confusing "cannot open shared object file" error only inside the container.

WORKDIR /app
COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src/ ./src/
RUN pip install -e . --no-deps

COPY config/ ./config/
COPY models/ ./models/
COPY data/processed/ ./data/processed/
COPY reports/metrics/ ./reports/metrics/

RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["sh", "-c", "uvicorn alphabench.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
