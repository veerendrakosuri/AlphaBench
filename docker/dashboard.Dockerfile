FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip && pip install \
    streamlit==1.62.0 plotly==7.0.0 httpx==0.28.1 pandas==2.3.3 pyyaml==6.0.3

COPY src/alphabench/dashboard/ ./src/alphabench/dashboard/

RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501
CMD ["sh", "-c", "streamlit run src/alphabench/dashboard/app.py \
     --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true"]
