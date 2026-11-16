# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    MODEL_DIR=/app/model

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Fetch the latest trained model from DagsHub through MLflow -------------
ARG MLFLOW_TRACKING_URI=https://dagshub.com/Redtom67/mlops-ex7.mlflow
ARG MLFLOW_TRACKING_USERNAME=Redtom67
ARG MODEL_NAME=digits-rf
# Changing CACHEBUST (the CI passes the run id) forces Docker to re-run the
# download step instead of reusing a cached, possibly outdated, model.
ARG CACHEBUST=0

COPY scripts/download_model.py scripts/download_model.py
# The DagsHub token is mounted as a BuildKit secret: it is available only during
# this RUN step and is never stored in an image layer.
RUN --mount=type=secret,id=dagshub_token,required=true \
    echo "cache bust: ${CACHEBUST}" && \
    MLFLOW_TRACKING_PASSWORD="$(cat /run/secrets/dagshub_token)" \
    python scripts/download_model.py --model-name "${MODEL_NAME}" --output-dir "${MODEL_DIR}"

# --- API ---------------------------------------------------------------------
COPY app/ app/

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
