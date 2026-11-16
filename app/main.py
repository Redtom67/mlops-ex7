"""FastAPI app serving the latest `digits-rf` model trained with MLflow.

The model is NOT downloaded at runtime: it is fetched from DagsHub (MLflow Model
Registry) when the Docker image is built, by scripts/download_model.py, and
stored in MODEL_DIR inside the image.
"""
import json
import os
from pathlib import Path

import mlflow.sklearn
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/model"))

# Load the model once, when the app starts.
model = mlflow.sklearn.load_model(str(MODEL_DIR))

# Which registry version is baked into this image (written by download_model.py).
_info_file = MODEL_DIR / "model_info.json"
model_info = json.loads(_info_file.read_text()) if _info_file.exists() else {}

app = FastAPI(title="Digits classifier API", version="1.0.0")


class PredictRequest(BaseModel):
    features: list[float] = Field(
        ...,
        min_length=64,
        max_length=64,
        description="An 8x8 handwritten digit image, flattened to 64 pixel values (0-16).",
    )


class PredictResponse(BaseModel):
    prediction: int
    model_name: str | None = None
    model_version: str | None = None


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    x = np.asarray(request.features, dtype=float).reshape(1, -1)
    prediction = int(model.predict(x)[0])
    return PredictResponse(
        prediction=prediction,
        model_name=model_info.get("name"),
        model_version=model_info.get("version"),
    )
