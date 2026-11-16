"""Download the latest version of a registered model from the MLflow Model Registry.

Used during `docker build`. Credentials come from the standard MLflow env vars:
    MLFLOW_TRACKING_URI       e.g. https://dagshub.com/Redtom67/mlops-ex7.mlflow
    MLFLOW_TRACKING_USERNAME  your DagsHub username
    MLFLOW_TRACKING_PASSWORD  a DagsHub access token
"""
import argparse
import json
from pathlib import Path

import mlflow
from mlflow import MlflowClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="digits-rf")
    parser.add_argument("--output-dir", default="model")
    args = parser.parse_args()

    client = MlflowClient()
    versions = client.search_model_versions(f"name='{args.model_name}'")
    if not versions:
        raise SystemExit(f"No registered versions found for model '{args.model_name}'")

    # "Latest trained model" = highest version number in the registry.
    latest = max(versions, key=lambda v: int(v.version))
    print(f"Latest version of '{args.model_name}': v{latest.version} (run {latest.run_id})")

    output_dir = Path(args.output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{args.model_name}/{latest.version}",
        dst_path=str(output_dir),
    )

    (output_dir / "model_info.json").write_text(
        json.dumps({"name": args.model_name, "version": str(latest.version), "run_id": latest.run_id})
    )
    print(f"Model downloaded to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
