import argparse
import os
import sys

import mlflow
from mlflow.tracking import MlflowClient


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    p.add_argument("--run-id", required=True)
    p.add_argument("--model-name", default="milestone3-model")
    p.add_argument("--stage", default="Staging")  # or Production
    p.add_argument("--description", default="Registered by pipeline")
    args = p.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()

    # Register the model logged as an artifact path "model_artifacts"
    # We logged model.pkl under artifact_path="model_artifacts"
    model_uri = f"runs:/{args.run_id}/model_artifacts"
    result = mlflow.register_model(model_uri=model_uri, name=args.model_name)

    version = result.version
    client.update_model_version(
        name=args.model_name,
        version=version,
        description=args.description,
    )

    # Add useful tags
    client.set_model_version_tag(args.model_name, version, "source_run_id", args.run_id)

    # Stage transition (None -> Staging -> Production)
    client.transition_model_version_stage(
        name=args.model_name,
        version=version,
        stage=args.stage,
        archive_existing_versions=False,
    )

    print(f"Registered {args.model_name} v{version} to stage={args.stage}")
    print(f"run_id={args.run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
