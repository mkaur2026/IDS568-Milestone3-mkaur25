import argparse
import hashlib
import json
import os
from datetime import datetime

import joblib
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def file_hash(path: str) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def ensure_local_experiment(experiment_name: str, artifact_dir: str) -> None:
    """
    For file-based tracking, force experiment artifact_location to a local file:// path
    so MLflow doesn't try to use the mlflow-artifacts proxy scheme (http/https only).
    """
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        client.create_experiment(
            name=experiment_name,
            artifact_location=f"file:{os.path.abspath(artifact_dir)}",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", f"file:{os.path.abspath('mlruns')}"),
        help="Use file:./mlruns for terminal-only local tracking.",
    )
    parser.add_argument(
        "--experiment",
        default=os.getenv("MLFLOW_EXPERIMENT_NAME", "milestone3"),
    )
    parser.add_argument("--C", type=float, default=1.0)  # hyperparameter to vary
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--outdir", default="artifacts")
    parser.add_argument("--artifact-dir", default="mlruns_artifacts")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs("mlruns", exist_ok=True)
    os.makedirs(args.artifact_dir, exist_ok=True)

    # ---------- MLflow setup (LOCAL) ----------
    mlflow.set_tracking_uri(args.tracking_uri)

    # IMPORTANT: Make sure experiment uses local file artifact location
    ensure_local_experiment(args.experiment, args.artifact_dir)
    mlflow.set_experiment(args.experiment)

    # ---------- Data (simple + reproducible) ----------
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()

    data_path = os.path.join(args.outdir, "train_data.csv")
    df.to_csv(data_path, index=False)

    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        # ---------- Train ----------
        model = LogisticRegression(C=args.C, max_iter=args.max_iter, n_jobs=1)
        model.fit(X_train, y_train)

        # ---------- Evaluate ----------
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")

        # ---------- Save model artifact ----------
        model_path = os.path.join(args.outdir, "model.pkl")
        joblib.dump(model, model_path)

        # ---------- Hashes (lineage) ----------
        model_hash = file_hash(model_path)
        data_hash = file_hash(data_path)

        # ---------- Log params + metrics ----------
        mlflow.log_params(
            {
                "C": args.C,
                "max_iter": args.max_iter,
                "data_path": os.path.abspath(data_path),
                "tracking_uri": args.tracking_uri,
            }
        )
        mlflow.log_metrics(
            {
                "accuracy": float(acc),
                "f1": float(f1),
            }
        )

        # ---------- Log artifacts ----------
        mlflow.log_artifact(model_path, artifact_path="model_artifacts")
        mlflow.log_artifact(data_path, artifact_path="data_artifacts")

        # ---------- Tags = lineage metadata ----------
        mlflow.set_tag("model_hash", model_hash)
        mlflow.set_tag("data_hash", data_hash)
        mlflow.set_tag("trained_at_utc", datetime.utcnow().isoformat())

        # ---------- Lineage JSON artifact ----------
        lineage = {
            "run_id": run_id,
            "params": {"C": args.C, "max_iter": args.max_iter},
            "metrics": {"accuracy": float(acc), "f1": float(f1)},
            "hashes": {"model_hash": model_hash, "data_hash": data_hash},
            "tracking": {"tracking_uri": args.tracking_uri, "experiment": args.experiment},
        }
        lineage_path = os.path.join(args.outdir, "lineage.json")
        with open(lineage_path, "w") as f:
            json.dump(lineage, f, indent=2)
        mlflow.log_artifact(lineage_path, artifact_path="lineage")

        print(run_id)


if __name__ == "__main__":
    main()
