import argparse
import sys

import mlflow
from mlflow.tracking import MlflowClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--min-accuracy", type=float, default=0.90)
    parser.add_argument("--min-f1", type=float, default=0.85)
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()

    run = client.get_run(args.run_id)
    metrics = run.data.metrics

    acc = metrics.get("accuracy")
    f1 = metrics.get("f1")

    if acc is None or f1 is None:
        print("FAILED: Missing required metrics in MLflow run (need accuracy and f1).")
        sys.exit(1)

    passed = True

    if acc < args.min_accuracy:
        print(f"FAILED: accuracy {acc:.4f} < {args.min_accuracy:.4f}")
        passed = False
    else:
        print(f"PASSED: accuracy {acc:.4f} >= {args.min_accuracy:.4f}")

    if f1 < args.min_f1:
        print(f"FAILED: f1 {f1:.4f} < {args.min_f1:.4f}")
        passed = False
    else:
        print(f"PASSED: f1 {f1:.4f} >= {args.min_f1:.4f}")

    if not passed:
        print("❌ MODEL VALIDATION FAILED")
        sys.exit(1)

    print("✅ MODEL VALIDATION PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()
