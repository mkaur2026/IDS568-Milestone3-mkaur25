import os
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "milestone3")
OUT_CSV = "run_comparison.csv"


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise SystemExit(f"Experiment not found: {EXPERIMENT_NAME}")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=50,
    )

    rows = []
    for r in runs:
        rows.append({
            "run_id": r.info.run_id,
            "start_time": r.info.start_time,
            "C": r.data.params.get("C"),
            "max_iter": r.data.params.get("max_iter"),
            "accuracy": r.data.metrics.get("accuracy"),
            "f1": r.data.metrics.get("f1"),
            "model_hash": r.data.tags.get("model_hash"),
            "data_hash": r.data.tags.get("data_hash"),
        })

    df = pd.DataFrame(rows)

    # keep newest 5 for the milestone requirement table
    df5 = df.head(5).copy()
    df5.to_csv(OUT_CSV, index=False)

    print(f"Saved {OUT_CSV}")
    print(df5.to_string(index=False))


if __name__ == "__main__":
    main()
