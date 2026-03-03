import os
import shutil
import subprocess
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator


# Repo paths
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARTIFACTS_DIR = os.path.join(REPO_ROOT, "artifacts")
MLFLOW_DB_PATH = os.path.join(REPO_ROOT, "mlflow.db")
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH}"
MODEL_NAME = "milestone3-model"


def on_failure_callback(context):
    """
    Called when any task fails.
    We clean up partial outputs for this execution date so retries are safe.
    """
    ds = context["ds"]  # e.g., 2026-03-03
    partial_dir = os.path.join(ARTIFACTS_DIR, "runs", ds)
    if os.path.exists(partial_dir):
        shutil.rmtree(partial_dir, ignore_errors=True)


def preprocess_data(**context):
    """
    Idempotent preprocessing:
    - Output path is versioned by Airflow execution date (ds)
    - If processed file exists, we skip doing work and just return the path
    """
    ds = context["ds"]
    run_dir = os.path.join(ARTIFACTS_DIR, "runs", ds)
    os.makedirs(run_dir, exist_ok=True)

    processed_path = os.path.join(run_dir, "processed.txt")

    if os.path.exists(processed_path):
        return {"processed_path": processed_path, "cached": True}

    # Minimal placeholder preprocess output (keeps demo simple)
    with open(processed_path, "w") as f:
        f.write("preprocess_complete\n")

    return {"processed_path": processed_path, "cached": False}


def train_model(**context):
    """
    Calls train.py using MLflow sqlite tracking. Captures and returns run_id via XCom.
    """
    ds = context["ds"]
    run_dir = os.path.join(ARTIFACTS_DIR, "runs", ds)
    os.makedirs(run_dir, exist_ok=True)

    # Hyperparameter (you can later make this variable)
    C = 1.0

    cmd = [
        "python",
        os.path.join(REPO_ROOT, "train.py"),
        "--tracking-uri",
        MLFLOW_TRACKING_URI,
        "--C",
        str(C),
        "--outdir",
        run_dir,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            "train.py failed\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}\n"
        )

    # train.py prints run_id on the last line
    run_id = result.stdout.strip().splitlines()[-1].strip()
    if not run_id:
        raise RuntimeError(f"Could not parse run_id from output:\n{result.stdout}")

    return {"run_id": run_id, "C": C, "run_dir": run_dir}


def register_model(**context):
    """
    Calls register_model.py to register into MLflow model registry and move to Staging.
    """
    ti = context["ti"]
    train_out = ti.xcom_pull(task_ids="train_model")
    run_id = train_out["run_id"]

    cmd = [
        "python",
        os.path.join(REPO_ROOT, "register_model.py"),
        "--tracking-uri",
        MLFLOW_TRACKING_URI,
        "--run-id",
        run_id,
        "--model-name",
        MODEL_NAME,
        "--stage",
        "Staging",
        "--description",
        f"Registered by Airflow ds={context['ds']}",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            "register_model.py failed\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}\n"
        )

    return {"registered_run_id": run_id, "model_name": MODEL_NAME, "stage": "Staging"}


default_args = {
    "owner": "mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
    "on_failure_callback": on_failure_callback,
}

with DAG(
    dag_id="train_pipeline",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["milestone3"],
) as dag:

    t_preprocess = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
        provide_context=True,
    )

    t_train = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
        provide_context=True,
    )

    t_register = PythonOperator(
        task_id="register_model",
        python_callable=register_model,
        provide_context=True,
    )

    t_preprocess >> t_train >> t_register
