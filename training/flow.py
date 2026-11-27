"""Prefect training pipeline: load -> validate -> train (MLflow) -> register.

Run in the `training` container. On start it trains once if no champion model
exists yet, then serves the flow as a Prefect deployment so the monitoring
service can trigger retraining when drift is detected.
"""

import logging
import os
import time

import pandas as pd
from prefect import flow, get_run_logger, task

from lcdp.schema import CHAMPION_ALIAS, FEATURES, REGISTERED_MODEL_NAME, TARGET
from lcdp.train import register_and_promote, train_and_log
from lcdp.validation import validate_training_data

TRAIN_PATH = os.getenv("TRAIN_PATH", "data/train.csv")
DATABASE_URL = os.getenv("DATABASE_URL")
DEPLOYMENT_NAME = "lcdp-training"


@task(retries=2, retry_delay_seconds=5)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    get_run_logger().info("Loaded %d rows from %s", len(df), path)
    return df


@task
def validate(df: pd.DataFrame) -> dict:
    summary = validate_training_data(df)
    get_run_logger().info("Validation passed: %s", summary)
    return summary


@task
def load_production_labels(database_url: str | None) -> pd.DataFrame:
    """Predictions whose real outcome has arrived (feedback), used when retraining after drift."""
    if not database_url:
        return pd.DataFrame(columns=[*FEATURES, TARGET])
    from sqlalchemy import create_engine

    query = f"SELECT {', '.join(FEATURES)}, actual AS \"{TARGET}\" FROM predictions WHERE actual IS NOT NULL"
    df = pd.read_sql(query, create_engine(database_url))
    get_run_logger().info("Loaded %d labelled production rows", len(df))
    return df


@task
def train(df: pd.DataFrame, reason: str) -> dict:
    return train_and_log(df, run_tags={"trigger": reason})


@task
def register(best: dict, reason: str) -> dict:
    return register_and_promote(best, reason=reason)


@flow(name="lending-club-training", log_prints=True)
def training_flow(reason: str = "manual", include_production_data: bool = False) -> dict:
    raw = load_data(TRAIN_PATH)
    summary = validate(raw)
    data = raw[[*FEATURES, TARGET]]
    if include_production_data:
        production = load_production_labels(DATABASE_URL)
        data = pd.concat([data, production], ignore_index=True)
    best = train(data, reason)
    result = register(best, reason)
    print(f"Best model: {best['name']} AUC={best['metrics']['roc_auc']:.4f} "
          f"(baseline {best['baseline_auc']:.4f}), version {result['version']}, rows {summary['n_rows']}")
    return {**result, "model": best["name"], "roc_auc": best["metrics"]["roc_auc"]}


def champion_exists() -> bool:
    from mlflow import MlflowClient

    for _ in range(30):  # MLflow may still be starting
        try:
            MlflowClient().get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
            return True
        except Exception as exc:
            if "RESOURCE_DOES_NOT_EXIST" in str(exc) or "not found" in str(exc).lower():
                return False
            logging.warning("MLflow not ready (%s), retrying", type(exc).__name__)
            time.sleep(5)
    raise RuntimeError("MLflow tracking server is not reachable")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if not champion_exists():
        training_flow(reason="initial")
    training_flow.serve(name=DEPLOYMENT_NAME)
