"""Monitoring job: every INTERVAL seconds, compare recent predictions with the
training reference using Evidently, store the results in Postgres for Grafana,
and trigger the Prefect retraining deployment when dataset drift is detected.
"""

import logging
import os
import time

import pandas as pd
import requests
from sqlalchemy import create_engine, text

from lcdp.drift import decision_cutoff, detect_data_drift, model_performance, should_retrain

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("monitor")

DATABASE_URL = os.environ["DATABASE_URL"]
REFERENCE_PATH = os.getenv("REFERENCE_PATH", "data/train.csv")
REFERENCE_YEAR = os.getenv("REFERENCE_YEAR", "2015")   # most recent training cohort
REFERENCE_SIZE = int(os.getenv("REFERENCE_SIZE", "5000"))
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "500"))
MIN_ROWS = int(os.getenv("MIN_ROWS", "200"))
INTERVAL = int(os.getenv("INTERVAL_SECONDS", "30"))
COOLDOWN = int(os.getenv("RETRAIN_COOLDOWN_SECONDS", "900"))
PREFECT_API_URL = os.getenv("PREFECT_API_URL", "http://prefect:4200/api")
DEPLOYMENT = os.getenv("TRAINING_DEPLOYMENT", "lending-club-training/lcdp-training")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)


def load_reference() -> pd.DataFrame:
    df = pd.read_csv(REFERENCE_PATH, low_memory=False)
    recent = df[df["issue_d"].astype(str).str.startswith(REFERENCE_YEAR)]
    sample = (recent if len(recent) >= REFERENCE_SIZE else df).sample(REFERENCE_SIZE, random_state=42)
    log.info("Reference: %d loans from %s", len(sample), REFERENCE_YEAR)
    return sample


def recent_window() -> pd.DataFrame:
    query = text("SELECT * FROM predictions ORDER BY id DESC LIMIT :n")
    return pd.read_sql(query, engine, params={"n": WINDOW_SIZE})


def last_trigger_time() -> float | None:
    """Time of the last retraining run that actually started (failed attempts don't count)."""
    with engine.connect() as conn:
        ts = conn.execute(text("""SELECT extract(epoch FROM max(created_at)) FROM retraining_events
                                  WHERE status = 'triggered'""")).scalar()
    return float(ts) if ts is not None else None


def retrained_this_episode() -> bool:
    """True if a retraining run already started since the last window without drift."""
    with engine.connect() as conn:
        return bool(conn.execute(text("""
            SELECT EXISTS (
                SELECT 1 FROM retraining_events
                WHERE status = 'triggered'
                  AND created_at > COALESCE((SELECT max(created_at) FROM drift_metrics WHERE NOT dataset_drift),
                                            '-infinity'::timestamptz))""")).scalar())


def trigger_retraining(drift_share: float) -> str | None:
    """Start a run of the Prefect training deployment through the REST API."""
    flow_name, deployment_name = DEPLOYMENT.split("/")
    deployment = requests.get(f"{PREFECT_API_URL}/deployments/name/{flow_name}/{deployment_name}",
                              timeout=10)
    deployment.raise_for_status()
    run = requests.post(f"{PREFECT_API_URL}/deployments/{deployment.json()['id']}/create_flow_run",
                        json={"parameters": {"reason": "drift", "include_production_data": True},
                              "tags": ["drift-retraining"]}, timeout=10)
    run.raise_for_status()
    log.warning("Drift share %.2f -> retraining flow run %s", drift_share, run.json()["id"])
    return run.json()["id"]


def save(window: pd.DataFrame, drift: dict, perf: dict | None) -> None:
    labelled = window["actual"].notna()
    row = {
        "window_size": len(window),
        "n_labelled": int(labelled.sum()),
        "dataset_drift": drift["dataset_drift"],
        "drift_share": drift["drift_share"],
        "n_drifted_features": drift["n_drifted"],
        "mean_prediction": float(window["default_probability"].mean()),
        "actual_default_rate": float(window.loc[labelled, "actual"].mean()) if labelled.any() else None,
        "share_drift_traffic": float((window["scenario"] == "drift").mean()),
        **{k: (perf or {}).get(k) for k in ["roc_auc", "accuracy", "precision", "recall", "f1"]},
    }
    with engine.begin() as conn:
        metric_id = conn.execute(text(f"""
            INSERT INTO drift_metrics ({', '.join(f'"{k}"' for k in row)}) VALUES ({', '.join(':' + k for k in row)})
            RETURNING id"""), row).scalar()
        conn.execute(text("""
            INSERT INTO feature_drift (metric_id, feature, drift_score, drifted, stattest)
            VALUES (:metric_id, :feature, :score, :drifted, :test)"""),
            [{"metric_id": metric_id, "feature": f, **v} for f, v in drift["features"].items()])


def run_once(reference: pd.DataFrame) -> None:
    window = recent_window()
    if len(window) < MIN_ROWS:
        log.info("Only %d predictions so far, waiting for %d", len(window), MIN_ROWS)
        return
    drift = detect_data_drift(reference, window)

    labelled = window[window["actual"].notna()]
    cutoff = decision_cutoff(window)
    # ROC AUC does not need a cutoff; precision/recall/F1 are only meaningful with the API's real one
    perf = model_performance(labelled, cutoff if cutoff is not None else 0.5)
    if perf and cutoff is None:
        perf = {"roc_auc": perf["roc_auc"]} if "roc_auc" in perf else None
    save(window, drift, perf)
    log.info("drift_share=%.2f dataset_drift=%s auc=%s cutoff=%s", drift["drift_share"],
             drift["dataset_drift"], perf and round(perf.get("roc_auc", 0), 3), cutoff and round(cutoff, 3))

    if should_retrain(drift["dataset_drift"], last_trigger_time(), time.time(), COOLDOWN,
                      retrained_this_episode=retrained_this_episode()):
        status, run_id = "triggered", None
        try:
            run_id = trigger_retraining(drift["drift_share"])
        except Exception as exc:
            status = f"failed: {exc}"[:200]
            log.error("Could not trigger retraining: %s", exc)
        with engine.begin() as conn:
            conn.execute(text("""INSERT INTO retraining_events (reason, drift_share, flow_run_id, status)
                                 VALUES ('data_drift', :share, :run_id, :status)"""),
                         {"share": drift["drift_share"], "run_id": run_id, "status": status})


def main() -> None:
    reference = load_reference()
    while True:
        try:
            run_once(reference)
        except Exception as exc:
            log.exception("Monitoring cycle failed: %s", exc)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
