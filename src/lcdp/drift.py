"""Drift injection, Evidently drift/performance checks and the retraining rule.

Kept free of database and network code so it can be unit tested directly.
"""

import numpy as np
import pandas as pd

from lcdp.schema import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES

DRIFT_SHARE_THRESHOLD = 0.5   # dataset drift when at least half of the features drift
RETRAIN_COOLDOWN_SECONDS = 900


def inject_drift(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Simulate a riskier applicant pool (e.g. a new marketing channel or a downturn).

    Lower FICO and income, higher DTI and loan size, and more renters.
    Five of the eight model inputs change.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["fico_n"] = (out["fico_n"] - rng.normal(45, 10, len(out))).clip(300, 850)
    out["revenue"] = (out["revenue"] * rng.uniform(0.45, 0.7, len(out))).round(0).clip(lower=1000)
    out["dti_n"] = (out["dti_n"] * rng.uniform(1.4, 1.9, len(out))).clip(upper=99)
    out["loan_amnt"] = (out["loan_amnt"] * rng.uniform(1.3, 1.6, len(out))).round(-2).clip(upper=40000)
    renters = rng.random(len(out)) < 0.6
    out.loc[renters, "home_ownership_n"] = "RENT"
    return out


def _column_mapping():
    from evidently import ColumnMapping

    return ColumnMapping(numerical_features=NUMERIC_FEATURES, categorical_features=CATEGORICAL_FEATURES)


def detect_data_drift(reference: pd.DataFrame, current: pd.DataFrame,
                      threshold: float = DRIFT_SHARE_THRESHOLD) -> dict:
    """Run Evidently's data drift preset on the model inputs."""
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    report = Report(metrics=[DataDriftPreset(drift_share=threshold)])
    report.run(reference_data=reference[FEATURES], current_data=current[FEATURES],
               column_mapping=_column_mapping())
    metrics = report.as_dict()["metrics"]
    summary, by_column = metrics[0]["result"], metrics[1]["result"]["drift_by_columns"]
    return {
        "dataset_drift": bool(summary["dataset_drift"]),
        "drift_share": float(summary["share_of_drifted_columns"]),
        "n_drifted": int(summary["number_of_drifted_columns"]),
        "features": {
            name: {"score": float(info["drift_score"]), "drifted": bool(info["drift_detected"]),
                   "test": info.get("stattest_name")}
            for name, info in by_column.items()
        },
    }


def model_performance(labelled: pd.DataFrame, cutoff: float) -> dict | None:
    """Evidently classification quality on predictions whose outcome is known.

    `labelled` needs columns `default_probability` and `actual`. Returns None when
    there are too few labels or only one class.
    """
    if len(labelled) < 50 or labelled["actual"].nunique() < 2:
        return None
    from evidently import ColumnMapping
    from evidently.metrics import ClassificationQualityMetric
    from evidently.report import Report

    data = labelled[["default_probability", "actual"]].astype({"actual": int})
    report = Report(metrics=[ClassificationQualityMetric(probas_threshold=cutoff)])
    report.run(reference_data=None, current_data=data,
               column_mapping=ColumnMapping(target="actual", prediction="default_probability"))
    result = report.as_dict()["metrics"][0]["result"]["current"]
    keys = ["roc_auc", "accuracy", "precision", "recall", "f1"]
    return {k: float(result[k]) for k in keys if result.get(k) is not None}


def should_retrain(dataset_drift: bool, last_trigger_ts: float | None, now: float,
                   cooldown: float = RETRAIN_COOLDOWN_SECONDS) -> bool:
    """Retrain on dataset drift, but at most once per cooldown window."""
    if not dataset_drift:
        return False
    return last_trigger_ts is None or now - last_trigger_ts >= cooldown
