"""Model candidates, evaluation and MLflow logging/registration.

Settings come from the INSY 674 notebooks (04 baseline, 05 tuning): a FICO-only
logistic regression is the baseline, and the tuned gradient boosting model was
the final choice. The approval policy approves the safest 80% of loans.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lcdp.preprocessing import build_preprocessor
from lcdp.schema import CHAMPION_ALIAS, FEATURES, REGISTERED_MODEL_NAME, TARGET

log = logging.getLogger(__name__)

RANDOM_STATE = 42
APPROVAL_RATE = 0.8
BASELINE_NAME = "fico_baseline"


def _full(model):
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])


def make_candidates() -> dict:
    """Return the models compared in every training run (name -> (pipeline, params))."""
    fico_only = ColumnTransformer([
        ("fico", Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                           ("scaler", StandardScaler())]), ["fico_n"]),
    ])
    gb_untuned = {"max_iter": 200, "learning_rate": 0.05, "max_leaf_nodes": 15,
                  "min_samples_leaf": 50, "l2_regularization": 1.0}
    gb_tuned = {"max_iter": 510, "learning_rate": 0.0249, "max_leaf_nodes": 15,
                "min_samples_leaf": 337, "l2_regularization": 0.0257, "max_features": 0.5823}
    logreg = {"C": 2.1368, "solver": "liblinear", "max_iter": 2000}

    return {
        BASELINE_NAME: (
            Pipeline([("preprocess", fico_only),
                      ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))]),
            {"model_type": "logistic_regression", "features": "fico_n only"},
        ),
        "logistic_regression": (
            _full(LogisticRegression(random_state=RANDOM_STATE, **logreg)),
            {"model_type": "logistic_regression", **logreg},
        ),
        "gradient_boosting_untuned": (
            _full(HistGradientBoostingClassifier(random_state=RANDOM_STATE, **gb_untuned)),
            {"model_type": "hist_gradient_boosting", **gb_untuned},
        ),
        "gradient_boosting_tuned": (
            _full(HistGradientBoostingClassifier(random_state=RANDOM_STATE, **gb_tuned)),
            {"model_type": "hist_gradient_boosting", **gb_tuned},
        ),
    }


def approval_cutoff(scores, rate: float = APPROVAL_RATE) -> float:
    """Highest predicted default probability that is still approved at this approval rate."""
    return float(np.quantile(np.asarray(scores), rate))


def evaluate(y_true, proba, cutoff: float) -> dict:
    """Ranking, calibration and business metrics for one model."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    approved = proba <= cutoff
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "average_precision": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "approval_rate": float(approved.mean()),
        "default_rate_approved": float(y_true[approved].mean()) if approved.any() else float("nan"),
        "default_rate_all": float(y_true.mean()),
    }


def split_train_valid(df: pd.DataFrame, valid_size: float = 0.2):
    """Stratified hold-out from the training period (the 2016 test file is kept for traffic)."""
    X, y = df[FEATURES], df[TARGET].astype(int)
    return train_test_split(X, y, test_size=valid_size, stratify=y, random_state=RANDOM_STATE)


def fit_candidate(pipeline, X_train, y_train, X_valid, y_valid) -> tuple:
    """Fit one pipeline and return (fitted pipeline, metrics, cutoff)."""
    pipeline.fit(X_train, y_train)
    cutoff = approval_cutoff(pipeline.predict_proba(X_train)[:, 1])
    metrics = evaluate(y_valid, pipeline.predict_proba(X_valid)[:, 1], cutoff)
    return pipeline, metrics, cutoff


def train_and_log(df: pd.DataFrame, experiment: str = "lending-club-default",
                  run_tags: dict | None = None) -> dict:
    """Train every candidate as its own MLflow run and return the best non-baseline result."""
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature

    mlflow.set_experiment(experiment)
    X_train, X_valid, y_train, y_valid = split_train_valid(df)
    results = {}

    for name, (pipeline, params) in make_candidates().items():
        with mlflow.start_run(run_name=name, tags=run_tags or {}) as run:
            log.info("Training %s on %d rows", name, len(X_train))
            fitted, metrics, cutoff = fit_candidate(pipeline, X_train, y_train, X_valid, y_valid)
            mlflow.log_params({**params, "approval_rate_target": APPROVAL_RATE,
                               "n_train": len(X_train), "n_valid": len(X_valid)})
            mlflow.log_metrics({**metrics, "approval_cutoff": cutoff})
            signature = infer_signature(X_valid.head(50), fitted.predict_proba(X_valid.head(50))[:, 1])
            mlflow.sklearn.log_model(fitted, artifact_path="model", signature=signature,
                                     input_example=X_valid.head(3))
            results[name] = {"run_id": run.info.run_id, "metrics": metrics, "cutoff": cutoff}
            log.info("%s: AUC %.4f", name, metrics["roc_auc"])

    baseline_auc = results[BASELINE_NAME]["metrics"]["roc_auc"]
    best_name = max((n for n in results if n != BASELINE_NAME),
                    key=lambda n: results[n]["metrics"]["roc_auc"])
    best = {"name": best_name, **results[best_name], "baseline_auc": baseline_auc}
    if best["metrics"]["roc_auc"] <= baseline_auc:
        raise RuntimeError(f"Best model {best_name} does not beat the FICO baseline")
    return best


def register_and_promote(best: dict, reason: str = "scheduled") -> dict:
    """Register the best run's model and move the champion alias when it should."""
    from mlflow import MlflowClient, register_model

    client = MlflowClient()
    version = register_model(f"runs:/{best['run_id']}/model", REGISTERED_MODEL_NAME)
    tags = {
        "approval_cutoff": f"{best['cutoff']:.6f}",
        "roc_auc": f"{best['metrics']['roc_auc']:.6f}",
        "candidate": best["name"],
        "trigger": reason,
    }
    for key, value in tags.items():
        client.set_model_version_tag(REGISTERED_MODEL_NAME, version.version, key, value)

    try:
        current = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
        current_auc = float(current.tags.get("roc_auc", 0))
    except Exception:  # no champion yet
        current, current_auc = None, 0.0

    # After drift the old validation score is no longer comparable, so retraining always promotes.
    promote = current is None or reason == "drift" or best["metrics"]["roc_auc"] > current_auc
    if promote:
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, version.version)
    log.info("Registered version %s (promoted=%s)", version.version, promote)
    return {"version": version.version, "promoted": promote}
