import numpy as np
import pytest

from lcdp.train import BASELINE_NAME, approval_cutoff, evaluate, fit_candidate, make_candidates, split_train_valid


@pytest.fixture(scope="module")
def fitted(train_df):
    X_train, X_valid, y_train, y_valid = split_train_valid(train_df)
    results = {}
    for name in [BASELINE_NAME, "gradient_boosting_untuned"]:
        pipeline, _ = make_candidates()[name]
        model, metrics, cutoff = fit_candidate(pipeline, X_train, y_train, X_valid, y_valid)
        results[name] = {"model": model, "metrics": metrics, "cutoff": cutoff}
    return results, X_valid, y_valid


def test_model_beats_fico_baseline(fitted):
    results, _, _ = fitted
    model_auc = results["gradient_boosting_untuned"]["metrics"]["roc_auc"]
    baseline_auc = results[BASELINE_NAME]["metrics"]["roc_auc"]
    assert model_auc > baseline_auc + 0.01


def test_model_beats_naive_prediction(fitted):
    results, _, y_valid = fitted
    # A naive model predicts the training default rate for everyone: AUC 0.5, Brier = p(1-p)
    p = y_valid.mean()
    metrics = results["gradient_boosting_untuned"]["metrics"]
    assert metrics["roc_auc"] > 0.6
    assert metrics["brier"] < p * (1 - p)


def test_approval_policy_lowers_default_rate(fitted):
    metrics = fitted[0]["gradient_boosting_untuned"]["metrics"]
    assert 0.75 < metrics["approval_rate"] < 0.85
    assert metrics["default_rate_approved"] < metrics["default_rate_all"]


def test_approval_cutoff_and_evaluate_on_known_input():
    scores = np.linspace(0, 1, 101)
    assert approval_cutoff(scores, 0.8) == pytest.approx(0.8)
    m = evaluate([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], cutoff=0.5)
    assert m["roc_auc"] == 1.0 and m["default_rate_approved"] == 0.0 and m["approval_rate"] == 0.5
