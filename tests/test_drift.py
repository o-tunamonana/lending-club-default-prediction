"""Drift detection: injected drift is flagged, clean data is not, and drift triggers retraining."""

import pandas as pd
import pytest

from lcdp.drift import decision_cutoff, detect_data_drift, inject_drift, model_performance, should_retrain


@pytest.fixture(scope="module")
def reference(train_df):
    return train_df[train_df["issue_d"].str.startswith("2015")].sample(3000, random_state=1)


def test_same_period_data_is_not_flagged(train_df, reference):
    current = train_df[train_df["issue_d"].str.startswith("2015")].drop(reference.index).sample(500, random_state=2)
    result = detect_data_drift(reference, current)
    assert result["dataset_drift"] is False


def test_injected_drift_is_flagged(test_df, reference):
    drifted = inject_drift(test_df.head(500))
    result = detect_data_drift(reference, drifted)
    assert result["dataset_drift"] is True
    for feature in ["fico_n", "revenue", "dti_n", "loan_amnt"]:
        assert result["features"][feature]["drifted"], feature


def test_drift_triggers_retraining_with_cooldown():
    assert should_retrain(True, last_trigger_ts=None, now=1000)
    assert not should_retrain(False, last_trigger_ts=None, now=1000)
    assert not should_retrain(True, last_trigger_ts=900, now=1000, cooldown=900)
    assert should_retrain(True, last_trigger_ts=0, now=1000, cooldown=900)


def test_end_to_end_drift_flag_leads_to_retraining(test_df, reference):
    result = detect_data_drift(reference, inject_drift(test_df.head(500)))
    assert should_retrain(result["dataset_drift"], last_trigger_ts=None, now=0)


def test_model_performance_from_labels(test_df):
    labelled = test_df.head(400).assign(default_probability=test_df.head(400)["Default"] * 0.6 + 0.2,
                                        actual=test_df.head(400)["Default"])
    perf = model_performance(labelled, cutoff=0.5)
    assert perf["roc_auc"] == pytest.approx(1.0)
    assert model_performance(labelled.head(10), cutoff=0.5) is None


def test_only_one_retraining_per_drift_episode():
    # Sustained drift after a successful retrain must not retrain again every cooldown period
    assert not should_retrain(True, last_trigger_ts=0, now=10_000, cooldown=900, retrained_this_episode=True)
    assert should_retrain(True, last_trigger_ts=0, now=10_000, cooldown=900, retrained_this_episode=False)


def test_cutoff_uses_logged_value_from_latest_prediction():
    window = pd.DataFrame({"id": [1, 2, 3], "decision": ["approve", "reject", "approve"],
                           "default_probability": [0.1, 0.5, 0.2], "approval_cutoff": [0.27, 0.27, 0.31]})
    assert decision_cutoff(window) == 0.31


def test_cutoff_fallback_without_logged_value():
    window = pd.DataFrame({"id": [1, 2, 3, 4], "decision": ["approve", "approve", "review", "reject"],
                           "default_probability": [0.10, 0.26, 0.30, 0.60], "approval_cutoff": [None] * 4})
    assert decision_cutoff(window) == pytest.approx(0.28)  # between highest approved and lowest other
    assert decision_cutoff(window.assign(decision="reject")) is None
