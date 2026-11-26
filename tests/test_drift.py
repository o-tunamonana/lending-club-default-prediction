"""Drift detection: injected drift is flagged, clean data is not, and drift triggers retraining."""

import pytest

from lcdp.drift import detect_data_drift, inject_drift, model_performance, should_retrain


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
