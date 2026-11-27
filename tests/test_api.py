"""API behaviour with a stub model."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main
from app.model_store import LoadedModel

GOOD = {"revenue": 65000, "dti_n": 18.5, "loan_amnt": 12000, "fico_n": 702,
        "emp_length": "5 years", "purpose": "debt_consolidation",
        "home_ownership_n": "RENT", "addr_state": "NY"}


class StubModel:
    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        return np.array([[1 - self.p, self.p]] * len(X))


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("MODEL_URI", "unused")
    monkeypatch.setattr(main.store, "load_local", lambda uri: None)
    monkeypatch.setattr(main.store, "_current", None)
    with TestClient(main.app) as client:
        yield client


def use_model(p, cutoff=0.27):
    main.store.set(LoadedModel(StubModel(p), "7", cutoff))


def test_valid_request_returns_readable_prediction(client, monkeypatch):
    observed = []

    def log_prediction(features, probability, decision, cutoff, version, scenario):
        observed.append((cutoff, version))
        return 1

    monkeypatch.setattr(main.db, "log_prediction", log_prediction)
    use_model(0.12)
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "approve" and body["risk_level"] == "low"
    assert body["default_probability"] == 0.12 and body["model_version"] == "7"
    assert observed == [(body["approval_cutoff"], "7")] and body["prediction_id"] == 1


@pytest.mark.parametrize("p, decision, risk", [(0.2, "approve", "medium"), (0.3, "review", "medium"),
                                               (0.6, "reject", "high")])
def test_decision_follows_cutoff(client, p, decision, risk):
    use_model(p)
    body = client.post("/predict", json=GOOD).json()
    assert (body["decision"], body["risk_level"]) == (decision, risk)


@pytest.mark.parametrize("change", [{"fico_n": 1200}, {"revenue": -5}, {"addr_state": "ZZ"},
                                    {"emp_length": "forever"}, {"loan_amnt": "lots"}, {"extra": 1}])
def test_bad_input_is_rejected(client, change):
    use_model(0.1)
    assert client.post("/predict", json={**GOOD, **change}).status_code == 422


def test_missing_field_is_rejected(client):
    use_model(0.1)
    payload = {k: v for k, v in GOOD.items() if k != "fico_n"}
    assert client.post("/predict", json=payload).status_code == 422


def test_no_model_returns_503(client):
    assert client.post("/predict", json=GOOD).status_code == 503
    assert client.get("/health").json()["model_loaded"] is False


def test_metrics_endpoint_counts_requests(client):
    use_model(0.1)
    client.post("/predict", json=GOOD)
    client.post("/predict", json={**GOOD, "fico_n": 999})
    metrics = client.get("/metrics").text
    assert "api_requests_total" in metrics and "api_request_latency_seconds_bucket" in metrics
    assert 'api_errors_total{endpoint="/predict",kind="validation",status="422"}' in metrics
    assert "model_predictions_total" in metrics and "model_default_probability_sum" in metrics
    assert 'model_version_info{version="7"} 1.0' in metrics
