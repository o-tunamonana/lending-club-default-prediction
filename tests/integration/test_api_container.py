"""Exercise the built API container over HTTP."""

import os
import time

import pytest
import requests

pytestmark = pytest.mark.integration
API_URL = os.getenv("API_URL", "http://localhost:8000")
GOOD = {"revenue": 65000, "dti_n": 18.5, "loan_amnt": 12000, "fico_n": 702,
        "emp_length": "5 years", "purpose": "debt_consolidation",
        "home_ownership_n": "RENT", "addr_state": "NY"}


@pytest.fixture(scope="module", autouse=True)
def api_ready():
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        try:
            if requests.get(f"{API_URL}/health", timeout=3).json().get("model_loaded"):
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    pytest.fail(f"API at {API_URL} did not load a model in time")


def test_container_scores_a_loan():
    response = requests.post(f"{API_URL}/predict", json=GOOD, timeout=10)
    assert response.status_code == 200, response.text
    body = response.json()
    assert 0 <= body["default_probability"] <= 1
    assert body["decision"] in {"approve", "review", "reject"}


def test_container_rejects_bad_input():
    response = requests.post(f"{API_URL}/predict", json={**GOOD, "fico_n": "high"}, timeout=10)
    assert response.status_code == 422


def test_container_exposes_prometheus_metrics():
    requests.post(f"{API_URL}/predict", json=GOOD, timeout=10).raise_for_status()
    metrics = requests.get(f"{API_URL}/metrics", timeout=10).text
    assert 'api_requests_total{endpoint="/predict"' in metrics
    assert "model_default_probability_count" in metrics
