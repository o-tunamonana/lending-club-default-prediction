from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, event, text

from api.main import Champion, create_app

LOAN = {
    "revenue": 65000, "dti_n": 18.5, "loan_amnt": 12000, "fico_n": 700,
    "emp_length": "5 years", "purpose": "debt_consolidation",
    "home_ownership_n": "RENT", "addr_state": "NY",
}


class FicoModel:
    def __init__(self):
        self.pipeline = make_pipeline(StandardScaler(), LogisticRegression())
        self.pipeline.fit([[620], [650], [700], [740], [780], [810]], [1, 1, 1, 0, 0, 0])
        self.classes_ = self.pipeline.classes_

    def predict_proba(self, frame):
        return self.pipeline.predict_proba(frame[["fico_n"]].to_numpy())


@pytest.fixture
def client(tmp_path):
    app = create_app()
    engine = create_engine(f"sqlite:///{tmp_path / 'predictions.db'}")

    @event.listens_for(engine, "connect")
    def add_now(connection, _):
        connection.create_function("now", 0, lambda: "2026-01-01")

    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE predictions (
                id INTEGER PRIMARY KEY, revenue REAL, dti_n REAL, loan_amnt REAL,
                fico_n REAL, emp_length TEXT, purpose TEXT, home_ownership_n TEXT,
                addr_state TEXT, default_probability REAL, approval_cutoff REAL, decision TEXT,
                model_version TEXT, scenario TEXT, actual INTEGER, feedback_at TEXT
            )
        """))

    @asynccontextmanager
    async def test_lifespan(app):
        app.state.engine = engine
        app.state.champion = Champion(FicoModel(), "7", 0.5)
        yield

    app.router.lifespan_context = test_lifespan
    try:
        with TestClient(app) as http:
            yield http, app, engine
    finally:
        engine.dispose()


def test_probability_and_decision_follow_fico(client):
    http, _, _ = client
    low = http.post("/predict", json={**LOAN, "fico_n": 680})
    high = http.post("/predict", json={**LOAN, "fico_n": 760})
    assert low.status_code == high.status_code == 200
    low, high = low.json(), high.json()
    assert 0 <= high["default_probability"] < low["default_probability"] <= 1
    assert high["decision"] == "approve"
    assert low["decision"] == "reject"


@pytest.mark.parametrize(("changes", "field"), [
    ({"fico_n": 1200}, "fico_n"),
    ({"revenue": 0}, "revenue"),
    ({"emp_length": "unknown"}, "emp_length"),
    ({"Default": 1}, "Default"),
])
def test_invalid_input_returns_field_and_updates_metrics(client, changes, field):
    http, _, engine = client
    response = http.post("/predict", json={**LOAN, **changes})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][-1] == field
    metrics = http.get("/metrics").text
    assert 'api_requests_total{endpoint="/predict",status="422"} 1.0' in metrics
    assert 'api_errors_total{endpoint="/predict",kind="validation",status="422"} 1.0' in metrics
    with engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM predictions")).scalar_one() == 0


def test_prediction_and_feedback_are_logged(client):
    http, _, engine = client
    result = http.post("/predict", json=LOAN, headers={"X-Traffic-Scenario": "drift"}).json()
    label = {"prediction_id": result["prediction_id"], "actual": 1}
    assert http.post("/feedback", json=label).status_code == 200
    assert http.post("/feedback", json=label).status_code == 200
    assert http.post("/feedback", json={**label, "actual": 0}).status_code == 409
    assert http.post("/feedback", json={**label, "prediction_id": 1000}).status_code == 404
    with engine.connect() as conn:
        row = conn.execute(text("SELECT * FROM predictions")).mappings().one()
    assert row["scenario"] == "drift"
    assert row["actual"] == 1
    assert row["fico_n"] == LOAN["fico_n"]
    assert row["default_probability"] == result["default_probability"]
    assert row["approval_cutoff"] == result["approval_cutoff"] == 0.5
    assert row["model_version"] == "7"


def test_reports_unready_without_champion(client):
    http, app, _ = client
    app.state.champion = None
    response = http.get("/health")
    assert response.status_code == 503
    assert response.json()["model_loaded"] is False
    assert http.post("/predict", json=LOAN).status_code == 503
