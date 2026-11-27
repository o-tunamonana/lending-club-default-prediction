"""FastAPI service for the registered champion model."""

import logging
import os
import time

import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app import db
from app.model_store import ModelStore
from app.schemas import Feedback, LoanApplication, Prediction
from lcdp.schema import FEATURES

log = logging.getLogger(__name__)
REQUESTS = Counter("api_requests_total", "HTTP requests", ["endpoint", "status"])
LATENCY = Histogram("api_request_latency_seconds", "Request latency", ["endpoint"])
ERRORS = Counter("api_errors_total", "HTTP errors", ["endpoint", "status", "kind"])
PREDICTIONS = Counter("model_predictions_total", "Predictions by decision", ["decision"])
PROBABILITY = Histogram("model_default_probability", "Predicted default probability",
                        buckets=[i / 20 for i in range(1, 21)])
MODEL_VERSION = Gauge("model_version_info", "Model version currently served", ["version"])
LOG_FAILURES = Counter("prediction_log_failures_total", "Predictions not written to Postgres")

store = ModelStore()
app = FastAPI(title="Lending Club Default Risk API", version="1.0.0")


@app.on_event("startup")
def load_model() -> None:
    local_uri = os.getenv("MODEL_URI")
    if local_uri:
        store.load_local(local_uri)
    else:
        store.start_background_refresh()


@app.middleware("http")
async def record_metrics(request: Request, call_next):
    endpoint = request.url.path
    if endpoint not in {"/predict", "/feedback", "/health"}:
        return await call_next(request)
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        LATENCY.labels(endpoint).observe(time.perf_counter() - start)
        REQUESTS.labels(endpoint, str(status)).inc()
        if status >= 400:
            kind = "server" if status >= 500 else "client"
            if status == 422:
                kind = "validation"
            ERRORS.labels(endpoint, str(status), kind).inc()


def classify(probability: float, cutoff: float) -> tuple[str, str]:
    if probability <= cutoff:
        return ("low" if probability <= 0.6 * cutoff else "medium"), "approve"
    if probability <= cutoff + 0.05:
        return "medium", "review"
    return "high", "reject"


@app.get("/health")
def health(response: Response):
    current = store.current
    engine = db.get_engine()
    database_ready = True
    if engine is not None:
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT approval_cutoff FROM predictions LIMIT 0"))
        except SQLAlchemyError:
            database_ready = False
    ready = current is not None and database_ready
    response.status_code = 200 if ready else 503
    return {"status": "ok" if ready else "loading", "model_loaded": ready,
            "model_version": current.version if current else None, "database_ready": database_ready}


@app.post("/predict", response_model=Prediction)
def predict(application: LoanApplication, x_traffic_scenario: str = Header(default="user")):
    current = store.current
    if current is None:
        raise HTTPException(503, "Model is not loaded yet, try again shortly")

    features = application.model_dump()
    frame = pd.DataFrame([features], columns=FEATURES)
    probability = float(current.model.predict_proba(frame)[0, 1])
    risk_level, decision = classify(probability, current.cutoff)

    prediction_id = None
    try:
        prediction_id = db.log_prediction(features, probability, decision, current.cutoff,
                                          current.version, x_traffic_scenario[:32])
    except SQLAlchemyError as exc:
        LOG_FAILURES.inc()
        log.exception("Could not log prediction")
        raise HTTPException(503, "Prediction store unavailable") from exc

    PREDICTIONS.labels(decision).inc()
    PROBABILITY.observe(probability)
    MODEL_VERSION.clear()
    MODEL_VERSION.labels(current.version).set(1)
    return Prediction(prediction_id=prediction_id, default_probability=probability,
                      risk_level=risk_level, decision=decision,
                      approval_cutoff=current.cutoff, model_version=current.version)


@app.post("/feedback")
def feedback(item: Feedback):
    try:
        updated = db.log_feedback(item.prediction_id, item.actual)
    except SQLAlchemyError as exc:
        log.exception("Could not log feedback")
        raise HTTPException(503, "Prediction store unavailable") from exc
    if not updated:
        raise HTTPException(404, "Unknown prediction_id")
    return {"status": "recorded"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
