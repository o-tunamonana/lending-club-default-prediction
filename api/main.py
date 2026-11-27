"""Serve the registered Lending Club champion and log predictions."""
import asyncio
import logging
import os
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from time import perf_counter
from typing import Annotated, Literal

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from mlflow import MlflowClient
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, Info, generate_latest
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from lcdp.schema import (
    CHAMPION_ALIAS,
    EMP_LENGTH_VALUES,
    FEATURES,
    HOME_OWNERSHIP_VALUES,
    PURPOSE_VALUES,
    REGISTERED_MODEL_NAME,
    US_STATES,
)

log = logging.getLogger(__name__)
ALLOWED = {
    "emp_length": EMP_LENGTH_VALUES,
    "purpose": PURPOSE_VALUES,
    "home_ownership_n": HOME_OWNERSHIP_VALUES,
    "addr_state": US_STATES,
}


class Loan(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)

    revenue: float = Field(gt=0)
    dti_n: float | None = Field(ge=0)
    loan_amnt: float = Field(gt=0)
    fico_n: float | None = Field(ge=300, le=850)
    emp_length: str
    purpose: str
    home_ownership_n: str
    addr_state: str

    @field_validator("emp_length", "purpose", "home_ownership_n", "addr_state")
    @classmethod
    def validate_category(cls, value, info):
        if value not in ALLOWED[info.field_name]:
            raise ValueError(f"Choose one of: {', '.join(ALLOWED[info.field_name])}")
        return value


class Feedback(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_id: int = Field(gt=0, le=2**63 - 1, strict=True)
    actual: int = Field(ge=0, le=1, strict=True)


@dataclass(frozen=True)
class Champion:
    pipeline: object
    version: str
    cutoff: float

    def score(self, features):
        row = pd.DataFrame([features], columns=FEATURES)
        for column in ("dti_n", "fico_n"):
            row[column] = row[column].astype(float)
        positive_class = list(self.pipeline.classes_).index(1)
        probability = float(self.pipeline.predict_proba(row)[0, positive_class])
        if not 0 <= probability <= 1:
            raise ValueError("Model returned an invalid probability")
        return probability


def load_champion(client, current):
    registered = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
    version = str(registered.version)
    if current is not None and current.version == version:
        return current

    cutoff = float(registered.tags["approval_cutoff"])
    if not 0 <= cutoff <= 1:
        raise ValueError("Invalid approval_cutoff model tag")
    pipeline = mlflow.sklearn.load_model(
        f"models:/{REGISTERED_MODEL_NAME}/{version}",
    )
    if 1 not in pipeline.classes_:
        raise ValueError("Champion has no Default = 1 class")
    log.info("Loaded champion version %s", version)
    return Champion(pipeline, version, cutoff)


INSERT_PREDICTION = text(
    f"INSERT INTO predictions ({', '.join(FEATURES)}, "
    "default_probability, decision, model_version, scenario, approval_cutoff) "
    f"VALUES ({', '.join(':' + field for field in FEATURES)}, "
    ":default_probability, :decision, :model_version, :scenario, :approval_cutoff) RETURNING id"
)
UPDATE_FEEDBACK = text(
    "UPDATE predictions SET actual = :actual, feedback_at = COALESCE(feedback_at, now()) "
    "WHERE id = :prediction_id AND (actual IS NULL OR actual = :actual) RETURNING id"
)


def create_app():
    registry = CollectorRegistry()
    requests = Counter("api_requests", "API requests", ["endpoint", "status"], registry=registry)
    errors = Counter(
        "api_errors", "API responses with status >= 400", ["endpoint", "status", "kind"], registry=registry,
    )
    latency = Histogram(
        "api_request_latency_seconds", "API request latency", ["endpoint"], registry=registry,
    )
    model_version = Info("model_version", "Loaded champion model version", registry=registry)
    decisions = Counter("model_predictions", "Served decisions", ["decision"], registry=registry)
    probabilities = Histogram(
        "model_default_probability", "Served default probabilities", registry=registry,
        buckets=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0),
    )

    @asynccontextmanager
    async def lifespan(app):
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)
        interval = float(os.getenv("MODEL_RELOAD_SECONDS", "30"))
        if interval <= 0:
            raise ValueError("MODEL_RELOAD_SECONDS must be positive")
        engine = create_engine(
            os.environ["DATABASE_URL"], pool_pre_ping=True, connect_args={"connect_timeout": 3},
        )
        app.state.engine = engine
        app.state.champion = None

        async def refresh():
            while True:
                try:
                    current = app.state.champion
                    loaded = await asyncio.to_thread(load_champion, client, current)
                    if loaded is not current:
                        app.state.champion = loaded
                        model_version.info({"version": loaded.version})
                except Exception:
                    log.warning("Could not refresh the MLflow champion", exc_info=True)
                await asyncio.sleep(interval)

        task = asyncio.create_task(refresh())
        try:
            yield
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            engine.dispose()

    app = FastAPI(title="Lending Club Model API", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        details = [
            {key: item[key] for key in ("loc", "msg", "type")}
            for item in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": details})

    @app.middleware("http")
    async def record_metrics(request: Request, call_next):
        endpoint = request.url.path
        if endpoint not in {"/predict", "/feedback", "/health"}:
            return await call_next(request)
        start = perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            requests.labels(endpoint, str(status)).inc()
            latency.labels(endpoint).observe(perf_counter() - start)
            if status >= 400:
                kind = "server" if status >= 500 else "client"
                if status == 422:
                    kind = "validation"
                errors.labels(endpoint, str(status), kind).inc()

    @app.get("/health")
    def health(response: Response):
        champion = app.state.champion
        try:
            with app.state.engine.connect() as conn:
                conn.execute(text("SELECT id FROM predictions LIMIT 0"))
            database_ready = True
        except SQLAlchemyError:
            database_ready = False
        ready = champion is not None and database_ready
        response.status_code = 200 if ready else 503
        return {
            "status": "ok" if ready else "unavailable",
            "model_loaded": ready,
            "model_version": champion.version if champion is not None else None,
            "database_ready": database_ready,
        }

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(registry), headers={"Content-Type": CONTENT_TYPE_LATEST})

    @app.post("/predict")
    def predict(
        loan: Loan,
        scenario: Annotated[
            Literal["normal", "drift", "frontend"], Header(alias="X-Traffic-Scenario")
        ] = "frontend",
    ):
        champion = app.state.champion
        if champion is None:
            raise HTTPException(503, "Champion is not loaded yet")
        features = loan.model_dump()
        probability = champion.score(features)
        decision = "approve" if probability <= champion.cutoff else "reject"
        risk_level = "low" if probability <= champion.cutoff / 2 else "medium"
        if decision == "reject":
            risk_level = "high"
        try:
            with app.state.engine.begin() as conn:
                prediction_id = conn.execute(INSERT_PREDICTION, {
                    **features,
                    "default_probability": probability,
                    "decision": decision,
                    "model_version": champion.version,
                    "scenario": scenario,
                    "approval_cutoff": champion.cutoff,
                }).scalar_one()
        except SQLAlchemyError as exc:
            log.exception("Could not log prediction")
            raise HTTPException(503, "Prediction storage is unavailable") from exc
        decisions.labels(decision).inc()
        probabilities.observe(probability)
        return {
            "prediction_id": prediction_id,
            "default_probability": probability,
            "decision": decision,
            "approval_cutoff": champion.cutoff,
            "risk_level": risk_level,
            "model_version": champion.version,
        }

    @app.post("/feedback")
    def feedback(label: Feedback):
        try:
            with app.state.engine.begin() as conn:
                saved = conn.execute(UPDATE_FEEDBACK, label.model_dump()).scalar_one_or_none()
                if saved is None:
                    exists = conn.execute(
                        text("SELECT id FROM predictions WHERE id = :id"), {"id": label.prediction_id},
                    ).scalar_one_or_none()
                    if exists is None:
                        raise HTTPException(404, "Prediction not found")
                    raise HTTPException(409, "A different outcome is already recorded")
        except SQLAlchemyError as exc:
            log.exception("Could not log feedback")
            raise HTTPException(503, "Prediction storage is unavailable") from exc
        return {"prediction_id": saved, "actual": label.actual}

    return app


app = create_app()
