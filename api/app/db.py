"""Write predictions and feedback to PostgreSQL."""

import os

from sqlalchemy import create_engine, text

_engine = None


def get_engine():
    global _engine
    url = os.getenv("DATABASE_URL")
    if not url:
        return None
    if _engine is None:
        _engine = create_engine(url, pool_pre_ping=True)
    return _engine


INSERT = text("""
    INSERT INTO predictions (revenue, dti_n, loan_amnt, fico_n, emp_length, purpose,
                             home_ownership_n, addr_state, default_probability, decision,
                             approval_cutoff, model_version, scenario)
    VALUES (:revenue, :dti_n, :loan_amnt, :fico_n, :emp_length, :purpose, :home_ownership_n,
            :addr_state, :default_probability, :decision, :approval_cutoff, :model_version,
            :scenario)
    RETURNING id
""")

UPDATE = text("UPDATE predictions SET actual = :actual, feedback_at = now() WHERE id = :id")


def log_prediction(features: dict, probability: float, decision: str, cutoff: float,
                   model_version: str, scenario: str) -> int | None:
    engine = get_engine()
    if engine is None:
        return None
    with engine.begin() as conn:
        row = conn.execute(INSERT, {**features, "default_probability": probability,
                                    "decision": decision, "approval_cutoff": cutoff,
                                    "model_version": model_version,
                                    "scenario": scenario}).one()
    return int(row.id)


def log_feedback(prediction_id: int, actual: int) -> bool:
    engine = get_engine()
    if engine is None:
        return False
    with engine.begin() as conn:
        result = conn.execute(UPDATE, {"id": prediction_id, "actual": actual})
    return result.rowcount == 1
