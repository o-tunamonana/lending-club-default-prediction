"""Request and response models for the scoring API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from lcdp.schema import EMP_LENGTH_VALUES, HOME_OWNERSHIP_VALUES, PURPOSE_VALUES, US_STATES

EmpLength = Literal[tuple(EMP_LENGTH_VALUES)]
Purpose = Literal[tuple(PURPOSE_VALUES)]
HomeOwnership = Literal[tuple(HOME_OWNERSHIP_VALUES)]


class LoanApplication(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "revenue": 65000, "dti_n": 18.5, "loan_amnt": 12000, "fico_n": 702,
        "emp_length": "5 years", "purpose": "debt_consolidation",
        "home_ownership_n": "RENT", "addr_state": "NY",
    }})

    revenue: float = Field(gt=0, le=10_000_000, description="Annual income (USD)")
    dti_n: float = Field(ge=0, le=999, description="Debt-to-income ratio (%)")
    loan_amnt: float = Field(gt=0, le=100_000, description="Requested amount (USD)")
    fico_n: float = Field(ge=300, le=850, description="FICO score")
    emp_length: EmpLength
    purpose: Purpose
    home_ownership_n: HomeOwnership
    addr_state: str = Field(min_length=2, max_length=2)

    @field_validator("addr_state")
    @classmethod
    def known_state(cls, value: str) -> str:
        value = value.upper()
        if value not in US_STATES:
            raise ValueError("must be a US state code, e.g. NY")
        return value


class Prediction(BaseModel):
    prediction_id: int | None
    default_probability: float
    risk_level: Literal["low", "medium", "high"]
    decision: Literal["approve", "review", "reject"]
    approval_cutoff: float
    model_version: str


class Feedback(BaseModel):
    prediction_id: int
    actual: Literal[0, 1]
