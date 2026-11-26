"""Data checks that run before training. A failed check stops the pipeline."""

import pandas as pd

from lcdp.schema import EMP_LENGTH_VALUES, REQUIRED_COLUMNS, TARGET


class DataValidationError(ValueError):
    """Raised when a data file is not safe to train on."""


def validate_training_data(df: pd.DataFrame, min_rows: int = 1000) -> dict:
    """Check a raw training frame and return a small summary for logging.

    Checks: required columns present, enough rows, target has no missing values
    and is binary, no duplicate loan ids, numeric fields are not negative and
    employment length uses known values.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")

    if len(df) < min_rows:
        raise DataValidationError(f"Only {len(df)} rows, need at least {min_rows}")

    if df[TARGET].isna().any():
        raise DataValidationError(f"Target '{TARGET}' has {int(df[TARGET].isna().sum())} missing values")

    target_values = set(df[TARGET].unique())
    if not target_values <= {0, 1}:
        raise DataValidationError(f"Target must be 0/1, found {sorted(target_values)}")
    if len(target_values) < 2:
        raise DataValidationError("Target has only one class")

    if df["id"].duplicated().any():
        raise DataValidationError(f"{int(df['id'].duplicated().sum())} duplicate loan ids")

    for col in ["revenue", "loan_amnt", "fico_n"]:
        if (df[col] < 0).any():
            raise DataValidationError(f"Negative values in '{col}'")

    unknown_emp = set(df["emp_length"].dropna().unique()) - set(EMP_LENGTH_VALUES)
    if unknown_emp:
        raise DataValidationError(f"Unknown emp_length values: {sorted(unknown_emp)}")

    return {
        "n_rows": int(len(df)),
        "default_rate": float(df[TARGET].mean()),
        "missing_share": float(df[REQUIRED_COLUMNS].isna().mean().mean()),
    }
