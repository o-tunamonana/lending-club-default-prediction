import numpy as np
import pytest

from lcdp.schema import REQUIRED_COLUMNS
from lcdp.validation import DataValidationError, validate_training_data


def test_real_training_data_passes(train_df):
    summary = validate_training_data(train_df)
    assert summary["n_rows"] == len(train_df)
    assert 0.1 < summary["default_rate"] < 0.3


@pytest.mark.parametrize("column", REQUIRED_COLUMNS)
def test_missing_required_column_fails(train_df, column):
    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate_training_data(train_df.drop(columns=[column]))


def test_missing_target_fails(train_df):
    df = train_df.copy()
    df.loc[df.index[:5], "Default"] = np.nan
    with pytest.raises(DataValidationError, match="missing values"):
        validate_training_data(df)


def test_non_binary_target_fails(train_df):
    df = train_df.copy()
    df.loc[df.index[0], "Default"] = 2
    with pytest.raises(DataValidationError, match="must be 0/1"):
        validate_training_data(df)


def test_duplicate_ids_fail(train_df):
    df = train_df.copy()
    df.loc[df.index[1], "id"] = df.loc[df.index[0], "id"]
    with pytest.raises(DataValidationError, match="duplicate"):
        validate_training_data(df)


def test_unknown_employment_length_fails(train_df):
    df = train_df.copy()
    df.loc[df.index[0], "emp_length"] = "eleven years"
    with pytest.raises(DataValidationError, match="emp_length"):
        validate_training_data(df)
