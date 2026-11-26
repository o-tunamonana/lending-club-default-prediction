from pathlib import Path

import pandas as pd
import pytest

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="session")
def train_df() -> pd.DataFrame:
    """A fixed 20,000-loan sample of the training file (keeps tests fast)."""
    return pd.read_csv(DATA / "train.csv", low_memory=False).sample(20_000, random_state=0)


@pytest.fixture(scope="session")
def test_df() -> pd.DataFrame:
    return pd.read_csv(DATA / "test.csv", low_memory=False).sample(5_000, random_state=0)
