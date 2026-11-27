"""Check probability bounds and directional changes in model risk."""

import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from lcdp.preprocessing import build_preprocessor
from lcdp.schema import FEATURES, TARGET


@pytest.fixture(scope="module")
def model_and_data(train_df):
    features, target = train_df[FEATURES], train_df[TARGET]
    x_train, x_valid, y_train, _ = train_test_split(features, target, stratify=target, random_state=0)
    model = make_pipeline(build_preprocessor(), HistGradientBoostingClassifier(random_state=0))
    model.fit(x_train, y_train)
    return model, x_valid.head(2000)


def test_probabilities_are_valid(model_and_data):
    model, features = model_and_data
    probability = model.predict_proba(features)[:, 1]
    assert ((probability >= 0) & (probability <= 1)).all()


def test_higher_fico_lowers_risk(model_and_data):
    model, features = model_and_data
    better = features.assign(fico_n=(features["fico_n"] + 60).clip(upper=850))
    assert model.predict_proba(better)[:, 1].mean() < model.predict_proba(features)[:, 1].mean()


def test_lower_income_raises_risk(model_and_data):
    model, features = model_and_data
    poorer = features.assign(revenue=features["revenue"] * 0.5)
    assert model.predict_proba(poorer)[:, 1].mean() > model.predict_proba(features)[:, 1].mean()
