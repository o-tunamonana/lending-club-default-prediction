import numpy as np
import pandas as pd

from lcdp.preprocessing import build_preprocessor, clean

ROW = {
    "id": 1, "issue_d": "2015-01-01", "revenue": 50_000.0, "dti_n": 999.0, "loan_amnt": 10_000,
    "fico_n": 700.0, "experience_c": 1, "emp_length": "NI", "purpose": "credit_card",
    "home_ownership_n": "RENT", "addr_state": "NY", "zip_code": "100xx", "title": "x", "desc": None,
}


def test_clean_known_input_gives_expected_features():
    out = clean(pd.DataFrame([ROW]))
    assert out.loc[0, "loan_to_income"] == 0.2           # 10,000 / 50,000
    assert np.isnan(out.loc[0, "dti_n"])                  # 999 placeholder -> missing
    assert np.isnan(out.loc[0, "emp_length"])             # "NI" -> missing
    assert not {"id", "issue_d", "zip_code", "title", "desc", "experience_c"} & set(out.columns)


def test_emp_length_and_dti_cap():
    rows = pd.DataFrame([{**ROW, "emp_length": "10+ years", "dti_n": 150.0},
                         {**ROW, "emp_length": "< 1 year", "dti_n": 20.0}])
    out = clean(rows)
    assert out["emp_length"].tolist() == [10, 0]
    assert out["dti_n"].tolist() == [100.0, 20.0]


def test_pipeline_output_has_no_missing_values(train_df):
    X, y = train_df.drop(columns=["Default"]), train_df["Default"]
    Xt = build_preprocessor().fit_transform(X, y)
    assert Xt.shape[0] == len(X)
    assert not np.isnan(Xt).any()


def test_pipeline_handles_unseen_category(train_df):
    X, y = train_df.drop(columns=["Default"]), train_df["Default"]
    pre = build_preprocessor().fit(X, y)
    new = pd.DataFrame([{**ROW, "purpose": "brand_new_purpose", "addr_state": "ZZ"}])
    assert not np.isnan(pre.transform(new)).any()
