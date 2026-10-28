"""
Shared preprocessing for the Lending Club default-prediction project (v2).

Every teammate imports from this file, so the whole team cleans the data
in exactly the same way:

    from preprocessing import load_split, build_preprocessor

    X_train, y_train = load_split("../data/train.csv")
    model = make_pipeline(build_preprocessor(), LogisticRegression())
    model.fit(X_train, y_train)

Requires scikit-learn >= 1.3 (for TargetEncoder).

Changes from v1, based on the EDA (notebook 02):
- removed `has_desc`: descriptions stop after 2014, so the flag is all 0 in the 2016 test set
- added `loan_to_income`: strongest numeric signal in the EDA (default 13% -> 27% across quintiles)
- `addr_state`: smoothed target encoding instead of 50 one-hot columns
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder,
                                   StandardScaler, TargetEncoder)

TARGET = "Default"

# Columns kept out of the model, and why (see README "Data preparation")
DROP_COLS = [
    "id",            # identifier only; no relationship with default
    "issue_d",       # used for the time split, not as a feature
    "experience_c",  # almost constant (5 zeros in 150k training loans)
    "title",         # free text duplicating `purpose`, inconsistent spelling
    "zip_code",      # 879 levels in train; also a proxy for protected attributes (fair lending)
    "desc",          # 86% empty and absent after 2014 -> would not exist in the 2016 test set
]

EMP_LENGTH_MAP = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
    "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
    "10+ years": 10,
    # "NI" (no information) is not in the map, so it becomes NaN;
    # the imputer then adds a "was missing" flag (NI borrowers default more)
}

DTI_PLACEHOLDER = 999   # undocumented code, treated as missing
DTI_CAP = 100           # debt payments above 100% of income are capped


def load_split(path):
    """Read train.csv or test.csv and return features X and target y."""
    df = pd.read_csv(path, low_memory=False)
    return df.drop(columns=[TARGET]), df[TARGET]


def clean(df):
    """
    Step 1 of the pipeline: fix data-entry quirks and build row-level features.

    Nothing here is *learned* from the data (no means, no medians), so this
    step cannot leak information from the test set.
    """
    df = df.copy()

    # emp_length: text -> number of years; "NI" -> NaN (imputed later, with a flag)
    df["emp_length"] = df["emp_length"].map(EMP_LENGTH_MAP)

    # dti_n: 999 is a placeholder -> NaN; extreme values capped
    df["dti_n"] = df["dti_n"].replace(DTI_PLACEHOLDER, np.nan).clip(upper=DTI_CAP)

    # loan size relative to income: how heavy the loan is for this borrower
    df["loan_to_income"] = df["loan_amnt"] / df["revenue"]

    return df.drop(columns=DROP_COLS, errors="ignore")


NUMERIC = ["fico_n", "dti_n", "emp_length"]
SKEWED = ["revenue", "loan_amnt", "loan_to_income"]   # long right tail -> log first
ONE_HOT = ["purpose", "home_ownership_n"]
TARGET_ENCODED = ["addr_state"]


STATE_SMOOTHING = 100   # a state's rate counts like 100 "virtual" loans at the overall rate


def _make_target_encoder():
    """Smoothed target encoder with reproducible cross-fitting (works on old and new sklearn)."""
    import sklearn
    from sklearn.model_selection import StratifiedKFold
    major, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
    if (major, minor) >= (1, 9):   # newer API: shuffling is set through the cv splitter
        return TargetEncoder(target_type="binary", smooth=STATE_SMOOTHING,
                             cv=StratifiedKFold(5, shuffle=True, random_state=42))
    return TargetEncoder(target_type="binary", smooth=STATE_SMOOTHING, random_state=42)


def build_preprocessor():
    """
    The full preprocessing pipeline (unfitted).

    clean  ->  ColumnTransformer:
        numeric        : fill missing with median (+ missing flag), standardize
        skewed         : log(1 + x), standardize
        one-hot        : purpose, home ownership; levels with < 500 rows grouped as "infrequent"
        target-encoded : state -> smoothed default rate, learned with internal cross-fitting
                         encoded = (n * state_rate + 100 * overall_rate) / (n + 100)
    """
    numeric = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
    )
    skewed = make_pipeline(
        FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
        StandardScaler(),
    )
    one_hot = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=500,
        sparse_output=False,
    )
    target_encoded = _make_target_encoder()

    columns = ColumnTransformer([
        ("num", numeric, NUMERIC),
        ("log", skewed, SKEWED),
        ("cat", one_hot, ONE_HOT),
        ("state", target_encoded, TARGET_ENCODED),
    ])

    return Pipeline([
        ("clean", FunctionTransformer(clean)),
        ("columns", columns),
    ])
