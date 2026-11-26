"""Column names and allowed values shared by training, the API and monitoring."""

TARGET = "Default"

# Raw application fields the model needs at prediction time (what a user enters).
FEATURES = [
    "revenue",
    "dti_n",
    "loan_amnt",
    "fico_n",
    "emp_length",
    "purpose",
    "home_ownership_n",
    "addr_state",
]

NUMERIC_FEATURES = ["revenue", "dti_n", "loan_amnt", "fico_n"]
CATEGORICAL_FEATURES = ["emp_length", "purpose", "home_ownership_n", "addr_state"]

# Columns that must exist in a raw data file before training.
REQUIRED_COLUMNS = ["id", "issue_d", *FEATURES, TARGET]

EMP_LENGTH_VALUES = [
    "< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years",
    "6 years", "7 years", "8 years", "9 years", "10+ years", "NI",
]

PURPOSE_VALUES = [
    "debt_consolidation", "credit_card", "home_improvement", "other",
    "major_purchase", "small_business", "car", "medical", "moving",
    "vacation", "house", "wedding", "renewable_energy", "educational",
]

HOME_OWNERSHIP_VALUES = ["MORTGAGE", "RENT", "OWN", "OTHER"]

US_STATES = [
    "AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "HI",
    "IA", "ID", "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MI", "MN",
    "MO", "MS", "MT", "NC", "ND", "NE", "NH", "NJ", "NM", "NV", "NY", "OH",
    "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VA", "VT", "WA",
    "WI", "WV", "WY",
]

REGISTERED_MODEL_NAME = "lending-club-default"
CHAMPION_ALIAS = "champion"
