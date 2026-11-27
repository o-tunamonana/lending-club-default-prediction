-- Runs once when the Postgres volume is created.

-- One row per prediction served by the API; `actual` arrives later through /feedback.
CREATE TABLE IF NOT EXISTS predictions (
    id                  BIGSERIAL PRIMARY KEY,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    revenue             DOUBLE PRECISION,
    dti_n               DOUBLE PRECISION,
    loan_amnt           DOUBLE PRECISION,
    fico_n              DOUBLE PRECISION,
    emp_length          TEXT,
    purpose             TEXT,
    home_ownership_n    TEXT,
    addr_state          TEXT,
    default_probability DOUBLE PRECISION NOT NULL,
    decision            TEXT NOT NULL,
    approval_cutoff     DOUBLE PRECISION,   -- cutoff the model version used for this decision
    model_version       TEXT,
    scenario            TEXT,
    actual              SMALLINT,
    feedback_at         TIMESTAMPTZ
);
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS approval_cutoff DOUBLE PRECISION;
CREATE INDEX IF NOT EXISTS predictions_created_at_idx ON predictions (created_at);

-- One row per Evidently run over the latest window of predictions.
CREATE TABLE IF NOT EXISTS drift_metrics (
    id                  BIGSERIAL PRIMARY KEY,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    window_size         INTEGER,
    n_labelled          INTEGER,
    dataset_drift       BOOLEAN,
    drift_share         DOUBLE PRECISION,
    n_drifted_features  INTEGER,
    mean_prediction     DOUBLE PRECISION,
    actual_default_rate DOUBLE PRECISION,
    share_drift_traffic DOUBLE PRECISION,
    roc_auc             DOUBLE PRECISION,
    accuracy            DOUBLE PRECISION,
    "precision"         DOUBLE PRECISION,
    recall              DOUBLE PRECISION,
    f1                  DOUBLE PRECISION
);

-- Per-feature drift scores for each Evidently run.
CREATE TABLE IF NOT EXISTS feature_drift (
    id          BIGSERIAL PRIMARY KEY,
    metric_id   BIGINT REFERENCES drift_metrics (id) ON DELETE CASCADE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    feature     TEXT NOT NULL,
    drift_score DOUBLE PRECISION,
    drifted     BOOLEAN,
    stattest    TEXT
);

-- Retraining runs started by the monitor.
CREATE TABLE IF NOT EXISTS retraining_events (
    id          BIGSERIAL PRIMARY KEY,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    reason      TEXT,
    drift_share DOUBLE PRECISION,
    flow_run_id TEXT,
    status      TEXT
);
