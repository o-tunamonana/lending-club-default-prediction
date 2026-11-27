"""Register a small deterministic model for the isolated API integration stack."""
import os
import time

import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lcdp.schema import CHAMPION_ALIAS, REGISTERED_MODEL_NAME


def main():
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = MlflowClient()
    for _ in range(40):
        try:
            client.search_experiments()
            break
        except mlflow.exceptions.MlflowException:
            time.sleep(1)
    else:
        raise RuntimeError("MLflow did not become ready")

    pipeline = make_pipeline(
        ColumnTransformer([("fico", StandardScaler(), ["fico_n"])]),
        LogisticRegression(),
    )
    pipeline.fit(pd.DataFrame({"fico_n": [620, 650, 700, 740, 780, 810]}), [1, 1, 1, 0, 0, 0])
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(pipeline, artifact_path="model", pip_requirements=[])
    version = mlflow.register_model(f"runs:/{run.info.run_id}/model", REGISTERED_MODEL_NAME)
    client.set_model_version_tag(REGISTERED_MODEL_NAME, version.version, "approval_cutoff", "0.5")
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, version.version)


if __name__ == "__main__":
    main()
