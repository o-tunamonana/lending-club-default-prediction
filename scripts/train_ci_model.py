"""Train a small model and save an MLflow model folder for CI."""

import sys
from pathlib import Path

import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lcdp.preprocessing import build_preprocessor  # noqa: E402
from lcdp.schema import FEATURES, TARGET  # noqa: E402


def main(out_dir: str) -> None:
    df = pd.read_csv(ROOT / "data" / "train.csv", low_memory=False).sample(20_000, random_state=0)
    model = make_pipeline(build_preprocessor(), HistGradientBoostingClassifier(random_state=0))
    model.fit(df[FEATURES], df[TARGET])
    mlflow.sklearn.save_model(model, out_dir)
    print(f"Saved model to {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ci-model")
