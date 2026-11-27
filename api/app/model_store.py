"""Load the champion model and refresh it as the registry alias moves."""

import logging
import os
import threading
import time
from dataclasses import dataclass

from lcdp.schema import CHAMPION_ALIAS, REGISTERED_MODEL_NAME

log = logging.getLogger(__name__)
DEFAULT_CUTOFF = 0.27


@dataclass
class LoadedModel:
    model: object
    version: str
    cutoff: float


class ModelStore:
    def __init__(self):
        self._current: LoadedModel | None = None
        self._lock = threading.Lock()
        self.reload_seconds = int(os.getenv("RELOAD_SECONDS", "30"))

    @property
    def current(self) -> LoadedModel | None:
        with self._lock:
            return self._current

    def set(self, loaded: LoadedModel) -> None:
        with self._lock:
            self._current = loaded

    def load_local(self, uri: str) -> None:
        import mlflow.sklearn

        cutoff = float(os.getenv("APPROVAL_CUTOFF", DEFAULT_CUTOFF))
        self.set(LoadedModel(mlflow.sklearn.load_model(uri), "local", cutoff))

    def refresh_from_registry(self) -> bool:
        import mlflow.sklearn
        from mlflow import MlflowClient

        version = MlflowClient().get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
        current = self.current
        if current is not None and current.version == version.version:
            return False
        model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}/{version.version}")
        cutoff = float(version.tags.get("approval_cutoff", DEFAULT_CUTOFF))
        self.set(LoadedModel(model, version.version, cutoff))
        log.info("Loaded model version %s (cutoff %.4f)", version.version, cutoff)
        return True

    def start_background_refresh(self) -> None:
        def loop():
            while True:
                try:
                    self.refresh_from_registry()
                except Exception as exc:
                    log.warning("Model refresh failed: %s", exc)
                time.sleep(self.reload_seconds)

        threading.Thread(target=loop, daemon=True, name="model-refresh").start()
