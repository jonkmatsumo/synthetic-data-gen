"""Model manager for dynamic model loading from MLflow/MinIO.

Provides a singleton ModelManager that handles:
- Loading production models from MLflow model registry
- Fallback to local model if MLflow/MinIO is unavailable
- Thread-safe model access
"""

import logging
import os
import pickle
from pathlib import Path
from threading import Lock
from typing import Any

import mlflow
import numpy as np
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)

# MLflow configuration from environment
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# Model registry name
MODEL_NAME = "ach-fraud-detection"

# Fallback model path
FALLBACK_MODEL_PATH = Path(__file__).parent.parent / "model" / "fallback_model.pkl"


class ModelManager:
    """Singleton manager for ML model loading and inference.

    Handles loading models from MLflow registry with fallback to local model
    if the registry is unavailable.
    """

    _instance = None
    _lock = Lock()

    def __new__(cls) -> "ModelManager":
        """Create singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the model manager."""
        if self._initialized:
            return

        self._model = None
        self._model_version: str | None = None
        self._model_source: str = "none"
        self._initialized = True

    @property
    def model_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self._model is not None

    @property
    def model_version(self) -> str:
        """Get the current model version."""
        return self._model_version or "unknown"

    @property
    def model_source(self) -> str:
        """Get the model source (mlflow, fallback, or none)."""
        return self._model_source

    def load_production_model(self) -> bool:
        """Load the production model from MLflow registry.

        Attempts to load from MLflow first. If that fails, falls back to
        a local pickle file if available.

        Returns:
            True if a model was loaded successfully, False otherwise.
        """
        # Try loading from MLflow first
        if self._load_from_mlflow():
            return True

        # Fall back to local model
        if self._load_fallback_model():
            return True

        logger.error("No model available - both MLflow and fallback failed")
        return False

    def _load_from_mlflow(self) -> bool:
        """Attempt to load model from MLflow registry.

        Returns:
            True if successful, False otherwise.
        """
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

            model_uri = f"models:/{MODEL_NAME}/Production"
            logger.info(f"Loading model from MLflow: {model_uri}")

            self._model = mlflow.pyfunc.load_model(model_uri)
            self._model_version = self._get_production_version()
            self._model_source = "mlflow"

            logger.info(
                f"Successfully loaded model version {self._model_version} from MLflow"
            )
            return True

        except Exception as e:
            logger.critical(
                f"Failed to load model from MLflow/MinIO: {e}. "
                "Attempting fallback to local model."
            )
            return False

    def _load_fallback_model(self) -> bool:
        """Attempt to load fallback model from local file.

        Returns:
            True if successful, False otherwise.
        """
        if not FALLBACK_MODEL_PATH.exists():
            logger.warning(f"Fallback model not found at {FALLBACK_MODEL_PATH}")
            return False

        try:
            logger.info(f"Loading fallback model from {FALLBACK_MODEL_PATH}")

            with open(FALLBACK_MODEL_PATH, "rb") as f:
                self._model = pickle.load(f)

            self._model_version = "fallback"
            self._model_source = "fallback"

            logger.warning(
                "Using fallback model - MLflow registry was unavailable. "
                "Model predictions may be outdated."
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load fallback model: {e}")
            return False

    def _get_production_version(self) -> str:
        """Get the version number of the production model.

        Returns:
            Version string or 'unknown'.
        """
        try:
            client = mlflow.MlflowClient()
            versions = client.search_model_versions(f"name='{MODEL_NAME}'")
            for v in versions:
                if v.current_stage == "Production":
                    return f"v{v.version}"
        except Exception:
            pass
        return "unknown"

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Generate predictions using the loaded model.

        Args:
            features: DataFrame with feature columns matching model input.

        Returns:
            Array of prediction probabilities.

        Raises:
            RuntimeError: If no model is loaded.
        """
        if self._model is None:
            raise RuntimeError("No model loaded. Call load_production_model() first.")

        try:
            # MLflow pyfunc models return predictions directly
            if self._model_source == "mlflow":
                predictions = self._model.predict(features)
            else:
                # Fallback sklearn model - get probability of positive class
                predictions = self._model.predict_proba(features)[:, 1]

            return np.asarray(predictions)

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

    def predict_single(self, features: dict[str, Any]) -> float:
        """Generate prediction for a single observation.

        Args:
            features: Dictionary of feature name -> value.

        Returns:
            Prediction probability for positive class.

        Raises:
            RuntimeError: If no model is loaded.
        """
        df = pd.DataFrame([features])
        predictions = self.predict(df)
        return float(predictions[0])


# Module-level singleton instance
_manager: ModelManager | None = None


def get_model_manager() -> ModelManager:
    """Get the singleton ModelManager instance.

    Returns:
        The ModelManager singleton.
    """
    global _manager
    if _manager is None:
        _manager = ModelManager()
    return _manager
