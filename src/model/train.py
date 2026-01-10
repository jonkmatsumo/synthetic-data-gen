"""MLflow-enabled training pipeline for fraud detection model."""

import os
import tempfile
from datetime import UTC, datetime, timedelta

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.metrics import average_precision_score, precision_score, recall_score
from xgboost import XGBClassifier

from model.loader import DataLoader

# MLflow configuration from environment
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Set MLflow tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Experiment name
EXPERIMENT_NAME = "ach-fraud-detection"


def train_model(
    scale_pos_weight: float | None = None,
    max_depth: int = 6,
    training_window_days: int = 30,
    database_url: str | None = None,
) -> str:
    """Train an XGBoost model with MLflow tracking.

    Args:
        scale_pos_weight: Weight for positive class. If None, computed automatically
            from class imbalance ratio.
        max_depth: Maximum tree depth. Default 6.
        training_window_days: Number of days before today for training cutoff.
            Default 30.
        database_url: Optional database URL override.

    Returns:
        The MLflow run ID.
    """
    # Set up MLflow experiment
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Calculate training cutoff date
    training_cutoff_date = datetime.now(UTC) - timedelta(days=training_window_days)

    # Load data
    loader = DataLoader(database_url=database_url)
    split = loader.load_train_test_split(training_cutoff_date)

    # Handle empty dataset
    if split.train_size == 0:
        raise ValueError("No training data available. Generate data first.")

    if split.test_size == 0:
        raise ValueError("No test data available. Adjust training_window_days.")

    # Calculate scale_pos_weight if not provided
    if scale_pos_weight is None:
        n_negative = (split.y_train == 0).sum()
        n_positive = (split.y_train == 1).sum()
        if n_positive > 0:
            scale_pos_weight = n_negative / n_positive
        else:
            scale_pos_weight = 1.0

    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_params(
            {
                "scale_pos_weight": scale_pos_weight,
                "max_depth": max_depth,
                "training_window_days": training_window_days,
                "training_cutoff_date": training_cutoff_date.isoformat(),
                "train_size": split.train_size,
                "test_size": split.test_size,
                "train_fraud_rate": split.train_fraud_rate,
                "test_fraud_rate": split.test_fraud_rate,
            }
        )

        # Train XGBoost model
        clf = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            max_depth=max_depth,
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        )

        clf.fit(split.X_train, split.y_train)

        # Generate predictions
        y_pred = clf.predict(split.X_test)
        y_pred_proba = clf.predict_proba(split.X_test)[:, 1]

        # Calculate metrics
        precision = precision_score(split.y_test, y_pred, zero_division=0)
        recall = recall_score(split.y_test, y_pred, zero_division=0)
        pr_auc = average_precision_score(split.y_test, y_pred_proba)

        # Log metrics
        mlflow.log_metrics(
            {
                "precision": precision,
                "recall": recall,
                "pr_auc": pr_auc,
            }
        )

        # Log the model with signature and input example
        signature = infer_signature(split.X_train, y_pred_proba)
        mlflow.sklearn.log_model(
            clf,
            "model",
            signature=signature,
            input_example=split.X_train.iloc[:1],
        )

        # Save and log reference data (X_test) for drift detection
        with tempfile.TemporaryDirectory() as tmpdir:
            reference_path = os.path.join(tmpdir, "reference_data.parquet")
            split.X_test.to_parquet(reference_path, index=False)
            mlflow.log_artifact(reference_path)

        # Register the model
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, EXPERIMENT_NAME)

        return run.info.run_id


def get_latest_model_version(model_name: str = EXPERIMENT_NAME) -> int | None:
    """Get the latest version number of a registered model.

    Args:
        model_name: Name of the registered model.

    Returns:
        Latest version number, or None if model doesn't exist.
    """
    client = mlflow.MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if versions:
            return max(int(v.version) for v in versions)
    except mlflow.exceptions.MlflowException:
        pass
    return None


def load_production_model(model_name: str = EXPERIMENT_NAME):
    """Load the latest version of the production model.

    Args:
        model_name: Name of the registered model.

    Returns:
        Loaded model object.

    Raises:
        ValueError: If no model versions exist.
    """
    version = get_latest_model_version(model_name)
    if version is None:
        raise ValueError(f"No model versions found for '{model_name}'")

    model_uri = f"models:/{model_name}/{version}"
    return mlflow.sklearn.load_model(model_uri)


if __name__ == "__main__":
    import sys

    # Allow overriding training window from command line
    window_days = int(sys.argv[1]) if len(sys.argv) > 1 else 30

    print(f"Training model with {window_days} day window...")
    run_id = train_model(training_window_days=window_days)
    print(f"Training complete. Run ID: {run_id}")
