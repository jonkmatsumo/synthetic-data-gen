"""MLflow utilities for the dashboard.

Provides helper functions for interacting with MLflow tracking server
and model registry. Keeps UI code clean by encapsulating MLflow client logic.
"""

import os
from typing import Any

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

# MLflow configuration from environment
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# Set tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Experiment name (must match train.py)
EXPERIMENT_NAME = "ach-fraud-detection"


def get_client() -> MlflowClient:
    """Get an MLflow client instance.

    Returns:
        MlflowClient configured with tracking URI.
    """
    return MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


def get_experiment_runs(
    experiment_name: str = EXPERIMENT_NAME,
    max_results: int = 100,
) -> pd.DataFrame:
    """Fetch experiment runs sorted by PR-AUC.

    Args:
        experiment_name: Name of the MLflow experiment.
        max_results: Maximum number of runs to return.

    Returns:
        DataFrame with run information, sorted by metrics.pr_auc descending.
        Empty DataFrame if experiment doesn't exist or has no runs.
    """
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            return pd.DataFrame()

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=max_results,
            order_by=["metrics.pr_auc DESC"],
        )

        if runs.empty:
            return pd.DataFrame()

        # Select and rename relevant columns
        columns_map = {
            "run_id": "Run ID",
            "start_time": "Started",
            "params.max_depth": "Max Depth",
            "params.training_window_days": "Window (days)",
            "params.train_size": "Train Size",
            "metrics.precision": "Precision",
            "metrics.recall": "Recall",
            "metrics.pr_auc": "PR-AUC",
        }

        available_cols = [c for c in columns_map.keys() if c in runs.columns]
        result = runs[available_cols].copy()
        result = result.rename(columns={k: v for k, v in columns_map.items()})

        return result

    except MlflowException:
        return pd.DataFrame()


def get_model_versions(
    model_name: str = EXPERIMENT_NAME,
) -> list[dict[str, Any]]:
    """Get all versions of a registered model.

    Args:
        model_name: Name of the registered model.

    Returns:
        List of version dictionaries with version, stage, and run_id.
    """
    client = get_client()
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        return [
            {
                "version": v.version,
                "stage": v.current_stage,
                "run_id": v.run_id,
                "created": v.creation_timestamp,
            }
            for v in versions
        ]
    except MlflowException:
        return []


def get_production_model_version(model_name: str = EXPERIMENT_NAME) -> str | None:
    """Get the version number of the current production model.

    Args:
        model_name: Name of the registered model.

    Returns:
        Version string if a production model exists, None otherwise.
    """
    versions = get_model_versions(model_name)
    for v in versions:
        if v["stage"] == "Production":
            return v["version"]
    return None


def promote_to_production(
    run_id: str,
    model_name: str = EXPERIMENT_NAME,
) -> dict[str, Any]:
    """Promote a model run to production stage.

    This finds the model version associated with the run_id and transitions
    it to the Production stage. Any existing production model is archived.

    Args:
        run_id: The MLflow run ID to promote.
        model_name: Name of the registered model.

    Returns:
        Dictionary with success status and message.
    """
    client = get_client()

    try:
        # Find the version associated with this run
        versions = client.search_model_versions(f"name='{model_name}'")
        target_version = None

        for v in versions:
            if v.run_id == run_id:
                target_version = v.version
                break

        if target_version is None:
            return {
                "success": False,
                "message": f"No model version found for run {run_id}",
            }

        # Archive current production model if exists
        for v in versions:
            if v.current_stage == "Production":
                client.transition_model_version_stage(
                    name=model_name,
                    version=v.version,
                    stage="Archived",
                )

        # Promote the target version to production
        client.transition_model_version_stage(
            name=model_name,
            version=target_version,
            stage="Production",
        )

        return {
            "success": True,
            "message": f"Model version {target_version} promoted to Production",
            "version": target_version,
        }

    except MlflowException as e:
        return {
            "success": False,
            "message": f"MLflow error: {e}",
        }


def check_mlflow_connection() -> bool:
    """Check if MLflow tracking server is accessible.

    Returns:
        True if connection successful, False otherwise.
    """
    try:
        client = get_client()
        # Try to list experiments as a connection test
        client.search_experiments(max_results=1)
        return True
    except Exception:
        return False
