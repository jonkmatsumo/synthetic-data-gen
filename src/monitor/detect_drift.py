"""Drift detection monitor using Population Stability Index (PSI).

Compares reference data (from MLflow model artifacts) with live data
(from feature_snapshots) to detect feature distribution drift.

Usage:
    uv run python src/monitor/detect_drift.py
    uv run python src/monitor/detect_drift.py --hours 48 --threshold 0.25
"""

import argparse
import logging
import os
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sqlalchemy import create_engine, text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration from environment
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://synthetic:synthetic_dev_password@localhost:5432/synthetic_data",
)

# Model registry name
MODEL_NAME = "ach-fraud-detection"

# Features to monitor for drift
MONITORED_FEATURES = [
    "velocity_24h",
    "amount_to_avg_ratio_30d",
    "balance_volatility_z_score",
]

# PSI thresholds
PSI_THRESHOLD_WARNING = 0.1  # Slight drift
PSI_THRESHOLD_CRITICAL = 0.2  # Significant drift requiring action


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    buckettype: str = "bins",
    buckets: int = 10,
) -> float:
    """Calculate Population Stability Index (PSI) between two distributions.

    PSI measures how much a distribution has shifted over time.
    - PSI < 0.1: No significant change
    - 0.1 <= PSI < 0.2: Slight change, monitor
    - PSI >= 0.2: Significant change, action required

    Args:
        expected: Reference/baseline distribution (training data).
        actual: Current/live distribution (production data).
        buckettype: How to create buckets - 'bins' for equal width,
            'quantiles' for equal frequency.
        buckets: Number of buckets to use.

    Returns:
        PSI value (float). Higher values indicate more drift.
    """
    # Remove NaN values
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        logger.warning("Empty array provided for PSI calculation")
        return 0.0

    # Create bucket boundaries
    if buckettype == "bins":
        # Equal-width bins based on expected distribution
        min_val = min(expected.min(), actual.min())
        max_val = max(expected.max(), actual.max())
        breakpoints = np.linspace(min_val, max_val, buckets + 1)
    elif buckettype == "quantiles":
        # Equal-frequency bins based on expected distribution
        breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
        # Ensure unique breakpoints
        breakpoints = np.unique(breakpoints)
    else:
        raise ValueError(f"Unknown buckettype: {buckettype}")

    # Calculate frequencies for each bucket
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    # Convert to proportions
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    # Avoid division by zero - use small epsilon
    epsilon = 1e-6
    expected_pct = np.clip(expected_pct, epsilon, 1)
    actual_pct = np.clip(actual_pct, epsilon, 1)

    # Calculate PSI for each bucket and sum
    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    psi = np.sum(psi_values)

    return float(psi)


def get_reference_data() -> pd.DataFrame | None:
    """Load reference data from the production model's artifacts.

    Downloads reference_data.parquet from MLflow artifacts storage (MinIO).

    Returns:
        DataFrame with reference features, or None if unavailable.
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()

        # Get the production model version
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        production_version = None

        for v in versions:
            if v.current_stage == "Production":
                production_version = v
                break

        if production_version is None:
            logger.error("No production model found in registry")
            return None

        run_id = production_version.run_id
        logger.info(
            f"Found production model: version={production_version.version}, "
            f"run_id={run_id}"
        )

        # Download the reference data artifact
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path="reference_data.parquet",
                dst_path=tmpdir,
            )
            df_reference = pd.read_parquet(artifact_path)

        logger.info(f"Loaded reference data: {len(df_reference)} records")
        return df_reference

    except Exception as e:
        logger.error(f"Failed to load reference data: {e}")
        return None


def get_live_data(hours: int = 24) -> pd.DataFrame:
    """Load live data from feature_snapshots table.

    Args:
        hours: Number of hours to look back. Default 24.

    Returns:
        DataFrame with current feature data.
    """
    cutoff_time = datetime.now(UTC) - timedelta(hours=hours)

    query = text("""
        SELECT
            fs.velocity_24h,
            fs.amount_to_avg_ratio_30d,
            fs.balance_volatility_z_score,
            fs.computed_at
        FROM feature_snapshots fs
        WHERE fs.computed_at >= :cutoff_time
        ORDER BY fs.computed_at DESC
    """)

    try:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            result = conn.execute(query, {"cutoff_time": cutoff_time})
            rows = result.fetchall()
            columns = result.keys()

        df_current = pd.DataFrame(rows, columns=list(columns))
        logger.info(f"Loaded live data: {len(df_current)} records from last {hours}h")
        return df_current

    except Exception as e:
        logger.error(f"Failed to load live data: {e}")
        return pd.DataFrame()


def detect_drift(
    hours: int = 24,
    threshold: float = PSI_THRESHOLD_CRITICAL,
) -> dict[str, Any]:
    """Run drift detection comparing reference vs live data.

    Args:
        hours: Hours of live data to analyze.
        threshold: PSI threshold for drift detection.

    Returns:
        Dictionary with drift detection results.
    """
    results = {
        "timestamp": datetime.now(UTC).isoformat(),
        "hours_analyzed": hours,
        "threshold": threshold,
        "reference_size": 0,
        "live_size": 0,
        "features": {},
        "drift_detected": False,
        "drifted_features": [],
    }

    # Load reference data
    df_reference = get_reference_data()
    if df_reference is None or len(df_reference) == 0:
        logger.error("Cannot perform drift detection: no reference data available")
        results["error"] = "No reference data available"
        return results

    results["reference_size"] = len(df_reference)

    # Load live data
    df_current = get_live_data(hours=hours)
    if len(df_current) == 0:
        logger.warning("No live data found for drift analysis")
        results["error"] = "No live data available"
        return results

    results["live_size"] = len(df_current)

    # Calculate PSI for each monitored feature
    for feature in MONITORED_FEATURES:
        if feature not in df_reference.columns:
            logger.warning(f"Feature '{feature}' not in reference data, skipping")
            continue

        if feature not in df_current.columns:
            logger.warning(f"Feature '{feature}' not in live data, skipping")
            continue

        expected = df_reference[feature].values.astype(float)
        actual = df_current[feature].values.astype(float)

        psi = calculate_psi(expected, actual, buckettype="quantiles", buckets=10)

        # Determine status
        if psi >= PSI_THRESHOLD_CRITICAL:
            status = "CRITICAL"
        elif psi >= PSI_THRESHOLD_WARNING:
            status = "WARNING"
        else:
            status = "OK"

        results["features"][feature] = {
            "psi": round(psi, 4),
            "status": status,
        }

        # Log result
        log_msg = f"Feature '{feature}': PSI={psi:.4f} [{status}]"
        if status == "CRITICAL":
            logger.critical(log_msg)
            results["drift_detected"] = True
            results["drifted_features"].append(feature)
        elif status == "WARNING":
            logger.warning(log_msg)
        else:
            logger.info(log_msg)

    # Final summary
    if results["drift_detected"]:
        logger.critical(f"DRIFT DETECTED in features: {results['drifted_features']}")
    else:
        logger.info("No significant drift detected")

    return results


def main() -> int:
    """CLI entry point for drift detection.

    Returns:
        Exit code: 0 for success, 1 for drift detected, 2 for error.
    """
    parser = argparse.ArgumentParser(
        description="Detect feature drift between reference and live data"
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Hours of live data to analyze (default: 24)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=PSI_THRESHOLD_CRITICAL,
        help=f"PSI threshold for drift detection (default: {PSI_THRESHOLD_CRITICAL})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    args = parser.parse_args()

    logger.info(
        f"Starting drift detection (hours={args.hours}, threshold={args.threshold})"
    )

    results = detect_drift(hours=args.hours, threshold=args.threshold)

    if args.json:
        import json

        print(json.dumps(results, indent=2))

    if "error" in results:
        return 2

    if results["drift_detected"]:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
