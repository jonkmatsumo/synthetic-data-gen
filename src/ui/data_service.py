"""Data service layer for the dashboard.

Provides read-only database access and API client for the Streamlit UI.
This module is isolated from backend dependencies - uses raw SQL queries
instead of ORM models to avoid coupling.
"""

import os
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import requests
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

# Configuration from environment
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://synthetic:synthetic_dev_password@localhost:5432/synthetic_data",
)
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# API timeout in seconds
API_TIMEOUT = 2.0

# Risk score threshold for alerts
ALERT_THRESHOLD = 80

# Module-level engine (lazy initialization)
_engine: Engine | None = None


def get_db_engine() -> Engine:
    """Get or create the database engine.

    Returns:
        SQLAlchemy Engine instance.
    """
    global _engine
    if _engine is None:
        _engine = create_engine(
            DATABASE_URL,
            pool_size=3,
            max_overflow=5,
            pool_pre_ping=True,
        )
    return _engine


@contextmanager
def get_db_connection():
    """Get a database connection context manager.

    Yields:
        Database connection that auto-closes on exit.

    Example:
        with get_db_connection() as conn:
            result = conn.execute(text("SELECT 1"))
    """
    engine = get_db_engine()
    connection = engine.connect()
    try:
        yield connection
    finally:
        connection.close()


def fetch_daily_stats(days: int = 30) -> pd.DataFrame:
    """Fetch daily transaction statistics.

    Joins evaluation_metadata with feature_snapshots and generated_records
    to get daily aggregates of transactions, fraud rates, and amounts.

    Args:
        days: Number of days to look back. Default 30.

    Returns:
        DataFrame with columns:
        - date: Transaction date
        - total_transactions: Count of transactions
        - fraud_count: Count of fraudulent transactions
        - fraud_rate: Percentage of fraud
        - total_amount: Sum of transaction amounts
        - avg_risk_score: Average risk score (from balance_volatility_z_score)
    """
    cutoff_date = datetime.now(UTC) - timedelta(days=days)

    query = text("""
        SELECT
            DATE(em.created_at) as date,
            COUNT(*) as total_transactions,
            SUM(CASE WHEN gr.is_fraudulent THEN 1 ELSE 0 END) as fraud_count,
            ROUND(
                100.0 * SUM(CASE WHEN gr.is_fraudulent THEN 1 ELSE 0 END) / COUNT(*),
                2
            ) as fraud_rate,
            COALESCE(SUM(gr.amount), 0) as total_amount,
            ROUND(AVG(fs.balance_volatility_z_score)::numeric, 2) as avg_z_score
        FROM evaluation_metadata em
        LEFT JOIN generated_records gr ON em.record_id = gr.record_id
        LEFT JOIN feature_snapshots fs ON em.record_id = fs.record_id
        WHERE em.created_at >= :cutoff_date
        GROUP BY DATE(em.created_at)
        ORDER BY date DESC
    """)

    try:
        with get_db_connection() as conn:
            result = conn.execute(query, {"cutoff_date": cutoff_date})
            rows = result.fetchall()
            columns = result.keys()
            return pd.DataFrame(rows, columns=list(columns))
    except SQLAlchemyError as e:
        print(f"Database error in fetch_daily_stats: {e}")
        return pd.DataFrame()


def fetch_transaction_details(days: int = 7) -> pd.DataFrame:
    """Fetch individual transaction details.

    Joins evaluation_metadata with feature_snapshots and generated_records
    to get transaction-level data for analysis.

    Args:
        days: Number of days to look back. Default 7.

    Returns:
        DataFrame with transaction details including features and labels.
    """
    cutoff_date = datetime.now(UTC) - timedelta(days=days)

    query = text("""
        SELECT
            em.record_id,
            em.user_id,
            em.created_at,
            em.is_train_eligible,
            em.is_pre_fraud,
            gr.amount,
            gr.is_fraudulent,
            gr.fraud_type,
            gr.is_off_hours_txn,
            gr.merchant_risk_score,
            fs.velocity_24h,
            fs.amount_to_avg_ratio_30d,
            fs.balance_volatility_z_score
        FROM evaluation_metadata em
        LEFT JOIN generated_records gr ON em.record_id = gr.record_id
        LEFT JOIN feature_snapshots fs ON em.record_id = fs.record_id
        WHERE em.created_at >= :cutoff_date
        ORDER BY em.created_at DESC
        LIMIT 1000
    """)

    try:
        with get_db_connection() as conn:
            result = conn.execute(query, {"cutoff_date": cutoff_date})
            rows = result.fetchall()
            columns = result.keys()
            return pd.DataFrame(rows, columns=list(columns))
    except SQLAlchemyError as e:
        print(f"Database error in fetch_transaction_details: {e}")
        return pd.DataFrame()


def fetch_recent_alerts(limit: int = 50) -> pd.DataFrame:
    """Fetch recent high-risk transactions.

    Selects records where computed risk indicators exceed thresholds,
    simulating alerts that would be generated by a real system.

    High risk is determined by:
    - High velocity (velocity_24h > 5)
    - Unusual amount ratio (amount_to_avg_ratio_30d > 3.0)
    - Low balance volatility z-score (< -2.0)
    - High merchant risk score (> 70)

    Args:
        limit: Maximum number of alerts to return. Default 50.

    Returns:
        DataFrame with high-risk transaction details.
    """
    query = text("""
        SELECT
            em.record_id,
            em.user_id,
            em.created_at,
            gr.amount,
            gr.is_fraudulent,
            gr.fraud_type,
            gr.merchant_risk_score,
            fs.velocity_24h,
            fs.amount_to_avg_ratio_30d,
            fs.balance_volatility_z_score,
            -- Compute a simple risk score for display
            CASE
                WHEN fs.velocity_24h > 5 THEN 20 ELSE 0
            END +
            CASE
                WHEN fs.amount_to_avg_ratio_30d > 3.0 THEN 25 ELSE 0
            END +
            CASE
                WHEN fs.balance_volatility_z_score < -2.0 THEN 20 ELSE 0
            END +
            CASE
                WHEN gr.merchant_risk_score > 70 THEN 20 ELSE 0
            END +
            CASE
                WHEN gr.is_off_hours_txn THEN 15 ELSE 0
            END as computed_risk_score
        FROM evaluation_metadata em
        INNER JOIN generated_records gr ON em.record_id = gr.record_id
        INNER JOIN feature_snapshots fs ON em.record_id = fs.record_id
        WHERE
            fs.velocity_24h > 5
            OR fs.amount_to_avg_ratio_30d > 3.0
            OR fs.balance_volatility_z_score < -2.0
            OR gr.merchant_risk_score > 70
        ORDER BY em.created_at DESC
        LIMIT :limit
    """)

    try:
        with get_db_connection() as conn:
            result = conn.execute(query, {"limit": limit})
            rows = result.fetchall()
            columns = result.keys()
            df = pd.DataFrame(rows, columns=list(columns))

            # Filter to only high-risk (score > threshold)
            if len(df) > 0 and "computed_risk_score" in df.columns:
                df = df[df["computed_risk_score"] >= ALERT_THRESHOLD]

            return df
    except SQLAlchemyError as e:
        print(f"Database error in fetch_recent_alerts: {e}")
        return pd.DataFrame()


def fetch_fraud_summary() -> dict[str, Any]:
    """Fetch summary statistics for fraud metrics.

    Returns:
        Dictionary with summary statistics:
        - total_transactions: Total transaction count
        - total_fraud: Total fraud count
        - fraud_rate: Overall fraud rate percentage
        - total_amount: Sum of all amounts
        - fraud_amount: Sum of fraudulent amounts
    """
    query = text("""
        SELECT
            COUNT(*) as total_transactions,
            SUM(CASE WHEN is_fraudulent THEN 1 ELSE 0 END) as total_fraud,
            ROUND(
                100.0 * SUM(CASE WHEN is_fraudulent THEN 1 ELSE 0 END) /
                NULLIF(COUNT(*), 0),
                2
            ) as fraud_rate,
            COALESCE(SUM(amount), 0) as total_amount,
            COALESCE(
                SUM(CASE WHEN is_fraudulent THEN amount ELSE 0 END),
                0
            ) as fraud_amount
        FROM generated_records
    """)

    try:
        with get_db_connection() as conn:
            result = conn.execute(query)
            row = result.fetchone()
            if row:
                return {
                    "total_transactions": row.total_transactions or 0,
                    "total_fraud": row.total_fraud or 0,
                    "fraud_rate": float(row.fraud_rate or 0),
                    "total_amount": float(row.total_amount or 0),
                    "fraud_amount": float(row.fraud_amount or 0),
                }
    except SQLAlchemyError as e:
        print(f"Database error in fetch_fraud_summary: {e}")

    return {
        "total_transactions": 0,
        "total_fraud": 0,
        "fraud_rate": 0.0,
        "total_amount": 0.0,
        "fraud_amount": 0.0,
    }


# --- API Client ---


def predict_risk(
    user_id: str,
    amount: float,
    currency: str = "USD",
    client_txn_id: str | None = None,
) -> dict[str, Any] | None:
    """Send a transaction for risk evaluation via the API.

    Args:
        user_id: User identifier.
        amount: Transaction amount.
        currency: Currency code (default: USD).
        client_txn_id: Optional client transaction ID. Auto-generated if None.

    Returns:
        API response dictionary with score and risk_components,
        or None if the request fails.

    Example response:
        {
            "request_id": "req_abc123",
            "score": 85,
            "risk_components": [
                {"key": "velocity", "label": "high_transaction_velocity"}
            ],
            "model_version": "v1.0.0"
        }
    """
    if client_txn_id is None:
        client_txn_id = f"ui_txn_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"

    url = f"{API_BASE_URL}/evaluate/signal"
    payload = {
        "user_id": user_id,
        "amount": amount,
        "currency": currency,
        "client_transaction_id": client_txn_id,
    }

    try:
        response = requests.post(url, json=payload, timeout=API_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.Timeout:
        print(f"API timeout after {API_TIMEOUT}s for user {user_id}")
        return None
    except requests.ConnectionError:
        print(f"API connection error: Could not connect to {url}")
        return None
    except requests.HTTPError as e:
        print(f"API HTTP error: {e}")
        return None
    except requests.RequestException as e:
        print(f"API request error: {e}")
        return None


def check_api_health() -> dict[str, Any] | None:
    """Check the API health status.

    Returns:
        Health check response or None if unavailable.
    """
    url = f"{API_BASE_URL}/health"

    try:
        response = requests.get(url, timeout=API_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None
