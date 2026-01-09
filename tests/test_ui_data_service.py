"""Tests for UI data service layer."""

from unittest.mock import MagicMock, patch

import pandas as pd
import requests


class TestGetDbEngine:
    """Tests for database engine initialization."""

    def test_get_db_engine_returns_engine(self):
        # Import here to avoid module-level side effects
        from ui.data_service import get_db_engine

        engine = get_db_engine()
        assert engine is not None

    def test_get_db_engine_singleton(self):
        from ui.data_service import get_db_engine

        engine1 = get_db_engine()
        engine2 = get_db_engine()
        assert engine1 is engine2


class TestGetDbConnection:
    """Tests for database connection context manager."""

    def test_connection_context_manager(self):
        from ui.data_service import get_db_connection

        with patch("ui.data_service.get_db_engine") as mock_engine:
            mock_conn = MagicMock()
            mock_engine.return_value.connect.return_value = mock_conn

            with get_db_connection() as conn:
                assert conn is mock_conn

            mock_conn.close.assert_called_once()


class TestFetchDailyStats:
    """Tests for fetch_daily_stats function."""

    def test_returns_dataframe(self):
        from ui.data_service import fetch_daily_stats

        with patch("ui.data_service.get_db_connection") as mock_ctx:
            mock_conn = MagicMock()
            mock_result = MagicMock()
            mock_result.fetchall.return_value = []
            mock_result.keys.return_value = [
                "date",
                "total_transactions",
                "fraud_count",
                "fraud_rate",
                "total_amount",
                "avg_z_score",
            ]
            mock_conn.execute.return_value = mock_result
            mock_ctx.return_value.__enter__.return_value = mock_conn

            result = fetch_daily_stats(days=7)

            assert isinstance(result, pd.DataFrame)

    def test_handles_database_error(self):
        from sqlalchemy.exc import SQLAlchemyError

        from ui.data_service import fetch_daily_stats

        with patch("ui.data_service.get_db_connection") as mock_ctx:
            mock_ctx.return_value.__enter__.side_effect = SQLAlchemyError("DB error")

            result = fetch_daily_stats()

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0


class TestFetchRecentAlerts:
    """Tests for fetch_recent_alerts function."""

    def test_returns_dataframe(self):
        from ui.data_service import fetch_recent_alerts

        with patch("ui.data_service.get_db_connection") as mock_ctx:
            mock_conn = MagicMock()
            mock_result = MagicMock()
            mock_result.fetchall.return_value = []
            mock_result.keys.return_value = [
                "record_id",
                "user_id",
                "created_at",
                "amount",
                "is_fraudulent",
                "fraud_type",
                "merchant_risk_score",
                "velocity_24h",
                "amount_to_avg_ratio_30d",
                "balance_volatility_z_score",
                "computed_risk_score",
            ]
            mock_conn.execute.return_value = mock_result
            mock_ctx.return_value.__enter__.return_value = mock_conn

            result = fetch_recent_alerts(limit=10)

            assert isinstance(result, pd.DataFrame)

    def test_respects_limit_parameter(self):
        from ui.data_service import fetch_recent_alerts

        with patch("ui.data_service.get_db_connection") as mock_ctx:
            mock_conn = MagicMock()
            mock_result = MagicMock()
            mock_result.fetchall.return_value = []
            mock_result.keys.return_value = [
                "record_id",
                "user_id",
                "created_at",
                "amount",
                "is_fraudulent",
                "fraud_type",
                "merchant_risk_score",
                "velocity_24h",
                "amount_to_avg_ratio_30d",
                "balance_volatility_z_score",
                "computed_risk_score",
            ]
            mock_conn.execute.return_value = mock_result
            mock_ctx.return_value.__enter__.return_value = mock_conn

            fetch_recent_alerts(limit=25)

            # Verify limit was passed to query
            call_args = mock_conn.execute.call_args
            assert call_args[0][1]["limit"] == 25


class TestFetchFraudSummary:
    """Tests for fetch_fraud_summary function."""

    def test_returns_dict(self):
        from ui.data_service import fetch_fraud_summary

        with patch("ui.data_service.get_db_connection") as mock_ctx:
            mock_conn = MagicMock()
            mock_result = MagicMock()
            mock_row = MagicMock()
            mock_row.total_transactions = 100
            mock_row.total_fraud = 5
            mock_row.fraud_rate = 5.0
            mock_row.total_amount = 10000.0
            mock_row.fraud_amount = 500.0
            mock_result.fetchone.return_value = mock_row
            mock_conn.execute.return_value = mock_result
            mock_ctx.return_value.__enter__.return_value = mock_conn

            result = fetch_fraud_summary()

            assert isinstance(result, dict)
            assert result["total_transactions"] == 100
            assert result["total_fraud"] == 5
            assert result["fraud_rate"] == 5.0

    def test_handles_empty_database(self):
        from ui.data_service import fetch_fraud_summary

        with patch("ui.data_service.get_db_connection") as mock_ctx:
            mock_conn = MagicMock()
            mock_result = MagicMock()
            mock_result.fetchone.return_value = None
            mock_conn.execute.return_value = mock_result
            mock_ctx.return_value.__enter__.return_value = mock_conn

            result = fetch_fraud_summary()

            assert result["total_transactions"] == 0
            assert result["fraud_rate"] == 0.0


class TestPredictRisk:
    """Tests for predict_risk API client."""

    def test_successful_request(self):
        from ui.data_service import predict_risk

        with patch("ui.data_service.requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "request_id": "req_123",
                "score": 75,
                "risk_components": [],
                "model_version": "v1.0.0",
            }
            mock_response.raise_for_status = MagicMock()
            mock_post.return_value = mock_response

            result = predict_risk("user_123", 100.0)

            assert result is not None
            assert result["score"] == 75
            assert result["request_id"] == "req_123"

    def test_handles_timeout(self):
        from ui.data_service import predict_risk

        with patch("ui.data_service.requests.post") as mock_post:
            mock_post.side_effect = requests.Timeout("Connection timed out")

            result = predict_risk("user_123", 100.0)

            assert result is None

    def test_handles_connection_error(self):
        from ui.data_service import predict_risk

        with patch("ui.data_service.requests.post") as mock_post:
            mock_post.side_effect = requests.ConnectionError("Connection refused")

            result = predict_risk("user_123", 100.0)

            assert result is None

    def test_handles_http_error(self):
        from ui.data_service import predict_risk

        with patch("ui.data_service.requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.raise_for_status.side_effect = requests.HTTPError(
                "500 Server Error"
            )
            mock_post.return_value = mock_response

            result = predict_risk("user_123", 100.0)

            assert result is None

    def test_sends_correct_payload(self):
        from ui.data_service import predict_risk

        with patch("ui.data_service.requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {"score": 50}
            mock_response.raise_for_status = MagicMock()
            mock_post.return_value = mock_response

            predict_risk("user_abc", 250.50, currency="EUR", client_txn_id="txn_xyz")

            call_args = mock_post.call_args
            payload = call_args[1]["json"]

            assert payload["user_id"] == "user_abc"
            assert payload["amount"] == 250.50
            assert payload["currency"] == "EUR"
            assert payload["client_transaction_id"] == "txn_xyz"

    def test_auto_generates_transaction_id(self):
        from ui.data_service import predict_risk

        with patch("ui.data_service.requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {"score": 50}
            mock_response.raise_for_status = MagicMock()
            mock_post.return_value = mock_response

            predict_risk("user_abc", 100.0)

            call_args = mock_post.call_args
            payload = call_args[1]["json"]

            assert payload["client_transaction_id"].startswith("ui_txn_")


class TestCheckApiHealth:
    """Tests for check_api_health function."""

    def test_healthy_api(self):
        from ui.data_service import check_api_health

        with patch("ui.data_service.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "status": "healthy",
                "model_loaded": True,
                "version": "v1.0.0",
            }
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            result = check_api_health()

            assert result is not None
            assert result["status"] == "healthy"

    def test_unhealthy_api(self):
        from ui.data_service import check_api_health

        with patch("ui.data_service.requests.get") as mock_get:
            mock_get.side_effect = requests.ConnectionError("Connection refused")

            result = check_api_health()

            assert result is None
