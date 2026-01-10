"""Tests for FastAPI signal evaluation API."""

from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.schemas import Currency, SignalRequest
from api.services import (
    AMOUNT_RATIO_HIGH_THRESHOLD,
    CONNECTION_BURST_THRESHOLD,
    VELOCITY_HIGH_THRESHOLD,
    FeatureVector,
    SignalEvaluator,
    get_evaluator,
)


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def evaluator():
    """Create signal evaluator."""
    return SignalEvaluator()


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data

    def test_health_status_healthy(self, client):
        response = client.get("/health")
        data = response.json()

        assert data["status"] == "healthy"
        # model_loaded is False in test environment (no MLflow/MinIO)
        assert isinstance(data["model_loaded"], bool)


class TestSignalEndpoint:
    """Tests for signal evaluation endpoint."""

    def test_evaluate_returns_200(self, client):
        response = client.post(
            "/evaluate/signal",
            json={
                "user_id": "user_123",
                "amount": 100.00,
                "currency": "USD",
                "client_transaction_id": "txn_abc",
            },
        )
        assert response.status_code == 200

    def test_evaluate_response_structure(self, client):
        response = client.post(
            "/evaluate/signal",
            json={
                "user_id": "user_123",
                "amount": 100.00,
                "currency": "USD",
                "client_transaction_id": "txn_abc",
            },
        )
        data = response.json()

        assert "request_id" in data
        assert "score" in data
        assert "risk_components" in data
        assert "model_version" in data

    def test_evaluate_score_in_range(self, client):
        response = client.post(
            "/evaluate/signal",
            json={
                "user_id": "user_456",
                "amount": 500.00,
                "currency": "USD",
                "client_transaction_id": "txn_def",
            },
        )
        data = response.json()

        assert 1 <= data["score"] <= 99

    def test_evaluate_request_id_format(self, client):
        response = client.post(
            "/evaluate/signal",
            json={
                "user_id": "user_789",
                "amount": 250.00,
                "currency": "EUR",
                "client_transaction_id": "txn_ghi",
            },
        )
        data = response.json()

        assert data["request_id"].startswith("req_")
        assert len(data["request_id"]) == 16  # "req_" + 12 hex chars

    def test_evaluate_model_version_present(self, client):
        response = client.post(
            "/evaluate/signal",
            json={
                "user_id": "user_test",
                "amount": 100.00,
                "currency": "USD",
                "client_transaction_id": "txn_test",
            },
        )
        data = response.json()

        assert data["model_version"] == "v1.0.0"

    def test_evaluate_risk_components_structure(self, client):
        response = client.post(
            "/evaluate/signal",
            json={
                "user_id": "user_risky",
                "amount": 1000.00,
                "currency": "USD",
                "client_transaction_id": "txn_risky",
            },
        )
        data = response.json()

        for component in data["risk_components"]:
            assert "key" in component
            assert "label" in component

    def test_evaluate_idempotent_same_user(self, client):
        """Same user should get consistent scoring."""
        request_data = {
            "user_id": "user_consistent",
            "amount": 200.00,
            "currency": "USD",
            "client_transaction_id": "txn_1",
        }

        response1 = client.post("/evaluate/signal", json=request_data)
        response2 = client.post("/evaluate/signal", json=request_data)

        # Score should be the same for same user
        assert response1.json()["score"] == response2.json()["score"]

        # Risk components should be the same
        assert (
            response1.json()["risk_components"] == response2.json()["risk_components"]
        )

        # Request IDs should be different (each request gets new ID)
        assert response1.json()["request_id"] != response2.json()["request_id"]


class TestSignalValidation:
    """Tests for request validation."""

    def test_missing_user_id(self, client):
        response = client.post(
            "/evaluate/signal",
            json={
                "amount": 100.00,
                "currency": "USD",
                "client_transaction_id": "txn_abc",
            },
        )
        assert response.status_code == 422

    def test_missing_amount(self, client):
        response = client.post(
            "/evaluate/signal",
            json={
                "user_id": "user_123",
                "currency": "USD",
                "client_transaction_id": "txn_abc",
            },
        )
        assert response.status_code == 422

    def test_negative_amount(self, client):
        response = client.post(
            "/evaluate/signal",
            json={
                "user_id": "user_123",
                "amount": -100.00,
                "currency": "USD",
                "client_transaction_id": "txn_abc",
            },
        )
        assert response.status_code == 422

    def test_zero_amount(self, client):
        response = client.post(
            "/evaluate/signal",
            json={
                "user_id": "user_123",
                "amount": 0,
                "currency": "USD",
                "client_transaction_id": "txn_abc",
            },
        )
        assert response.status_code == 422

    def test_invalid_currency(self, client):
        response = client.post(
            "/evaluate/signal",
            json={
                "user_id": "user_123",
                "amount": 100.00,
                "currency": "INVALID",
                "client_transaction_id": "txn_abc",
            },
        )
        assert response.status_code == 422

    def test_missing_transaction_id(self, client):
        response = client.post(
            "/evaluate/signal",
            json={
                "user_id": "user_123",
                "amount": 100.00,
                "currency": "USD",
            },
        )
        assert response.status_code == 422


class TestSignalEvaluator:
    """Tests for SignalEvaluator service."""

    def test_evaluate_returns_response(self, evaluator):
        request = SignalRequest(
            user_id="user_test",
            amount=Decimal("100.00"),
            currency=Currency.USD,
            client_transaction_id="txn_test",
        )

        response = evaluator.evaluate(request)

        assert response.request_id.startswith("req_")
        assert 1 <= response.score <= 99
        assert response.model_version == "v1.0.0"

    def test_probability_calculation_base(self, evaluator):
        """Low-risk features should give low probability."""
        features = FeatureVector(
            velocity_24h=1,
            amount_to_avg_ratio_30d=1.0,
            balance_volatility_z_score=0.0,
            bank_connections_24h=1,
            merchant_risk_score=20,
            has_history=True,
        )

        prob = evaluator._calculate_probability(features)
        assert prob < 0.1

    def test_probability_high_velocity(self, evaluator):
        """High velocity should increase probability."""
        low_velocity = FeatureVector(velocity_24h=1, has_history=True)
        high_velocity = FeatureVector(velocity_24h=10, has_history=True)

        prob_low = evaluator._calculate_probability(low_velocity)
        prob_high = evaluator._calculate_probability(high_velocity)

        assert prob_high > prob_low

    def test_probability_high_amount_ratio(self, evaluator):
        """High amount ratio should increase probability."""
        normal = FeatureVector(amount_to_avg_ratio_30d=1.0, has_history=True)
        high = FeatureVector(amount_to_avg_ratio_30d=5.0, has_history=True)

        prob_normal = evaluator._calculate_probability(normal)
        prob_high = evaluator._calculate_probability(high)

        assert prob_high > prob_normal

    def test_probability_no_history(self, evaluator):
        """No history should increase probability."""
        with_history = FeatureVector(has_history=True)
        no_history = FeatureVector(has_history=False)

        prob_with = evaluator._calculate_probability(with_history)
        prob_without = evaluator._calculate_probability(no_history)

        assert prob_without > prob_with

    def test_probability_capped_at_099(self, evaluator):
        """Probability should be capped at 0.99."""
        extreme_features = FeatureVector(
            velocity_24h=100,
            amount_to_avg_ratio_30d=20.0,
            balance_volatility_z_score=-5.0,
            bank_connections_24h=20,
            merchant_risk_score=100,
            has_history=False,
        )

        prob = evaluator._calculate_probability(extreme_features)
        assert prob <= 0.99


class TestRiskComponents:
    """Tests for risk component identification."""

    def test_velocity_component(self, evaluator):
        features = FeatureVector(
            velocity_24h=VELOCITY_HIGH_THRESHOLD + 1,
            has_history=True,
        )

        components = evaluator._identify_risk_components(features)
        keys = [c.key for c in components]

        assert "velocity" in keys

    def test_amount_ratio_component(self, evaluator):
        features = FeatureVector(
            amount_to_avg_ratio_30d=AMOUNT_RATIO_HIGH_THRESHOLD + 1,
            has_history=True,
        )

        components = evaluator._identify_risk_components(features)
        keys = [c.key for c in components]

        assert "amount_ratio" in keys

    def test_connection_component(self, evaluator):
        features = FeatureVector(
            bank_connections_24h=CONNECTION_BURST_THRESHOLD + 1,
            has_history=True,
        )

        components = evaluator._identify_risk_components(features)
        keys = [c.key for c in components]

        assert "connections" in keys

    def test_history_component(self, evaluator):
        features = FeatureVector(has_history=False)

        components = evaluator._identify_risk_components(features)
        keys = [c.key for c in components]

        assert "history" in keys

    def test_no_components_low_risk(self, evaluator):
        """Low-risk features should have no components."""
        features = FeatureVector(
            velocity_24h=1,
            amount_to_avg_ratio_30d=1.0,
            balance_volatility_z_score=0.0,
            bank_connections_24h=1,
            merchant_risk_score=20,
            has_history=True,
        )

        components = evaluator._identify_risk_components(features)
        assert len(components) == 0


class TestGetEvaluator:
    """Tests for evaluator singleton."""

    def test_returns_evaluator(self):
        evaluator = get_evaluator()
        assert isinstance(evaluator, SignalEvaluator)

    def test_returns_same_instance(self):
        evaluator1 = get_evaluator()
        evaluator2 = get_evaluator()
        assert evaluator1 is evaluator2
