"""Business logic for signal evaluation."""

import logging
import os
import uuid
from dataclasses import dataclass, field
from decimal import Decimal

import numpy as np
import pandas as pd
from sqlalchemy import text

from api.schemas import RiskComponent, SignalRequest, SignalResponse
from model.evaluate import ScoreCalibrator
from synthetic_pipeline.db.session import DatabaseSession

logger = logging.getLogger(__name__)

# Feature thresholds for risk component detection (based on percentiles)
VELOCITY_HIGH_THRESHOLD = 5  # 24h transaction count threshold
AMOUNT_RATIO_HIGH_THRESHOLD = 3.0  # Amount vs 30d avg threshold
BALANCE_VOLATILITY_THRESHOLD = -2.0  # Z-score threshold (negative = low balance)
MERCHANT_RISK_THRESHOLD = 70  # Merchant risk score threshold
CONNECTION_BURST_THRESHOLD = 4  # 24h bank connections threshold

MODEL_VERSION = "v1.0.0"


@dataclass
class FeatureVector:
    """Container for user features used in scoring."""

    velocity_24h: int = 0
    amount_to_avg_ratio_30d: float = 1.0
    balance_volatility_z_score: float = 0.0
    bank_connections_24h: int = 0
    merchant_risk_score: int = 0
    has_history: bool = True
    transaction_amount: Decimal = Decimal("0")


@dataclass
class SignalEvaluator:
    """Evaluates fraud signals for transactions.

    This service is idempotent - it only assesses risk without modifying
    any transaction state.

    Uses the trained ML model when available, falls back to rule-based scoring.
    """

    calibrator: ScoreCalibrator = field(default_factory=ScoreCalibrator)
    model_version: str = MODEL_VERSION
    db_session: DatabaseSession | None = field(default=None)

    def __post_init__(self):
        """Initialize database session."""
        if self.db_session is None:
            self.db_session = DatabaseSession()

    def evaluate(self, request: SignalRequest) -> SignalResponse:
        """Evaluate fraud signal for a transaction.

        Args:
            request: The signal evaluation request.

        Returns:
            SignalResponse with score and risk components.
        """
        from api.model_manager import get_model_manager

        # Generate unique request ID
        request_id = f"req_{uuid.uuid4().hex[:12]}"

        # Fetch features for the user from database
        features = self._fetch_features(request)

        # Get model manager
        manager = get_model_manager()

        # Use ML model if available and features were found in DB
        if manager.model_loaded and features.has_history:
            raw_probability = self._predict_with_model(manager, features)
            model_version = manager.model_version
        else:
            # Fall back to rule-based scoring
            raw_probability = self._calculate_probability(features)
            model_version = self.model_version

        # Calibrate to 1-99 score
        score = self._calibrate_score(raw_probability)

        # Identify risk components based on feature values
        risk_components = self._identify_risk_components(features)

        return SignalResponse(
            request_id=request_id,
            score=score,
            risk_components=risk_components,
            model_version=model_version,
        )

    def _predict_with_model(self, manager, features: FeatureVector) -> float:
        """Use the ML model for prediction.

        Args:
            manager: ModelManager with loaded model.
            features: Feature vector from database.

        Returns:
            Probability of fraud.
        """
        try:
            feature_dict = {
                "velocity_24h": features.velocity_24h,
                "amount_to_avg_ratio_30d": features.amount_to_avg_ratio_30d,
                "balance_volatility_z_score": features.balance_volatility_z_score,
            }
            probability = manager.predict_single(feature_dict)
            logger.debug(f"ML model prediction: {probability}")
            return float(probability)
        except Exception as e:
            logger.warning(f"ML prediction failed, falling back to rules: {e}")
            return self._calculate_probability(features)

    def _fetch_features(self, request: SignalRequest) -> FeatureVector:
        """Fetch features for the user from feature store.

        Queries the feature_snapshots table for the most recent features
        for the given user. Falls back to simulated features if not found.

        Args:
            request: The signal request containing user_id.

        Returns:
            FeatureVector with user features.
        """
        try:
            with self.db_session.get_session() as session:
                # Get most recent feature snapshot for this user
                query = text("""
                    SELECT
                        velocity_24h,
                        amount_to_avg_ratio_30d,
                        balance_volatility_z_score
                    FROM feature_snapshots
                    WHERE user_id = :user_id
                    ORDER BY computed_at DESC
                    LIMIT 1
                """)
                result = session.execute(query, {"user_id": request.user_id})
                row = result.fetchone()

                if row is not None:
                    logger.debug(f"Found features for user {request.user_id}")
                    return FeatureVector(
                        velocity_24h=int(row.velocity_24h),
                        amount_to_avg_ratio_30d=float(row.amount_to_avg_ratio_30d),
                        balance_volatility_z_score=float(row.balance_volatility_z_score),
                        bank_connections_24h=0,  # Not in feature_snapshots yet
                        merchant_risk_score=0,  # Not in feature_snapshots yet
                        has_history=True,
                        transaction_amount=request.amount,
                    )
                else:
                    logger.debug(f"No features found for user {request.user_id}")

        except Exception as e:
            logger.warning(f"Failed to fetch features from DB: {e}")

        # Fallback: Use simulated features for unknown users
        return self._simulate_features(request)

    def _simulate_features(self, request: SignalRequest) -> FeatureVector:
        """Generate simulated features for users not in database.

        Args:
            request: The signal request containing user_id.

        Returns:
            FeatureVector with simulated features.
        """
        # Use deterministic hash for consistent results per user
        user_hash = hash(request.user_id) % 1000

        # Simulate feature distribution based on user hash
        velocity = (user_hash % 10) + 1
        amount_ratio = 0.5 + (user_hash % 50) / 10.0
        balance_z = -3.0 + (user_hash % 60) / 10.0
        connections = user_hash % 8
        merchant_risk = user_hash % 100

        return FeatureVector(
            velocity_24h=velocity,
            amount_to_avg_ratio_30d=amount_ratio,
            balance_volatility_z_score=balance_z,
            bank_connections_24h=connections,
            merchant_risk_score=merchant_risk,
            has_history=False,  # Mark as no history since not in DB
            transaction_amount=request.amount,
        )

    def _calculate_probability(self, features: FeatureVector) -> float:
        """Calculate raw fraud probability from features.

        Uses a simple logistic-style combination of risk factors.
        In production, this would be an XGBoost model.

        Args:
            features: The feature vector for scoring.

        Returns:
            Raw probability between 0.0 and 1.0.
        """
        # Base probability
        prob = 0.05

        # Velocity contribution
        if features.velocity_24h > VELOCITY_HIGH_THRESHOLD:
            prob += 0.15 * min(features.velocity_24h / 10, 1.0)

        # Amount ratio contribution
        if features.amount_to_avg_ratio_30d > AMOUNT_RATIO_HIGH_THRESHOLD:
            ratio_factor = min(features.amount_to_avg_ratio_30d / 10.0, 1.0)
            prob += 0.20 * ratio_factor

        # Balance volatility contribution (negative z-score = low balance)
        if features.balance_volatility_z_score < BALANCE_VOLATILITY_THRESHOLD:
            vol_factor = min(abs(features.balance_volatility_z_score) / 5.0, 1.0)
            prob += 0.15 * vol_factor

        # Connection burst contribution
        if features.bank_connections_24h > CONNECTION_BURST_THRESHOLD:
            conn_factor = min(features.bank_connections_24h / 10.0, 1.0)
            prob += 0.20 * conn_factor

        # Merchant risk contribution
        if features.merchant_risk_score > MERCHANT_RISK_THRESHOLD:
            merchant_factor = (features.merchant_risk_score - 70) / 30.0
            prob += 0.15 * min(merchant_factor, 1.0)

        # Insufficient history penalty
        if not features.has_history:
            prob += 0.10

        # Cap probability
        return min(prob, 0.99)

    def _calibrate_score(self, probability: float) -> int:
        """Convert probability to calibrated 1-99 score.

        Args:
            probability: Raw probability between 0.0 and 1.0.

        Returns:
            Integer score between 1 and 99.
        """
        prob_array = np.array([probability])
        scores = self.calibrator.transform(prob_array)
        return int(scores[0])

    def _identify_risk_components(self, features: FeatureVector) -> list[RiskComponent]:
        """Identify risk components based on feature thresholds.

        This provides interpretability by flagging which features
        contributed to a high score.

        Args:
            features: The feature vector.

        Returns:
            List of risk components that triggered.
        """
        components = []

        if features.velocity_24h > VELOCITY_HIGH_THRESHOLD:
            components.append(
                RiskComponent(
                    key="velocity",
                    label="high_transaction_velocity",
                )
            )

        if features.amount_to_avg_ratio_30d > AMOUNT_RATIO_HIGH_THRESHOLD:
            components.append(
                RiskComponent(
                    key="amount_ratio",
                    label="unusual_transaction_amount",
                )
            )

        if features.balance_volatility_z_score < BALANCE_VOLATILITY_THRESHOLD:
            components.append(
                RiskComponent(
                    key="balance",
                    label="low_balance_volatility",
                )
            )

        if features.bank_connections_24h > CONNECTION_BURST_THRESHOLD:
            components.append(
                RiskComponent(
                    key="connections",
                    label="connection_burst_detected",
                )
            )

        if features.merchant_risk_score > MERCHANT_RISK_THRESHOLD:
            components.append(
                RiskComponent(
                    key="merchant",
                    label="high_risk_merchant",
                )
            )

        if not features.has_history:
            components.append(
                RiskComponent(
                    key="history",
                    label="insufficient_history",
                )
            )

        return components


# Singleton evaluator instance
_evaluator: SignalEvaluator | None = None


def get_evaluator() -> SignalEvaluator:
    """Get or create the signal evaluator singleton.

    Returns:
        SignalEvaluator instance.
    """
    global _evaluator
    if _evaluator is None:
        _evaluator = SignalEvaluator()
    return _evaluator
