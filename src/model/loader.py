"""Data loader for XGBoost training with temporal splitting."""

from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from synthetic_pipeline.db.session import DatabaseSession


@dataclass
class TrainTestSplit:
    """Container for train/test split data."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series

    @property
    def train_size(self) -> int:
        return len(self.X_train)

    @property
    def test_size(self) -> int:
        return len(self.X_test)

    @property
    def train_fraud_rate(self) -> float:
        if len(self.y_train) == 0:
            return 0.0
        return self.y_train.mean()

    @property
    def test_fraud_rate(self) -> float:
        if len(self.y_test) == 0:
            return 0.0
        return self.y_test.mean()


class DataLoader:
    """Loads and prepares data for XGBoost training with strict temporal splitting.

    Implements two key concepts:
    1. Strict Temporal Splitting: Train on data before cutoff, test on data after.
    2. Label Maturity (Knowledge Horizon): In training set, only label fraud if
       fraud_confirmed_at <= cutoff. This simulates not knowing about fraud
       that hasn't been detected yet.
    """

    # Feature columns from feature_snapshots table
    FEATURE_COLUMNS = [
        "velocity_24h",
        "amount_to_avg_ratio_30d",
        "balance_volatility_z_score",
    ]

    def __init__(self, database_url: str | None = None):
        """Initialize DataLoader.

        Args:
            database_url: Database connection URL. Defaults to env vars.
        """
        self.db_session = DatabaseSession(database_url=database_url)

    def load_train_test_split(
        self,
        training_cutoff_date: str | datetime,
        session: Session | None = None,
    ) -> TrainTestSplit:
        """Load train/test split with temporal splitting and label maturity.

        Args:
            training_cutoff_date: Cutoff date for train/test split (e.g., '2024-04-01').
                Records with created_at < cutoff go to train, >= cutoff go to test.
            session: Optional existing database session.

        Returns:
            TrainTestSplit containing X_train, y_train, X_test, y_test.
        """
        if isinstance(training_cutoff_date, str):
            cutoff = datetime.fromisoformat(training_cutoff_date)
        else:
            cutoff = training_cutoff_date

        if session is not None:
            return self._load_with_session(session, cutoff)

        with self.db_session.get_session() as session:
            return self._load_with_session(session, cutoff)

    def _load_with_session(
        self,
        session: Session,
        cutoff: datetime,
    ) -> TrainTestSplit:
        """Load data using provided session."""
        train_df = self._load_train_set(session, cutoff)
        test_df = self._load_test_set(session, cutoff)

        # Extract features and labels
        if len(train_df) > 0:
            features_train = train_df[self.FEATURE_COLUMNS]
            labels_train = train_df["label"]
        else:
            features_train = pd.DataFrame()
            labels_train = pd.Series(dtype=int)

        if len(test_df) > 0:
            features_test = test_df[self.FEATURE_COLUMNS]
            labels_test = test_df["label"]
        else:
            features_test = pd.DataFrame()
            labels_test = pd.Series(dtype=int)

        return TrainTestSplit(
            X_train=features_train,
            y_train=labels_train,
            X_test=features_test,
            y_test=labels_test,
        )

    def _load_train_set(self, session: Session, cutoff: datetime) -> pd.DataFrame:
        """Load training set with label maturity enforcement.

        Train Set Rules:
        - created_at < cutoff
        - is_train_eligible = True
        - Label is fraud ONLY IF fraud_confirmed_at <= cutoff (knowledge horizon)
        """
        query = text("""
            SELECT
                fs.record_id,
                fs.user_id,
                fs.velocity_24h,
                fs.amount_to_avg_ratio_30d,
                fs.balance_volatility_z_score,
                fs.experimental_signals,
                em.is_train_eligible,
                em.fraud_confirmed_at,
                gr.is_fraudulent,
                em.created_at,
                -- Knowledge Horizon: Only label fraud if confirmed before cutoff
                CASE
                    WHEN gr.is_fraudulent = TRUE
                         AND em.fraud_confirmed_at IS NOT NULL
                         AND em.fraud_confirmed_at <= :cutoff
                    THEN 1
                    ELSE 0
                END AS label
            FROM feature_snapshots fs
            INNER JOIN evaluation_metadata em ON fs.record_id = em.record_id
            INNER JOIN generated_records gr ON fs.record_id = gr.record_id
            WHERE em.created_at < :cutoff
              AND em.is_train_eligible = TRUE
            ORDER BY em.created_at
        """)

        result = session.execute(query, {"cutoff": cutoff})
        rows = result.fetchall()
        columns = result.keys()

        return pd.DataFrame(rows, columns=list(columns))

    def _load_test_set(self, session: Session, cutoff: datetime) -> pd.DataFrame:
        """Load test set (all records after cutoff).

        Test Set Rules:
        - created_at >= cutoff
        - Uses actual fraud label (no knowledge horizon restriction)
        """
        query = text("""
            SELECT
                fs.record_id,
                fs.user_id,
                fs.velocity_24h,
                fs.amount_to_avg_ratio_30d,
                fs.balance_volatility_z_score,
                fs.experimental_signals,
                em.is_train_eligible,
                em.fraud_confirmed_at,
                gr.is_fraudulent,
                em.created_at,
                -- Test set uses actual fraud label
                CASE WHEN gr.is_fraudulent = TRUE THEN 1 ELSE 0 END AS label
            FROM feature_snapshots fs
            INNER JOIN evaluation_metadata em ON fs.record_id = em.record_id
            INNER JOIN generated_records gr ON fs.record_id = gr.record_id
            WHERE em.created_at >= :cutoff
            ORDER BY em.created_at
        """)

        result = session.execute(query, {"cutoff": cutoff})
        rows = result.fetchall()
        columns = result.keys()

        return pd.DataFrame(rows, columns=list(columns))

    def get_split_summary(self, split: TrainTestSplit) -> dict:
        """Get summary statistics for the train/test split.

        Args:
            split: TrainTestSplit object.

        Returns:
            Dictionary with summary statistics.
        """
        train_fraud = int(split.y_train.sum()) if len(split.y_train) > 0 else 0
        test_fraud = int(split.y_test.sum()) if len(split.y_test) > 0 else 0

        return {
            "train_size": split.train_size,
            "test_size": split.test_size,
            "train_fraud_rate": split.train_fraud_rate,
            "test_fraud_rate": split.test_fraud_rate,
            "train_fraud_count": train_fraud,
            "test_fraud_count": test_fraud,
        }
