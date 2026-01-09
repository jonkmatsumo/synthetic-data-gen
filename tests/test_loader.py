"""Tests for DataLoader with temporal splitting and label maturity."""

from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from model.loader import DataLoader, TrainTestSplit


class TestTrainTestSplit:
    """Tests for TrainTestSplit dataclass."""

    def test_train_size(self):
        split = TrainTestSplit(
            X_train=pd.DataFrame({"a": [1, 2, 3]}),
            y_train=pd.Series([0, 1, 0]),
            X_test=pd.DataFrame({"a": [4, 5]}),
            y_test=pd.Series([1, 1]),
        )
        assert split.train_size == 3

    def test_test_size(self):
        split = TrainTestSplit(
            X_train=pd.DataFrame({"a": [1, 2, 3]}),
            y_train=pd.Series([0, 1, 0]),
            X_test=pd.DataFrame({"a": [4, 5]}),
            y_test=pd.Series([1, 1]),
        )
        assert split.test_size == 2

    def test_train_fraud_rate(self):
        split = TrainTestSplit(
            X_train=pd.DataFrame({"a": [1, 2, 3, 4]}),
            y_train=pd.Series([0, 1, 0, 1]),
            X_test=pd.DataFrame(),
            y_test=pd.Series(dtype=int),
        )
        assert split.train_fraud_rate == 0.5

    def test_test_fraud_rate(self):
        split = TrainTestSplit(
            X_train=pd.DataFrame(),
            y_train=pd.Series(dtype=int),
            X_test=pd.DataFrame({"a": [1, 2, 3, 4, 5]}),
            y_test=pd.Series([1, 1, 0, 0, 0]),
        )
        assert split.test_fraud_rate == 0.4

    def test_empty_train_fraud_rate(self):
        split = TrainTestSplit(
            X_train=pd.DataFrame(),
            y_train=pd.Series(dtype=int),
            X_test=pd.DataFrame({"a": [1]}),
            y_test=pd.Series([1]),
        )
        assert split.train_fraud_rate == 0.0

    def test_empty_test_fraud_rate(self):
        split = TrainTestSplit(
            X_train=pd.DataFrame({"a": [1]}),
            y_train=pd.Series([0]),
            X_test=pd.DataFrame(),
            y_test=pd.Series(dtype=int),
        )
        assert split.test_fraud_rate == 0.0


class TestDataLoaderInit:
    """Tests for DataLoader initialization."""

    def test_init_with_url(self):
        loader = DataLoader(database_url="postgresql://test:test@localhost/test")
        assert loader.db_session.database_url == "postgresql://test:test@localhost/test"

    def test_feature_columns_defined(self):
        assert "velocity_24h" in DataLoader.FEATURE_COLUMNS
        assert "amount_to_avg_ratio_30d" in DataLoader.FEATURE_COLUMNS
        assert "balance_volatility_z_score" in DataLoader.FEATURE_COLUMNS


class TestDataLoaderTemporalSplit:
    """Tests for temporal splitting logic."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        session = MagicMock()
        return session

    @pytest.fixture
    def loader(self):
        """Create a DataLoader instance."""
        return DataLoader(database_url="postgresql://test:test@localhost/test")

    def test_cutoff_date_string_parsing(self, loader, mock_session):
        """Test that string dates are parsed correctly."""
        cutoff = "2024-04-01"

        # Mock empty results
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_result.keys.return_value = [
            "record_id",
            "user_id",
            "velocity_24h",
            "amount_to_avg_ratio_30d",
            "balance_volatility_z_score",
            "experimental_signals",
            "is_train_eligible",
            "fraud_confirmed_at",
            "is_fraudulent",
            "created_at",
            "label",
        ]
        mock_session.execute.return_value = mock_result

        split = loader.load_train_test_split(cutoff, session=mock_session)
        assert isinstance(split, TrainTestSplit)

    def test_cutoff_date_datetime_accepted(self, loader, mock_session):
        """Test that datetime objects are accepted."""
        cutoff = datetime(2024, 4, 1)

        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_result.keys.return_value = [
            "record_id",
            "user_id",
            "velocity_24h",
            "amount_to_avg_ratio_30d",
            "balance_volatility_z_score",
            "experimental_signals",
            "is_train_eligible",
            "fraud_confirmed_at",
            "is_fraudulent",
            "created_at",
            "label",
        ]
        mock_session.execute.return_value = mock_result

        split = loader.load_train_test_split(cutoff, session=mock_session)
        assert isinstance(split, TrainTestSplit)

    def test_train_query_filters_by_cutoff_and_eligibility(self, loader, mock_session):
        """Test that train query uses correct WHERE clauses."""
        cutoff = datetime(2024, 4, 1)

        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_result.keys.return_value = [
            "record_id",
            "user_id",
            "velocity_24h",
            "amount_to_avg_ratio_30d",
            "balance_volatility_z_score",
            "experimental_signals",
            "is_train_eligible",
            "fraud_confirmed_at",
            "is_fraudulent",
            "created_at",
            "label",
        ]
        mock_session.execute.return_value = mock_result

        loader._load_train_set(mock_session, cutoff)

        # Verify execute was called
        mock_session.execute.assert_called_once()
        call_args = mock_session.execute.call_args
        query_text = str(call_args[0][0])

        # Check for correct filtering conditions
        assert "created_at < :cutoff" in query_text
        assert "is_train_eligible = TRUE" in query_text

    def test_test_query_filters_by_cutoff(self, loader, mock_session):
        """Test that test query uses correct WHERE clause."""
        cutoff = datetime(2024, 4, 1)

        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_result.keys.return_value = [
            "record_id",
            "user_id",
            "velocity_24h",
            "amount_to_avg_ratio_30d",
            "balance_volatility_z_score",
            "experimental_signals",
            "is_train_eligible",
            "fraud_confirmed_at",
            "is_fraudulent",
            "created_at",
            "label",
        ]
        mock_session.execute.return_value = mock_result

        loader._load_test_set(mock_session, cutoff)

        mock_session.execute.assert_called_once()
        call_args = mock_session.execute.call_args
        query_text = str(call_args[0][0])

        assert "created_at >= :cutoff" in query_text


class TestDataLoaderLabelMaturity:
    """Tests for label maturity (knowledge horizon) logic."""

    @pytest.fixture
    def mock_session(self):
        return MagicMock()

    @pytest.fixture
    def loader(self):
        return DataLoader(database_url="postgresql://test:test@localhost/test")

    def test_train_label_checks_fraud_confirmed_at(self, loader, mock_session):
        """Train set should only label fraud if confirmed before cutoff."""
        cutoff = datetime(2024, 4, 1)

        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_result.keys.return_value = [
            "record_id",
            "user_id",
            "velocity_24h",
            "amount_to_avg_ratio_30d",
            "balance_volatility_z_score",
            "experimental_signals",
            "is_train_eligible",
            "fraud_confirmed_at",
            "is_fraudulent",
            "created_at",
            "label",
        ]
        mock_session.execute.return_value = mock_result

        loader._load_train_set(mock_session, cutoff)

        call_args = mock_session.execute.call_args
        query_text = str(call_args[0][0])

        # Verify knowledge horizon is enforced
        assert "fraud_confirmed_at <= :cutoff" in query_text
        assert "is_fraudulent = TRUE" in query_text

    def test_test_label_uses_actual_fraud(self, loader, mock_session):
        """Test set should use actual fraud label without horizon check."""
        cutoff = datetime(2024, 4, 1)

        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_result.keys.return_value = [
            "record_id",
            "user_id",
            "velocity_24h",
            "amount_to_avg_ratio_30d",
            "balance_volatility_z_score",
            "experimental_signals",
            "is_train_eligible",
            "fraud_confirmed_at",
            "is_fraudulent",
            "created_at",
            "label",
        ]
        mock_session.execute.return_value = mock_result

        loader._load_test_set(mock_session, cutoff)

        call_args = mock_session.execute.call_args
        query_text = str(call_args[0][0])

        # Verify test uses actual label
        expected_label = "CASE WHEN gr.is_fraudulent = TRUE THEN 1 ELSE 0 END"
        assert expected_label in query_text


class TestDataLoaderJoinLogic:
    """Tests for join logic between tables."""

    @pytest.fixture
    def mock_session(self):
        return MagicMock()

    @pytest.fixture
    def loader(self):
        return DataLoader(database_url="postgresql://test:test@localhost/test")

    def test_joins_feature_snapshots_and_evaluation_metadata(
        self, loader, mock_session
    ):
        """Test that feature_snapshots is joined with evaluation_metadata."""
        cutoff = datetime(2024, 4, 1)

        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_result.keys.return_value = [
            "record_id",
            "user_id",
            "velocity_24h",
            "amount_to_avg_ratio_30d",
            "balance_volatility_z_score",
            "experimental_signals",
            "is_train_eligible",
            "fraud_confirmed_at",
            "is_fraudulent",
            "created_at",
            "label",
        ]
        mock_session.execute.return_value = mock_result

        loader._load_train_set(mock_session, cutoff)

        call_args = mock_session.execute.call_args
        query_text = str(call_args[0][0])

        assert "feature_snapshots fs" in query_text
        assert "evaluation_metadata em" in query_text
        assert "fs.record_id = em.record_id" in query_text

    def test_joins_generated_records_for_labels(self, loader, mock_session):
        """Test that generated_records is joined for fraud labels."""
        cutoff = datetime(2024, 4, 1)

        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_result.keys.return_value = [
            "record_id",
            "user_id",
            "velocity_24h",
            "amount_to_avg_ratio_30d",
            "balance_volatility_z_score",
            "experimental_signals",
            "is_train_eligible",
            "fraud_confirmed_at",
            "is_fraudulent",
            "created_at",
            "label",
        ]
        mock_session.execute.return_value = mock_result

        loader._load_train_set(mock_session, cutoff)

        call_args = mock_session.execute.call_args
        query_text = str(call_args[0][0])

        assert "generated_records gr" in query_text
        assert "fs.record_id = gr.record_id" in query_text


class TestGetSplitSummary:
    """Tests for get_split_summary method."""

    def test_summary_contains_all_fields(self):
        loader = DataLoader(database_url="postgresql://test:test@localhost/test")
        split = TrainTestSplit(
            X_train=pd.DataFrame({"a": [1, 2, 3]}),
            y_train=pd.Series([0, 1, 0]),
            X_test=pd.DataFrame({"a": [4, 5]}),
            y_test=pd.Series([1, 1]),
        )

        summary = loader.get_split_summary(split)

        assert "train_size" in summary
        assert "test_size" in summary
        assert "train_fraud_rate" in summary
        assert "test_fraud_rate" in summary
        assert "train_fraud_count" in summary
        assert "test_fraud_count" in summary

    def test_summary_values_correct(self):
        loader = DataLoader(database_url="postgresql://test:test@localhost/test")
        split = TrainTestSplit(
            X_train=pd.DataFrame({"a": [1, 2, 3, 4]}),
            y_train=pd.Series([0, 1, 1, 0]),
            X_test=pd.DataFrame({"a": [5, 6, 7]}),
            y_test=pd.Series([1, 0, 1]),
        )

        summary = loader.get_split_summary(split)

        assert summary["train_size"] == 4
        assert summary["test_size"] == 3
        assert summary["train_fraud_rate"] == 0.5
        assert abs(summary["test_fraud_rate"] - 2 / 3) < 0.01
        assert summary["train_fraud_count"] == 2
        assert summary["test_fraud_count"] == 2
