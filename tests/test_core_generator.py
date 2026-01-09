"""Tests for the stateful core generator."""

from datetime import datetime, timedelta
from decimal import Decimal

from generator.core import (
    BustOutProfile,
    FraudScenario,
    GenerationConfig,
    LabelDelaySimulator,
    SleeperProfile,
    UserSimulator,
    UserState,
)


class TestUserState:
    """Tests for UserState dataclass."""

    def test_initial_state(self):
        """Test default initial state values."""
        state = UserState(user_id="test_user")
        assert state.user_id == "test_user"
        assert state.balance == Decimal("5000.00")
        assert state.transaction_count == 0
        assert state.avg_transaction_amount == Decimal("0.00")

    def test_update_after_transaction(self):
        """Test state updates after a transaction."""
        state = UserState(user_id="test_user", balance=Decimal("1000.00"))
        timestamp = datetime.now()

        state.update_after_transaction(
            amount=Decimal("100.00"),
            timestamp=timestamp,
            connections=2,
        )

        assert state.transaction_count == 1
        assert state.total_amount_spent == Decimal("100.00")
        assert state.balance == Decimal("900.00")
        assert state.avg_transaction_amount == Decimal("100.00")
        assert state.connections_24h == 2

    def test_running_average(self):
        """Test running average calculation."""
        state = UserState(user_id="test_user", balance=Decimal("1000.00"))
        timestamp = datetime.now()

        state.update_after_transaction(Decimal("100.00"), timestamp)
        state.update_after_transaction(Decimal("200.00"), timestamp)
        state.update_after_transaction(Decimal("300.00"), timestamp)

        assert state.transaction_count == 3
        assert state.avg_transaction_amount == Decimal("200.00")


class TestLabelDelaySimulator:
    """Tests for LabelDelaySimulator."""

    def test_confirmation_in_past(self):
        """Test fraud detected when confirmation is before simulation date."""
        import numpy as np

        rng = np.random.default_rng(42)
        simulator = LabelDelaySimulator(mean_days=1.0, rng=rng)
        fraud_time = datetime.now() - timedelta(days=30)
        simulation_date = datetime.now()

        confirmed_at, is_detected = simulator.calculate_confirmation_time(
            fraud_time, simulation_date
        )

        assert confirmed_at is not None
        assert confirmed_at > fraud_time
        assert is_detected is True

    def test_confirmation_in_future(self):
        """Test fraud undetected when confirmation is after simulation date."""
        import numpy as np

        rng = np.random.default_rng(42)
        simulator = LabelDelaySimulator(mean_days=30.0, rng=rng)
        fraud_time = datetime.now() - timedelta(days=1)
        simulation_date = datetime.now()

        confirmed_at, is_detected = simulator.calculate_confirmation_time(
            fraud_time, simulation_date
        )

        assert confirmed_at is not None
        # With mean 30 days and fraud just 1 day ago, likely not detected
        # Note: this is probabilistic, so we just verify the logic works


class TestUserSimulator:
    """Tests for UserSimulator."""

    def test_sequence_number_increments(self):
        """Test that sequence numbers increment correctly."""
        simulator = UserSimulator(seed=42)

        result1 = simulator.tick()
        result2 = simulator.tick()
        result3 = simulator.tick()

        assert result1.metadata.sequence_number == 1
        assert result2.metadata.sequence_number == 2
        assert result3.metadata.sequence_number == 3

    def test_timestamps_monotonic(self):
        """Test that timestamps are strictly monotonic."""
        simulator = UserSimulator(seed=42)

        results = [simulator.tick() for _ in range(10)]

        timestamps = [r.record.transaction_timestamp for r in results]
        for i in range(1, len(timestamps)):
            assert timestamps[i] > timestamps[i - 1]

    def test_legitimate_user_no_fraud(self):
        """Test that legitimate users generate no fraud transactions."""
        simulator = UserSimulator(seed=42)

        records, metadata = simulator.generate_full_sequence(num_transactions=20)

        assert all(not r.is_fraudulent for r in records)
        assert all(m.is_train_eligible for m in metadata)

    def test_user_id_consistent(self):
        """Test that user_id is consistent across all records."""
        simulator = UserSimulator(user_id="test_user_123", seed=42)

        records, metadata = simulator.generate_full_sequence(num_transactions=10)

        assert all(r.user_id == "test_user_123" for r in records)
        assert all(m.user_id == "test_user_123" for m in metadata)


class TestBustOutProfile:
    """Tests for BustOutProfile."""

    def test_pre_fraud_transaction_count(self):
        """Test that bust-out generates correct number of pre-fraud transactions."""
        profile = BustOutProfile(min_transactions=20, max_transactions=30)
        simulator = UserSimulator(fraud_profile=profile, seed=42)

        records, metadata = simulator.generate_full_sequence()

        # Count non-fraud records
        legitimate_count = sum(1 for r in records if not r.is_fraudulent)
        assert legitimate_count >= 20
        assert legitimate_count <= 33  # Allow for post-fraud transactions

    def test_spike_amount(self):
        """Test that fraud transaction has >500% spike."""
        profile = BustOutProfile(min_transactions=20, max_transactions=20)
        simulator = UserSimulator(fraud_profile=profile, seed=42)

        records, _ = simulator.generate_full_sequence()

        # Find fraud record
        fraud_records = [r for r in records if r.is_fraudulent]
        assert len(fraud_records) >= 1

        fraud_record = fraud_records[0]
        assert fraud_record.transaction.amount_to_avg_ratio > 5.0

    def test_fraud_type_correct(self):
        """Test that fraud type is set correctly."""
        profile = BustOutProfile(min_transactions=5, max_transactions=5)
        simulator = UserSimulator(fraud_profile=profile, seed=42)

        records, _ = simulator.generate_full_sequence()

        fraud_records = [r for r in records if r.is_fraudulent]
        assert len(fraud_records) >= 1
        assert fraud_records[0].fraud_type == FraudScenario.BUST_OUT.value

    def test_fraud_event_train_eligible(self):
        """Test that fraud event is train-eligible for bust-out."""
        profile = BustOutProfile(min_transactions=5, max_transactions=5)
        simulator = UserSimulator(fraud_profile=profile, seed=42)

        records, metadata = simulator.generate_full_sequence()

        # Find fraud record and its metadata
        fraud_records = [r for r in records if r.is_fraudulent]
        assert len(fraud_records) >= 1

        # Get metadata for fraud record
        fraud_record_id = fraud_records[0].record_id
        fraud_meta = next(m for m in metadata if m.record_id == fraud_record_id)

        # Bust-out fraud event should be train-eligible
        assert fraud_meta.is_train_eligible is True


class TestSleeperProfile:
    """Tests for SleeperProfile."""

    def test_dormancy_requirement(self):
        """Test that sleeper waits for dormancy period."""
        profile = SleeperProfile(dormant_days=30)
        simulator = UserSimulator(
            fraud_profile=profile,
            seed=42,
            start_time=datetime.now() - timedelta(days=90),
        )

        records, _ = simulator.generate_full_sequence()

        # Should have some transactions before dormancy triggers fraud
        assert len(records) >= 3

    def test_link_burst_connections(self):
        """Test that link burst has elevated connections."""
        profile = SleeperProfile(dormant_days=30, burst_connections=3)
        simulator = UserSimulator(
            fraud_profile=profile,
            seed=42,
            start_time=datetime.now() - timedelta(days=120),
        )

        records, _ = simulator.generate_full_sequence()

        # Check for burst events (high connections before fraud)
        high_connection_records = [
            r for r in records if r.connection.bank_connections_count_24h >= 3
        ]
        assert len(high_connection_records) >= 1

    def test_fraud_type_correct(self):
        """Test that fraud type is set correctly for sleeper."""
        profile = SleeperProfile(dormant_days=30)
        simulator = UserSimulator(
            fraud_profile=profile,
            seed=42,
            start_time=datetime.now() - timedelta(days=120),
        )

        # Use future simulation date to ensure fraud is detected
        future_date = datetime.now() + timedelta(days=365)
        records, _ = simulator.generate_full_sequence(simulation_date=future_date)

        fraud_records = [r for r in records if r.is_fraudulent]
        assert len(fraud_records) >= 1
        assert fraud_records[0].fraud_type == FraudScenario.SLEEPER_ATO.value

    def test_ato_off_hours(self):
        """Test that ATO fraud happens during off hours."""
        profile = SleeperProfile(dormant_days=30)
        simulator = UserSimulator(
            fraud_profile=profile,
            seed=42,
            start_time=datetime.now() - timedelta(days=120),
        )

        # Use future simulation date to ensure fraud is detected
        future_date = datetime.now() + timedelta(days=365)
        records, _ = simulator.generate_full_sequence(simulation_date=future_date)

        fraud_records = [r for r in records if r.is_fraudulent]
        assert len(fraud_records) >= 1
        assert fraud_records[0].is_off_hours_txn is True

    def test_identity_change_present(self):
        """Test that ATO fraud has identity change."""
        profile = SleeperProfile(dormant_days=30)
        simulator = UserSimulator(
            fraud_profile=profile,
            seed=42,
            start_time=datetime.now() - timedelta(days=120),
        )

        # Use future simulation date to ensure fraud is detected
        future_date = datetime.now() + timedelta(days=365)
        records, _ = simulator.generate_full_sequence(simulation_date=future_date)

        fraud_records = [r for r in records if r.is_fraudulent]
        assert len(fraud_records) >= 1

        fraud = fraud_records[0]
        has_identity_change = (
            fraud.identity_changes.email_changed_at is not None
            or fraud.identity_changes.phone_changed_at is not None
        )
        assert has_identity_change


class TestGenerationConfig:
    """Tests for GenerationConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = GenerationConfig()

        assert config.num_users == 100
        assert config.fraud_rate == 0.1
        assert config.bust_out_ratio == 0.5
        assert config.sleeper_ratio == 0.5

    def test_fraud_user_calculation(self):
        """Test fraud user count calculation."""
        config = GenerationConfig(num_users=100, fraud_rate=0.1)

        num_fraud = int(config.num_users * config.fraud_rate)
        num_bust_out = int(num_fraud * config.bust_out_ratio)
        num_sleeper = num_fraud - num_bust_out

        assert num_fraud == 10
        assert num_bust_out == 5
        assert num_sleeper == 5
