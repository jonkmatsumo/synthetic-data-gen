"""Stateful synthetic data generator with fraud scenario profiles.

This module provides a UserSimulator class that maintains persistent state
for generating realistic transaction sequences with proper temporal ordering
and fraud detection simulation.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from faker import Faker

if TYPE_CHECKING:
    from numpy.random import Generator as NPGenerator

from synthetic_pipeline.db import (
    DatabaseSession,
    EvaluationMetadataDB,
    GeneratedRecordDB,
)
from synthetic_pipeline.models import (
    AccountSnapshot,
    BehaviorMetrics,
    ConnectionMetrics,
    EvaluationMetadata,
    GeneratedRecord,
    IdentityChangeInfo,
    TransactionEvaluation,
)


class FraudScenario(str, Enum):
    """Types of fraud scenarios for stateful generation."""

    BUST_OUT = "bust_out"
    SLEEPER_ATO = "sleeper_ato"
    LEGITIMATE = "legitimate"


@dataclass
class UserState:
    """Persistent state for a user across transactions.

    Attributes:
        user_id: Unique identifier for the user.
        balance: Current available balance.
        transaction_count: Total number of transactions generated.
        total_amount_spent: Cumulative amount spent.
        avg_transaction_amount: Running average of transaction amounts.
        last_transaction_time: Timestamp of the most recent transaction.
        last_connection_time: Timestamp of the most recent bank connection.
        connections_24h: Bank connections in the last 24 hours.
        days_since_last_activity: Days since last transaction.
    """

    user_id: str
    balance: Decimal = Decimal("5000.00")
    transaction_count: int = 0
    total_amount_spent: Decimal = Decimal("0.00")
    avg_transaction_amount: Decimal = Decimal("0.00")
    last_transaction_time: datetime | None = None
    last_connection_time: datetime | None = None
    connections_24h: int = 0
    connections_7d: int = 0
    days_since_last_activity: float = 0.0
    pii: tuple[str, str, str] | None = None  # (name, email, phone)

    def update_after_transaction(
        self, amount: Decimal, timestamp: datetime, connections: int = 0
    ) -> None:
        """Update state after a transaction."""
        self.transaction_count += 1
        self.total_amount_spent += amount
        self.balance -= amount

        # Update running average
        if self.transaction_count > 0:
            self.avg_transaction_amount = (
                self.total_amount_spent / self.transaction_count
            )

        # Track time gaps
        if self.last_transaction_time is not None:
            delta = timestamp - self.last_transaction_time
            self.days_since_last_activity = delta.total_seconds() / 86400

        self.last_transaction_time = timestamp

        # Track connections
        if connections > 0:
            self.connections_24h = connections
            self.last_connection_time = timestamp


@dataclass
class TransactionResult:
    """Result of generating a single transaction."""

    record: GeneratedRecord
    metadata: EvaluationMetadata
    is_fraud_event: bool = False


class LabelDelaySimulator:
    """Simulates the delay between fraud occurrence and detection.

    Uses a Log-Normal distribution to model the realistic delay in fraud
    detection systems. Some fraud may remain undetected if the confirmation
    date is in the future relative to the simulation date.

    Attributes:
        mean_days: Mean delay in days (default 5).
        sigma: Standard deviation for log-normal distribution.
        rng: Numpy random generator.
    """

    def __init__(
        self,
        mean_days: float = 5.0,
        sigma: float = 0.8,
        rng: NPGenerator | None = None,
    ):
        """Initialize the label delay simulator.

        Args:
            mean_days: Mean delay in days until fraud is confirmed.
            sigma: Standard deviation for the log-normal distribution.
            rng: Numpy random generator for reproducibility.
        """
        self.mean_days = mean_days
        self.sigma = sigma
        self.rng = rng or np.random.default_rng()

    def calculate_confirmation_time(
        self,
        fraud_transaction_time: datetime,
        simulation_date: datetime | None = None,
    ) -> tuple[datetime | None, bool]:
        """Calculate when fraud will be confirmed.

        Args:
            fraud_transaction_time: When the fraudulent transaction occurred.
            simulation_date: Current date in simulation (defaults to now).

        Returns:
            Tuple of (fraud_confirmed_at, is_detected).
            If fraud_confirmed_at > simulation_date, is_detected=False
            (simulating undetected fraud).
        """
        if simulation_date is None:
            simulation_date = datetime.now()

        # Log-normal delay calculation
        # mu for log-normal to achieve desired mean
        mu = np.log(self.mean_days) - (self.sigma**2) / 2
        delay_days = float(self.rng.lognormal(mu, self.sigma))

        # Clamp to reasonable range (1 hour to 60 days)
        delay_days = max(1 / 24, min(delay_days, 60.0))

        fraud_confirmed_at = fraud_transaction_time + timedelta(days=delay_days)

        # If confirmation is in the future, fraud appears "clean"
        is_detected = fraud_confirmed_at <= simulation_date

        return fraud_confirmed_at, is_detected


class FraudProfile(ABC):
    """Abstract base class for fraud scenario profiles."""

    @abstractmethod
    def should_trigger_fraud(self, state: UserState) -> bool:
        """Determine if fraud should be triggered based on current state."""
        pass

    @abstractmethod
    def get_pre_fraud_transaction_count(self, rng: NPGenerator) -> int:
        """Get the number of legitimate transactions before fraud."""
        pass

    @abstractmethod
    def generate_fraud_transaction(
        self,
        simulator: UserSimulator,
        state: UserState,
    ) -> GeneratedRecord:
        """Generate the fraud transaction."""
        pass

    @abstractmethod
    def get_scenario_type(self) -> FraudScenario:
        """Get the fraud scenario type."""
        pass


class BustOutProfile(FraudProfile):
    """The "Bust-Out" fraud profile.

    Behavior: 20-50 small legitimate transactions (is_pre_fraud=False),
    followed by a sudden spike in amount (>500% of avg) which is the fraud event.

    The fraud event has is_train_eligible=True since it represents the actual
    detectable fraud pattern.
    """

    def __init__(
        self,
        min_transactions: int = 20,
        max_transactions: int = 50,
        spike_multiplier: float = 5.0,
    ):
        """Initialize Bust-Out profile.

        Args:
            min_transactions: Minimum legitimate transactions before fraud.
            max_transactions: Maximum legitimate transactions before fraud.
            spike_multiplier: Amount spike multiplier (>500% = 5.0).
        """
        self.min_transactions = min_transactions
        self.max_transactions = max_transactions
        self.spike_multiplier = spike_multiplier

    def should_trigger_fraud(self, state: UserState) -> bool:
        """Trigger fraud after sufficient legitimate transaction history."""
        return state.transaction_count >= self.min_transactions

    def get_pre_fraud_transaction_count(self, rng: NPGenerator) -> int:
        """Get random count between min and max transactions."""
        return int(rng.integers(self.min_transactions, self.max_transactions + 1))

    def generate_fraud_transaction(
        self,
        simulator: UserSimulator,
        state: UserState,
    ) -> GeneratedRecord:
        """Generate the bust-out fraud transaction with >500% spike."""
        # Calculate spike amount (>500% of average)
        if state.avg_transaction_amount > 0:
            spike_amount = state.avg_transaction_amount * Decimal(
                str(self.spike_multiplier + simulator.rng.uniform(0.5, 2.0))
            )
        else:
            spike_amount = Decimal(str(simulator.rng.uniform(2000, 5000)))

        spike_amount = Decimal(str(round(float(spike_amount), 2)))

        # Calculate amount_to_avg_ratio
        if state.avg_transaction_amount > 0:
            amount_to_avg = float(spike_amount / state.avg_transaction_amount)
        else:
            amount_to_avg = self.spike_multiplier + 1.0

        name, email, phone = state.pii or simulator._generate_pii()
        timestamp = simulator._next_timestamp(state)

        return GeneratedRecord(
            record_id=simulator._generate_record_id(),
            user_id=state.user_id,
            full_name=name,
            email=email,
            phone=phone,
            transaction_timestamp=timestamp,
            is_off_hours_txn=False,
            account=AccountSnapshot(
                available_balance=state.balance,
                balance_to_transaction_ratio=float(state.balance / spike_amount)
                if spike_amount > 0
                else 0.0,
            ),
            behavior=BehaviorMetrics(
                avg_available_balance_30d=Decimal(
                    str(round(float(state.balance * Decimal("1.2")), 2))
                ),
                balance_volatility_z_score=float(simulator.rng.uniform(-1.5, -0.5)),
            ),
            connection=ConnectionMetrics(
                bank_connections_count_24h=state.connections_24h,
                bank_connections_count_7d=state.connections_7d,
                bank_connections_avg_30d=float(simulator.rng.uniform(0.5, 2.0)),
            ),
            transaction=TransactionEvaluation(
                amount=spike_amount,
                amount_to_avg_ratio=amount_to_avg,
                merchant_risk_score=int(simulator.rng.integers(50, 85)),
                is_returned=False,
            ),
            identity_changes=IdentityChangeInfo(),
            is_fraudulent=True,
            fraud_type=FraudScenario.BUST_OUT.value,
        )

    def get_scenario_type(self) -> FraudScenario:
        return FraudScenario.BUST_OUT


class SleeperProfile(FraudProfile):
    """The "Sleeper" Account (ATO) fraud profile.

    Behavior: Dormant for 30+ days, then a "Link Burst" (3+ connections in 1 hour)
    followed by high-value debit.

    The burst events are marked is_pre_fraud=True since they precede the actual
    fraud transaction.
    """

    def __init__(
        self,
        dormant_days: int = 30,
        burst_connections: int = 3,
        high_value_multiplier: float = 3.0,
    ):
        """Initialize Sleeper profile.

        Args:
            dormant_days: Minimum days of inactivity before ATO.
            burst_connections: Number of connections in the link burst.
            high_value_multiplier: Multiplier for high-value debit amount.
        """
        self.dormant_days = dormant_days
        self.burst_connections = burst_connections
        self.high_value_multiplier = high_value_multiplier

    def should_trigger_fraud(self, state: UserState) -> bool:
        """Trigger fraud after dormancy period."""
        return state.days_since_last_activity >= self.dormant_days

    def get_pre_fraud_transaction_count(self, rng: NPGenerator) -> int:
        """Sleeper accounts have minimal pre-fraud transactions."""
        return int(rng.integers(3, 10))

    def generate_link_burst_event(
        self,
        simulator: UserSimulator,
        state: UserState,
    ) -> GeneratedRecord:
        """Generate the link burst event (precursor to fraud)."""
        name, email, phone = state.pii or simulator._generate_pii()
        timestamp = simulator._next_timestamp(state)

        # Small transaction during burst
        amount = Decimal(str(round(float(simulator.rng.uniform(10, 50)), 2)))

        return GeneratedRecord(
            record_id=simulator._generate_record_id(),
            user_id=state.user_id,
            full_name=name,
            email=email,
            phone=phone,
            transaction_timestamp=timestamp,
            is_off_hours_txn=False,
            account=AccountSnapshot(
                available_balance=state.balance,
                balance_to_transaction_ratio=float(state.balance / amount)
                if amount > 0
                else 0.0,
            ),
            behavior=BehaviorMetrics(
                avg_available_balance_30d=Decimal(
                    str(round(float(state.balance * Decimal("0.9")), 2))
                ),
                balance_volatility_z_score=float(simulator.rng.uniform(-0.5, 0.5)),
            ),
            connection=ConnectionMetrics(
                bank_connections_count_24h=self.burst_connections
                + int(simulator.rng.integers(0, 3)),
                bank_connections_count_7d=self.burst_connections
                + int(simulator.rng.integers(2, 8)),
                bank_connections_avg_30d=float(simulator.rng.uniform(0.1, 0.5)),
            ),
            transaction=TransactionEvaluation(
                amount=amount,
                amount_to_avg_ratio=float(simulator.rng.uniform(0.2, 0.8)),
                merchant_risk_score=int(simulator.rng.integers(20, 50)),
                is_returned=False,
            ),
            identity_changes=IdentityChangeInfo(),
            is_fraudulent=False,  # Burst itself is not fraud
            fraud_type=None,
        )

    def generate_fraud_transaction(
        self,
        simulator: UserSimulator,
        state: UserState,
    ) -> GeneratedRecord:
        """Generate the high-value debit fraud transaction."""
        # High-value amount
        if state.avg_transaction_amount > 0:
            high_value = state.avg_transaction_amount * Decimal(
                str(self.high_value_multiplier + simulator.rng.uniform(0.5, 2.0))
            )
        else:
            high_value = Decimal(str(simulator.rng.uniform(1500, 4000)))

        high_value = Decimal(str(round(float(high_value), 2)))

        name, email, phone = state.pii or simulator._generate_pii()
        timestamp = simulator._next_timestamp(state, off_hours=True)

        # Identity change within 72 hours (ATO indicator)
        hours_ago = float(simulator.rng.uniform(1, 70))
        change_time = timestamp - timedelta(hours=hours_ago)

        return GeneratedRecord(
            record_id=simulator._generate_record_id(),
            user_id=state.user_id,
            full_name=name,
            email=email,
            phone=phone,
            transaction_timestamp=timestamp,
            is_off_hours_txn=True,
            account=AccountSnapshot(
                available_balance=state.balance,
                balance_to_transaction_ratio=float(state.balance / high_value)
                if high_value > 0
                else 0.0,
            ),
            behavior=BehaviorMetrics(
                avg_available_balance_30d=Decimal(
                    str(round(float(state.balance * Decimal("0.8")), 2))
                ),
                balance_volatility_z_score=float(simulator.rng.uniform(-2.0, -0.5)),
            ),
            connection=ConnectionMetrics(
                bank_connections_count_24h=self.burst_connections + 2,
                bank_connections_count_7d=self.burst_connections + 5,
                bank_connections_avg_30d=float(simulator.rng.uniform(0.1, 0.3)),
            ),
            transaction=TransactionEvaluation(
                amount=high_value,
                amount_to_avg_ratio=float(high_value / state.avg_transaction_amount)
                if state.avg_transaction_amount > 0
                else 5.0,
                merchant_risk_score=int(simulator.rng.integers(60, 95)),
                is_returned=False,
            ),
            identity_changes=IdentityChangeInfo(
                email_changed_at=change_time,
                phone_changed_at=change_time if simulator.rng.random() > 0.5 else None,
            ),
            is_fraudulent=True,
            fraud_type=FraudScenario.SLEEPER_ATO.value,
        )

    def get_scenario_type(self) -> FraudScenario:
        return FraudScenario.SLEEPER_ATO


class UserSimulator:
    """Stateful user simulator that generates transaction sequences.

    Maintains persistent state for a user including balance, transaction history,
    and timing. The tick() method generates the next transaction with:
    - Incrementing sequence numbers (1, 2, 3...)
    - Strictly monotonic timestamps

    Attributes:
        state: Current user state.
        sequence_number: Current sequence number (1-indexed).
        fraud_profile: Optional fraud profile for this user.
        label_delay: Label delay simulator for fraud confirmation.
    """

    def __init__(
        self,
        user_id: str | None = None,
        initial_balance: Decimal = Decimal("5000.00"),
        fraud_profile: FraudProfile | None = None,
        seed: int | None = None,
        start_time: datetime | None = None,
    ):
        """Initialize the user simulator.

        Args:
            user_id: Unique user ID (generated if not provided).
            initial_balance: Starting account balance.
            fraud_profile: Optional fraud profile for this user.
            seed: Random seed for reproducibility.
            start_time: Starting timestamp for transactions.
        """
        self.rng = np.random.default_rng(seed)
        self.faker = Faker()
        if seed is not None:
            Faker.seed(seed)

        self.user_id = user_id or self._generate_user_id()
        self.state = UserState(
            user_id=self.user_id,
            balance=initial_balance,
            pii=self._generate_pii(),
        )
        self.sequence_number = 0
        self.fraud_profile = fraud_profile
        self.label_delay = LabelDelaySimulator(rng=self.rng)

        # Initialize timing
        self._current_time = start_time or (
            datetime.now() - timedelta(days=int(self.rng.integers(60, 180)))
        )
        self._fraud_triggered = False
        self._fraud_confirmed_at: datetime | None = None
        self._fraud_transaction_time: datetime | None = None
        self._burst_events_generated = 0

        # For Bust-Out: track target transaction count
        if fraud_profile is not None:
            self._target_pre_fraud_count = (
                fraud_profile.get_pre_fraud_transaction_count(self.rng)
            )
        else:
            self._target_pre_fraud_count = None

        # For Sleeper: track if dormancy has been triggered
        self._dormancy_triggered = False

    def _generate_record_id(self) -> str:
        """Generate a unique record ID."""
        return f"rec_{uuid.uuid4().hex[:12]}"

    def _generate_user_id(self) -> str:
        """Generate a unique user ID."""
        return f"user_{uuid.uuid4().hex[:12]}"

    def _generate_pii(self) -> tuple[str, str, str]:
        """Generate fake PII (name, email, phone)."""
        name = self.faker.name()
        email = self.faker.email()
        phone = self.faker.phone_number()
        return name, email, phone

    def _next_timestamp(
        self,
        state: UserState,
        off_hours: bool = False,
        min_gap_hours: float = 0.5,
    ) -> datetime:
        """Generate the next strictly monotonic timestamp.

        Args:
            state: Current user state.
            off_hours: If True, generate time between 10pm-6am.
            min_gap_hours: Minimum hours between transactions.

        Returns:
            Next timestamp, guaranteed to be after the previous one.
        """
        # Add random gap (at least min_gap_hours)
        gap_hours = float(self.rng.uniform(min_gap_hours, 72))
        self._current_time += timedelta(hours=gap_hours)

        if off_hours:
            # Adjust to off-hours: 10pm (22:00) to 6am (06:00)
            hour = int(self.rng.choice(list(range(22, 24)) + list(range(0, 6))))
            self._current_time = self._current_time.replace(
                hour=hour,
                minute=int(self.rng.integers(0, 60)),
                second=int(self.rng.integers(0, 60)),
                microsecond=0,
            )
        else:
            # Normal hours adjustment
            self._current_time = self._current_time.replace(
                minute=int(self.rng.integers(0, 60)),
                second=int(self.rng.integers(0, 60)),
                microsecond=0,
            )

        return self._current_time

    def _sample_lognormal_amount(
        self,
        mean: float = 75.0,
        sigma: float = 1.0,
    ) -> Decimal:
        """Sample transaction amount from log-normal distribution."""
        mu = np.log(mean) - (sigma**2) / 2
        amount = float(self.rng.lognormal(mu, sigma))
        amount = max(0.01, min(amount, 50000.0))
        return Decimal(str(round(amount, 2)))

    def _generate_legitimate_transaction(self) -> GeneratedRecord:
        """Generate a legitimate transaction."""
        name, email, phone = self.state.pii or self._generate_pii()
        timestamp = self._next_timestamp(self.state)
        amount = self._sample_lognormal_amount()

        # Normal balance volatility
        volatility_z = float(self.rng.uniform(-1.0, 1.0))

        # Normal connection patterns
        connections_24h = int(self.rng.integers(0, 3))
        connections_7d = int(self.rng.integers(0, 10))

        # Amount ratio close to 1.0
        if self.state.avg_transaction_amount > 0:
            amount_to_avg = float(amount / self.state.avg_transaction_amount)
        else:
            amount_to_avg = 1.0

        return GeneratedRecord(
            record_id=self._generate_record_id(),
            user_id=self.user_id,
            full_name=name,
            email=email,
            phone=phone,
            transaction_timestamp=timestamp,
            is_off_hours_txn=False,
            account=AccountSnapshot(
                available_balance=self.state.balance,
                balance_to_transaction_ratio=float(self.state.balance / amount)
                if amount > 0
                else 0.0,
            ),
            behavior=BehaviorMetrics(
                avg_available_balance_30d=Decimal(
                    str(round(float(self.state.balance * Decimal("1.1")), 2))
                ),
                balance_volatility_z_score=volatility_z,
            ),
            connection=ConnectionMetrics(
                bank_connections_count_24h=connections_24h,
                bank_connections_count_7d=connections_7d,
                bank_connections_avg_30d=float(self.rng.uniform(0.5, 2.0)),
            ),
            transaction=TransactionEvaluation(
                amount=amount,
                amount_to_avg_ratio=amount_to_avg,
                merchant_risk_score=int(self.rng.integers(5, 45)),
                is_returned=False,
            ),
            identity_changes=IdentityChangeInfo(),
            is_fraudulent=False,
            fraud_type=None,
        )

    def tick(self, simulation_date: datetime | None = None) -> TransactionResult:
        """Generate the next transaction for this user.

        This method:
        1. Increments the sequence_number (1, 2, 3...)
        2. Ensures timestamps are strictly monotonic
        3. Handles fraud profile transitions (pre-fraud -> fraud -> post-fraud)
        4. Applies label delay simulation for fraud confirmation

        Args:
            simulation_date: Current date in simulation (for label delay).

        Returns:
            TransactionResult with record, metadata, and fraud status.
        """
        self.sequence_number += 1

        if simulation_date is None:
            simulation_date = datetime.now()

        is_fraud_event = False
        record: GeneratedRecord

        # Check if we should trigger fraud
        if self.fraud_profile is not None and not self._fraud_triggered:
            # For Sleeper profile: check dormancy and generate burst first
            if isinstance(self.fraud_profile, SleeperProfile):
                dormant_threshold = self.fraud_profile.dormant_days
                burst_threshold = self.fraud_profile.burst_connections

                # Force dormancy gap after initial transactions
                if (
                    not self._dormancy_triggered
                    and self._target_pre_fraud_count is not None
                    and self.state.transaction_count >= self._target_pre_fraud_count
                ):
                    # Jump forward in time to simulate dormancy
                    dormancy_days = dormant_threshold + float(self.rng.uniform(1, 10))
                    self._current_time += timedelta(days=dormancy_days)
                    self.state.days_since_last_activity = dormancy_days
                    self._dormancy_triggered = True

                # Once dormancy is triggered, continue with burst and fraud
                if self._dormancy_triggered:
                    # Generate burst events first
                    if self._burst_events_generated < burst_threshold:
                        record = self.fraud_profile.generate_link_burst_event(
                            self, self.state
                        )
                        self._burst_events_generated += 1
                    else:
                        # After burst, generate fraud
                        record = self.fraud_profile.generate_fraud_transaction(
                            self, self.state
                        )
                        is_fraud_event = True
                        self._fraud_triggered = True
                        self._fraud_transaction_time = record.transaction_timestamp

                        # Calculate fraud confirmation with label delay
                        (
                            self._fraud_confirmed_at,
                            is_detected,
                        ) = self.label_delay.calculate_confirmation_time(
                            record.transaction_timestamp,
                            simulation_date,
                        )

                        # If not detected yet, appears clean
                        if not is_detected:
                            record = record.model_copy(
                                update={
                                    "is_fraudulent": False,
                                    "fraud_type": None,
                                }
                            )
                else:
                    # Still building history, generate legitimate transaction
                    record = self._generate_legitimate_transaction()

            # For Bust-Out profile: check transaction count
            elif isinstance(self.fraud_profile, BustOutProfile):
                if (
                    self._target_pre_fraud_count is not None
                    and self.state.transaction_count >= self._target_pre_fraud_count
                ):
                    record = self.fraud_profile.generate_fraud_transaction(
                        self, self.state
                    )
                    is_fraud_event = True
                    self._fraud_triggered = True
                    self._fraud_transaction_time = record.transaction_timestamp

                    # Calculate fraud confirmation with label delay
                    (
                        self._fraud_confirmed_at,
                        is_detected,
                    ) = self.label_delay.calculate_confirmation_time(
                        record.transaction_timestamp,
                        simulation_date,
                    )

                    # If not detected yet, appears clean
                    if not is_detected:
                        record = record.model_copy(
                            update={
                                "is_fraudulent": False,
                                "fraud_type": None,
                            }
                        )
                else:
                    record = self._generate_legitimate_transaction()
            else:
                record = self._generate_legitimate_transaction()
        else:
            record = self._generate_legitimate_transaction()

        # Update state
        self.state.update_after_transaction(
            record.transaction.amount,
            record.transaction_timestamp,
            record.connection.bank_connections_count_24h,
        )

        # Generate evaluation metadata
        if self._fraud_confirmed_at is not None:
            delta = self._fraud_confirmed_at - record.transaction_timestamp
            days_to_fraud = delta.total_seconds() / 86400
            is_pre_fraud = days_to_fraud > 0

            # For Bust-Out: fraud event is train-eligible
            # For Sleeper: burst events are pre-fraud (train-eligible)
            if is_fraud_event and isinstance(self.fraud_profile, BustOutProfile):
                is_train_eligible = True
            else:
                is_train_eligible = is_pre_fraud
        else:
            days_to_fraud = None
            is_pre_fraud = True
            is_train_eligible = True

        metadata = EvaluationMetadata(
            user_id=self.user_id,
            record_id=record.record_id,
            sequence_number=self.sequence_number,
            fraud_confirmed_at=self._fraud_confirmed_at,
            is_pre_fraud=is_pre_fraud,
            days_to_fraud=days_to_fraud,
            is_train_eligible=is_train_eligible,
        )

        return TransactionResult(
            record=record,
            metadata=metadata,
            is_fraud_event=is_fraud_event,
        )

    def generate_full_sequence(
        self,
        num_transactions: int | None = None,
        post_fraud_transactions: int = 3,
        simulation_date: datetime | None = None,
    ) -> tuple[list[GeneratedRecord], list[EvaluationMetadata]]:
        """Generate a complete transaction sequence for this user.

        Args:
            num_transactions: Total transactions (auto-determined if None).
            post_fraud_transactions: Transactions after fraud event.
            simulation_date: Current date in simulation.

        Returns:
            Tuple of (records, metadata) lists.
        """
        records: list[GeneratedRecord] = []
        metadata: list[EvaluationMetadata] = []

        if num_transactions is None:
            has_fraud_profile = (
                self.fraud_profile is not None
                and self._target_pre_fraud_count is not None
            )
            if has_fraud_profile:
                # Use the pre-computed target count for consistency
                num_transactions = (
                    self._target_pre_fraud_count
                    + 1  # fraud event
                    + post_fraud_transactions
                )
                # For sleeper, add burst events
                if isinstance(self.fraud_profile, SleeperProfile):
                    num_transactions += self.fraud_profile.burst_connections
            else:
                num_transactions = int(self.rng.integers(5, 20))

        for _ in range(num_transactions):
            result = self.tick(simulation_date)
            records.append(result.record)
            metadata.append(result.metadata)

        return records, metadata


@dataclass
class GenerationConfig:
    """Configuration for batch data generation."""

    num_users: int = 100
    fraud_rate: float = 0.1
    bust_out_ratio: float = 0.5  # Of fraud users
    sleeper_ratio: float = 0.5  # Of fraud users
    seed: int | None = None
    simulation_date: datetime | None = None


def pydantic_to_db(record: GeneratedRecord) -> GeneratedRecordDB:
    """Convert a Pydantic GeneratedRecord to SQLAlchemy model."""
    return GeneratedRecordDB(
        record_id=record.record_id,
        user_id=record.user_id,
        full_name=record.full_name,
        email=record.email,
        phone=record.phone,
        transaction_timestamp=record.transaction_timestamp,
        is_off_hours_txn=record.is_off_hours_txn,
        available_balance=record.account.available_balance,
        balance_to_transaction_ratio=record.account.balance_to_transaction_ratio,
        avg_available_balance_30d=record.behavior.avg_available_balance_30d,
        balance_volatility_z_score=record.behavior.balance_volatility_z_score,
        bank_connections_count_24h=record.connection.bank_connections_count_24h,
        bank_connections_count_7d=record.connection.bank_connections_count_7d,
        bank_connections_avg_30d=record.connection.bank_connections_avg_30d,
        amount=record.transaction.amount,
        amount_to_avg_ratio=record.transaction.amount_to_avg_ratio,
        merchant_risk_score=record.transaction.merchant_risk_score,
        is_returned=record.transaction.is_returned,
        email_changed_at=record.identity_changes.email_changed_at,
        phone_changed_at=record.identity_changes.phone_changed_at,
        is_fraudulent=record.is_fraudulent,
        fraud_type=record.fraud_type,
    )


def metadata_to_db(meta: EvaluationMetadata) -> EvaluationMetadataDB:
    """Convert a Pydantic EvaluationMetadata to SQLAlchemy model."""
    return EvaluationMetadataDB(
        user_id=meta.user_id,
        record_id=meta.record_id,
        sequence_number=meta.sequence_number,
        fraud_confirmed_at=meta.fraud_confirmed_at,
        is_pre_fraud=meta.is_pre_fraud,
        days_to_fraud=meta.days_to_fraud,
        is_train_eligible=meta.is_train_eligible,
    )


def generate_and_persist(
    config: GenerationConfig,
    database_url: str | None = None,
    batch_size: int = 500,
) -> tuple[int, int]:
    """Generate synthetic data and persist to database.

    Uses SQLAlchemy to batch insert into both generated_records and
    evaluation_metadata tables within a single transaction to ensure
    referential integrity.

    Args:
        config: Generation configuration.
        database_url: Database connection URL.
        batch_size: Batch size for inserts.

    Returns:
        Tuple of (records_inserted, metadata_inserted).
    """
    rng = np.random.default_rng(config.seed)

    # Calculate user distribution
    num_fraud_users = int(config.num_users * config.fraud_rate)
    num_bust_out = int(num_fraud_users * config.bust_out_ratio)
    num_sleeper = num_fraud_users - num_bust_out
    num_legitimate = config.num_users - num_fraud_users

    all_records: list[GeneratedRecord] = []
    all_metadata: list[EvaluationMetadata] = []

    # Generate legitimate users
    for i in range(num_legitimate):
        seed = int(rng.integers(0, 2**31)) if config.seed else None
        simulator = UserSimulator(seed=seed)
        records, metadata = simulator.generate_full_sequence(
            simulation_date=config.simulation_date
        )
        all_records.extend(records)
        all_metadata.extend(metadata)

    # Generate Bust-Out fraud users
    for i in range(num_bust_out):
        seed = int(rng.integers(0, 2**31)) if config.seed else None
        simulator = UserSimulator(
            fraud_profile=BustOutProfile(),
            seed=seed,
        )
        records, metadata = simulator.generate_full_sequence(
            simulation_date=config.simulation_date
        )
        all_records.extend(records)
        all_metadata.extend(metadata)

    # Generate Sleeper/ATO fraud users
    for i in range(num_sleeper):
        seed = int(rng.integers(0, 2**31)) if config.seed else None
        simulator = UserSimulator(
            fraud_profile=SleeperProfile(),
            seed=seed,
            # Start further back to allow dormancy period
            start_time=datetime.now() - timedelta(days=int(rng.integers(90, 180))),
        )
        records, metadata = simulator.generate_full_sequence(
            simulation_date=config.simulation_date
        )
        all_records.extend(records)
        all_metadata.extend(metadata)

    # Persist to database in a single transaction
    db = DatabaseSession(database_url=database_url)

    with db.get_session() as session:
        # Convert to DB models
        db_records = [pydantic_to_db(r) for r in all_records]
        db_metadata = [metadata_to_db(m) for m in all_metadata]

        # Batch insert records
        records_inserted = 0
        for i in range(0, len(db_records), batch_size):
            batch = db_records[i : i + batch_size]
            session.add_all(batch)
            records_inserted += len(batch)

        # Batch insert metadata
        metadata_inserted = 0
        for i in range(0, len(db_metadata), batch_size):
            batch = db_metadata[i : i + batch_size]
            session.add_all(batch)
            metadata_inserted += len(batch)

        # Commit as single transaction
        session.commit()

    return records_inserted, metadata_inserted
