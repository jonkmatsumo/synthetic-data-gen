"""Synthetic data generator for legitimate and fraudulent transactions."""

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Literal

import numpy as np
from faker import Faker

from synthetic_pipeline.models import (
    AccountSnapshot,
    BehaviorMetrics,
    ConnectionMetrics,
    EvaluationMetadata,
    GeneratedRecord,
    IdentityChangeInfo,
    TransactionEvaluation,
)


class FraudType(str, Enum):
    """Types of fraud scenarios."""

    LIQUIDITY_CRUNCH = "liquidity_crunch"
    LINK_BURST = "link_burst"
    ATO = "ato"


@dataclass
class UserSequenceResult:
    """Result of generating a user's transaction sequence."""

    records: list[GeneratedRecord]
    metadata: list[EvaluationMetadata]


class DataGenerator:
    """Generator for synthetic transaction data.

    Supports two modes:
    - Legitimate: Normal transaction patterns
    - Fraudulent: Specific fraud scenarios (liquidity crunch, link burst, ATO)

    Uses log-normal distribution for transaction amounts [cite: 58].
    """

    def __init__(self, seed: int | None = None):
        """Initialize the generator.

        Args:
            seed: Random seed for reproducibility.
        """
        self.rng = np.random.default_rng(seed)
        self.faker = Faker()
        if seed is not None:
            Faker.seed(seed)

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

    def _generate_timestamp(
        self,
        off_hours: bool = False,
        base_time: datetime | None = None,
    ) -> datetime:
        """Generate a transaction timestamp.

        Args:
            off_hours: If True, generate time between 10pm-6am [cite: 73].
            base_time: Base time to use. Defaults to now.
        """
        if base_time is None:
            base_time = datetime.now()

        if off_hours:
            # Off-hours: 10pm (22:00) to 6am (06:00)
            hour = self.rng.choice(list(range(22, 24)) + list(range(0, 6)))
        else:
            # Normal hours: 6am to 10pm
            hour = self.rng.integers(6, 22)

        return base_time.replace(
            hour=int(hour),
            minute=int(self.rng.integers(0, 60)),
            second=int(self.rng.integers(0, 60)),
            microsecond=0,
        )

    def _sample_lognormal_amount(
        self,
        mean: float = 75.0,
        sigma: float = 1.0,
    ) -> Decimal:
        """Sample transaction amount from log-normal distribution [cite: 58].

        Args:
            mean: Mean of the underlying normal distribution.
            sigma: Standard deviation of the underlying normal distribution.

        Returns:
            Transaction amount as Decimal with 2 decimal places.
        """
        # Log-normal parameters
        mu = np.log(mean) - (sigma**2) / 2
        amount = float(self.rng.lognormal(mu, sigma))
        # Clamp to reasonable range
        amount = max(0.01, min(amount, 50000.0))
        return Decimal(str(round(amount, 2)))

    def generate_legitimate(self, count: int = 1) -> list[GeneratedRecord]:
        """Generate legitimate transaction records.

        Characteristics:
        - Log-normal distribution for amounts [cite: 58]
        - balance_volatility_z_score between -1.0 and 1.0
        - merchant_risk_score averaging 20-30

        Args:
            count: Number of records to generate.

        Returns:
            List of legitimate transaction records.
        """
        records = []

        for _ in range(count):
            user_id = self._generate_user_id()
            name, email, phone = self._generate_pii()
            timestamp = self._generate_timestamp(off_hours=False)
            amount = self._sample_lognormal_amount()

            # Legitimate balance: reasonable range
            available_balance = Decimal(
                str(round(float(self.rng.uniform(500, 15000)), 2))
            )
            avg_balance_30d = Decimal(
                str(round(float(self.rng.uniform(400, 12000)), 2))
            )

            # Normal volatility: z-score between -1.0 and 1.0
            volatility_z = float(self.rng.uniform(-1.0, 1.0))

            # Normal connection patterns
            connections_24h = int(self.rng.integers(0, 3))
            connections_7d = int(self.rng.integers(0, 10))
            connections_avg_30d = float(self.rng.uniform(0.5, 2.0))

            # Amount ratio close to 1.0 (normal spending)
            amount_to_avg = float(self.rng.uniform(0.3, 2.5))

            # Low merchant risk score (average 20-30)
            merchant_risk = int(self.rng.integers(5, 45))

            record = GeneratedRecord(
                record_id=self._generate_record_id(),
                user_id=user_id,
                full_name=name,
                email=email,
                phone=phone,
                transaction_timestamp=timestamp,
                is_off_hours_txn=False,
                account=AccountSnapshot(
                    available_balance=available_balance,
                    balance_to_transaction_ratio=float(available_balance / amount)
                    if amount > 0
                    else 0.0,
                ),
                behavior=BehaviorMetrics(
                    avg_available_balance_30d=avg_balance_30d,
                    balance_volatility_z_score=volatility_z,
                ),
                connection=ConnectionMetrics(
                    bank_connections_count_24h=connections_24h,
                    bank_connections_count_7d=connections_7d,
                    bank_connections_avg_30d=connections_avg_30d,
                ),
                transaction=TransactionEvaluation(
                    amount=amount,
                    amount_to_avg_ratio=amount_to_avg,
                    merchant_risk_score=merchant_risk,
                    is_returned=False,
                ),
                identity_changes=IdentityChangeInfo(),
                is_fraudulent=False,
                fraud_type=None,
            )
            records.append(record)

        return records

    def generate_fraudulent(
        self,
        fraud_type: FraudType | Literal["liquidity_crunch", "link_burst", "ato"],
        count: int = 1,
    ) -> list[GeneratedRecord]:
        """Generate fraudulent transaction records.

        Args:
            fraud_type: Type of fraud scenario to generate.
            count: Number of records to generate.

        Returns:
            List of fraudulent transaction records.
        """
        if isinstance(fraud_type, str):
            fraud_type = FraudType(fraud_type)

        generators = {
            FraudType.LIQUIDITY_CRUNCH: self._generate_liquidity_crunch,
            FraudType.LINK_BURST: self._generate_link_burst,
            FraudType.ATO: self._generate_ato,
        }

        generator_fn = generators[fraud_type]
        return [generator_fn() for _ in range(count)]

    def _generate_liquidity_crunch(
        self, user_id: str | None = None, pii: tuple[str, str, str] | None = None
    ) -> GeneratedRecord:
        """Generate a liquidity crunch fraud scenario.

        Characteristics:
        - Low available_balance
        - balance_volatility_z_score < -2.5
        """
        if user_id is None:
            user_id = self._generate_user_id()
        if pii is None:
            pii = self._generate_pii()
        name, email, phone = pii

        timestamp = self._generate_timestamp()
        amount = self._sample_lognormal_amount(mean=150.0)

        # Low balance indicating liquidity issues
        available_balance = Decimal(str(round(float(self.rng.uniform(5, 100)), 2)))

        # Very low 30d average (depleting account)
        avg_balance_30d = Decimal(str(round(float(self.rng.uniform(50, 300)), 2)))

        # High negative volatility z-score (< -2.5)
        volatility_z = float(self.rng.uniform(-4.0, -2.6))

        # Normal-ish other metrics to isolate the signal
        connections_24h = int(self.rng.integers(0, 3))
        connections_7d = int(self.rng.integers(0, 8))
        amount_to_avg = float(self.rng.uniform(0.8, 2.0))
        merchant_risk = int(self.rng.integers(15, 50))

        return GeneratedRecord(
            record_id=self._generate_record_id(),
            user_id=user_id,
            full_name=name,
            email=email,
            phone=phone,
            transaction_timestamp=timestamp,
            is_off_hours_txn=False,
            account=AccountSnapshot(
                available_balance=available_balance,
                balance_to_transaction_ratio=float(available_balance / amount)
                if amount > 0
                else 0.0,
            ),
            behavior=BehaviorMetrics(
                avg_available_balance_30d=avg_balance_30d,
                balance_volatility_z_score=volatility_z,
            ),
            connection=ConnectionMetrics(
                bank_connections_count_24h=connections_24h,
                bank_connections_count_7d=connections_7d,
                bank_connections_avg_30d=1.0,
            ),
            transaction=TransactionEvaluation(
                amount=amount,
                amount_to_avg_ratio=amount_to_avg,
                merchant_risk_score=merchant_risk,
                is_returned=True,  # Likely to be returned due to insufficient funds
            ),
            identity_changes=IdentityChangeInfo(),
            is_fraudulent=True,
            fraud_type=FraudType.LIQUIDITY_CRUNCH.value,
        )

    def _generate_link_burst(
        self, user_id: str | None = None, pii: tuple[str, str, str] | None = None
    ) -> GeneratedRecord:
        """Generate a link burst fraud scenario.

        Characteristics:
        - bank_connections_count_24h between 5 and 15 (anomaly threshold > 4)
        """
        if user_id is None:
            user_id = self._generate_user_id()
        if pii is None:
            pii = self._generate_pii()
        name, email, phone = pii

        timestamp = self._generate_timestamp()
        amount = self._sample_lognormal_amount()

        # Normal account state
        available_balance = Decimal(str(round(float(self.rng.uniform(500, 5000)), 2)))
        avg_balance_30d = Decimal(str(round(float(self.rng.uniform(400, 4000)), 2)))
        volatility_z = float(self.rng.uniform(-1.0, 0.5))

        # Anomalous connection pattern: 5-15 connections in 24h
        connections_24h = int(self.rng.integers(5, 16))
        # Also elevated 7d count
        connections_7d = int(self.rng.integers(15, 50))
        connections_avg_30d = float(self.rng.uniform(1.0, 3.0))

        amount_to_avg = float(self.rng.uniform(0.5, 2.0))
        merchant_risk = int(self.rng.integers(20, 60))

        return GeneratedRecord(
            record_id=self._generate_record_id(),
            user_id=user_id,
            full_name=name,
            email=email,
            phone=phone,
            transaction_timestamp=timestamp,
            is_off_hours_txn=False,
            account=AccountSnapshot(
                available_balance=available_balance,
                balance_to_transaction_ratio=float(available_balance / amount)
                if amount > 0
                else 0.0,
            ),
            behavior=BehaviorMetrics(
                avg_available_balance_30d=avg_balance_30d,
                balance_volatility_z_score=volatility_z,
            ),
            connection=ConnectionMetrics(
                bank_connections_count_24h=connections_24h,
                bank_connections_count_7d=connections_7d,
                bank_connections_avg_30d=connections_avg_30d,
            ),
            transaction=TransactionEvaluation(
                amount=amount,
                amount_to_avg_ratio=amount_to_avg,
                merchant_risk_score=merchant_risk,
                is_returned=False,
            ),
            identity_changes=IdentityChangeInfo(),
            is_fraudulent=True,
            fraud_type=FraudType.LINK_BURST.value,
        )

    def _generate_ato(
        self, user_id: str | None = None, pii: tuple[str, str, str] | None = None
    ) -> GeneratedRecord:
        """Generate an Account Takeover (ATO) fraud scenario.

        Characteristics [cite: 73, 88]:
        - Sudden spending deviation: amount_to_avg_ratio > 5.0
        - Temporal anomaly: is_off_hours_txn = True
        - Identity change: email/phone changed < 72 hours before transaction
        """
        if user_id is None:
            user_id = self._generate_user_id()
        if pii is None:
            pii = self._generate_pii()
        name, email, phone = pii

        # Transaction during off-hours
        timestamp = self._generate_timestamp(off_hours=True)

        # Higher than normal amount
        amount = self._sample_lognormal_amount(mean=500.0, sigma=1.2)

        # Account looks normal
        available_balance = Decimal(str(round(float(self.rng.uniform(1000, 10000)), 2)))
        avg_balance_30d = Decimal(str(round(float(self.rng.uniform(800, 8000)), 2)))
        volatility_z = float(self.rng.uniform(-0.5, 0.5))

        # Normal connections
        connections_24h = int(self.rng.integers(0, 3))
        connections_7d = int(self.rng.integers(0, 10))

        # High spending deviation (> 5.0)
        amount_to_avg = float(self.rng.uniform(5.5, 15.0))

        # Potentially higher merchant risk
        merchant_risk = int(self.rng.integers(40, 85))

        # Identity change within 72 hours [cite: 88]
        hours_ago = float(self.rng.uniform(1, 70))
        change_time = timestamp - timedelta(hours=hours_ago)

        # Randomly choose email or phone change (or both)
        change_type = self.rng.choice(["email", "phone", "both"])
        identity_changes = IdentityChangeInfo(
            email_changed_at=change_time if change_type in ("email", "both") else None,
            phone_changed_at=change_time if change_type in ("phone", "both") else None,
        )

        return GeneratedRecord(
            record_id=self._generate_record_id(),
            user_id=user_id,
            full_name=name,
            email=email,
            phone=phone,
            transaction_timestamp=timestamp,
            is_off_hours_txn=True,
            account=AccountSnapshot(
                available_balance=available_balance,
                balance_to_transaction_ratio=float(available_balance / amount)
                if amount > 0
                else 0.0,
            ),
            behavior=BehaviorMetrics(
                avg_available_balance_30d=avg_balance_30d,
                balance_volatility_z_score=volatility_z,
            ),
            connection=ConnectionMetrics(
                bank_connections_count_24h=connections_24h,
                bank_connections_count_7d=connections_7d,
                bank_connections_avg_30d=1.5,
            ),
            transaction=TransactionEvaluation(
                amount=amount,
                amount_to_avg_ratio=amount_to_avg,
                merchant_risk_score=merchant_risk,
                is_returned=False,
            ),
            identity_changes=identity_changes,
            is_fraudulent=True,
            fraud_type=FraudType.ATO.value,
        )

    def _generate_legitimate_transaction(
        self,
        user_id: str,
        pii: tuple[str, str, str],
        base_time: datetime,
    ) -> GeneratedRecord:
        """Generate a single legitimate transaction for a user sequence."""
        name, email, phone = pii
        timestamp = self._generate_timestamp(off_hours=False, base_time=base_time)
        amount = self._sample_lognormal_amount()

        available_balance = Decimal(str(round(float(self.rng.uniform(500, 15000)), 2)))
        avg_balance_30d = Decimal(str(round(float(self.rng.uniform(400, 12000)), 2)))
        volatility_z = float(self.rng.uniform(-1.0, 1.0))
        connections_24h = int(self.rng.integers(0, 3))
        connections_7d = int(self.rng.integers(0, 10))
        connections_avg_30d = float(self.rng.uniform(0.5, 2.0))
        amount_to_avg = float(self.rng.uniform(0.3, 2.5))
        merchant_risk = int(self.rng.integers(5, 45))

        return GeneratedRecord(
            record_id=self._generate_record_id(),
            user_id=user_id,
            full_name=name,
            email=email,
            phone=phone,
            transaction_timestamp=timestamp,
            is_off_hours_txn=False,
            account=AccountSnapshot(
                available_balance=available_balance,
                balance_to_transaction_ratio=float(available_balance / amount)
                if amount > 0
                else 0.0,
            ),
            behavior=BehaviorMetrics(
                avg_available_balance_30d=avg_balance_30d,
                balance_volatility_z_score=volatility_z,
            ),
            connection=ConnectionMetrics(
                bank_connections_count_24h=connections_24h,
                bank_connections_count_7d=connections_7d,
                bank_connections_avg_30d=connections_avg_30d,
            ),
            transaction=TransactionEvaluation(
                amount=amount,
                amount_to_avg_ratio=amount_to_avg,
                merchant_risk_score=merchant_risk,
                is_returned=False,
            ),
            identity_changes=IdentityChangeInfo(),
            is_fraudulent=False,
            fraud_type=None,
        )

    def generate_user_sequence(
        self,
        is_fraud_user: bool = False,
        fraud_type: FraudType | None = None,
        min_transactions: int = 3,
        max_transactions: int = 15,
        post_fraud_transactions: int = 2,
    ) -> UserSequenceResult:
        """Generate a sequence of transactions for a single user.

        For fraud users, generates pre-fraud legitimate transactions,
        the fraud transaction, and optionally post-fraud transactions.

        Args:
            is_fraud_user: Whether this user will have a fraud event.
            fraud_type: Type of fraud (required if is_fraud_user=True).
            min_transactions: Minimum transactions before fraud.
            max_transactions: Maximum transactions before fraud.
            post_fraud_transactions: Number of transactions after fraud detection.

        Returns:
            UserSequenceResult with records and evaluation metadata.
        """
        user_id = self._generate_user_id()
        pii = self._generate_pii()
        records: list[GeneratedRecord] = []
        metadata: list[EvaluationMetadata] = []

        # Start date for this user's history (30-180 days ago)
        history_days = int(self.rng.integers(30, 180))
        start_date = datetime.now() - timedelta(days=history_days)

        # Number of pre-fraud transactions
        num_pre_fraud = int(self.rng.integers(min_transactions, max_transactions + 1))

        # Generate timestamps spread across the history period
        if is_fraud_user:
            # Reserve last portion for fraud event
            fraud_day = history_days - int(self.rng.integers(1, 10))
            fraud_time = start_date + timedelta(days=fraud_day)
            fraud_confirmed_at = fraud_time + timedelta(
                hours=float(self.rng.uniform(1, 48))
            )
        else:
            fraud_time = None
            fraud_confirmed_at = None

        # Generate pre-fraud transactions
        for i in range(num_pre_fraud):
            # Spread transactions across the time period
            if is_fraud_user:
                # Before fraud event
                days_offset = float(self.rng.uniform(0, fraud_day - 1))
            else:
                days_offset = float(self.rng.uniform(0, history_days))

            txn_time = start_date + timedelta(days=days_offset)
            record = self._generate_legitimate_transaction(user_id, pii, txn_time)
            records.append(record)

        # Generate fraud transaction if applicable
        if is_fraud_user and fraud_type is not None:
            generators = {
                FraudType.LIQUIDITY_CRUNCH: self._generate_liquidity_crunch,
                FraudType.LINK_BURST: self._generate_link_burst,
                FraudType.ATO: self._generate_ato,
            }
            fraud_record = generators[fraud_type](user_id=user_id, pii=pii)
            # Override timestamp to be at fraud_time
            fraud_record = fraud_record.model_copy(
                update={"transaction_timestamp": fraud_time}
            )
            records.append(fraud_record)

            # Generate post-fraud transactions (after fraud was detected)
            for _ in range(post_fraud_transactions):
                days_after = float(self.rng.uniform(1, 14))
                post_time = fraud_time + timedelta(days=days_after)
                post_record = self._generate_legitimate_transaction(
                    user_id, pii, post_time
                )
                records.append(post_record)

        # Sort records by timestamp
        records.sort(key=lambda r: r.transaction_timestamp)

        # Generate evaluation metadata
        for seq_num, record in enumerate(records, start=1):
            if fraud_confirmed_at is not None:
                delta = fraud_confirmed_at - record.transaction_timestamp
                days_to_fraud = delta.total_seconds() / 86400
                is_pre_fraud = days_to_fraud > 0
                # Only train-eligible if pre-fraud
                is_train_eligible = is_pre_fraud
            else:
                days_to_fraud = None
                is_pre_fraud = True
                is_train_eligible = True

            meta = EvaluationMetadata(
                user_id=user_id,
                record_id=record.record_id,
                sequence_number=seq_num,
                fraud_confirmed_at=fraud_confirmed_at,
                is_pre_fraud=is_pre_fraud,
                days_to_fraud=days_to_fraud,
                is_train_eligible=is_train_eligible,
            )
            metadata.append(meta)

        return UserSequenceResult(records=records, metadata=metadata)

    def generate_dataset_with_sequences(
        self,
        num_users: int = 100,
        fraud_rate: float = 0.05,
    ) -> UserSequenceResult:
        """Generate a complete dataset with user sequences.

        Args:
            num_users: Total number of unique users.
            fraud_rate: Fraction of users who will have fraud events.

        Returns:
            UserSequenceResult with all records and metadata.
        """
        all_records: list[GeneratedRecord] = []
        all_metadata: list[EvaluationMetadata] = []

        num_fraud_users = int(num_users * fraud_rate)
        num_legitimate_users = num_users - num_fraud_users

        fraud_types = list(FraudType)

        # Generate legitimate user sequences
        for _ in range(num_legitimate_users):
            result = self.generate_user_sequence(is_fraud_user=False)
            all_records.extend(result.records)
            all_metadata.extend(result.metadata)

        # Generate fraud user sequences
        for i in range(num_fraud_users):
            fraud_type = fraud_types[i % len(fraud_types)]
            result = self.generate_user_sequence(
                is_fraud_user=True, fraud_type=fraud_type
            )
            all_records.extend(result.records)
            all_metadata.extend(result.metadata)

        return UserSequenceResult(records=all_records, metadata=all_metadata)
