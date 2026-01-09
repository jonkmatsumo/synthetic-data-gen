"""Generated record model combining all metrics."""

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, Field

from synthetic_pipeline.models.account import AccountSnapshot
from synthetic_pipeline.models.behavior import BehaviorMetrics
from synthetic_pipeline.models.connection import ConnectionMetrics
from synthetic_pipeline.models.transaction import TransactionEvaluation


class IdentityChangeInfo(BaseModel):
    """Tracks recent identity changes for ATO detection."""

    email_changed_at: datetime | None = Field(
        default=None,
        description="Timestamp of last email change",
        examples=[None, datetime(2024, 1, 15, 10, 30, 0)],
    )

    phone_changed_at: datetime | None = Field(
        default=None,
        description="Timestamp of last phone number change",
        examples=[None, datetime(2024, 1, 15, 14, 45, 0)],
    )

    def hours_since_email_change(self, reference_time: datetime) -> float | None:
        """Calculate hours since last email change."""
        if self.email_changed_at is None:
            return None
        delta = reference_time - self.email_changed_at
        return delta.total_seconds() / 3600

    def hours_since_phone_change(self, reference_time: datetime) -> float | None:
        """Calculate hours since last phone change."""
        if self.phone_changed_at is None:
            return None
        delta = reference_time - self.phone_changed_at
        return delta.total_seconds() / 3600

    def has_recent_change(
        self, reference_time: datetime, threshold_hours: float = 72
    ) -> bool:
        """Check if any identity change occurred within threshold hours."""
        email_hours = self.hours_since_email_change(reference_time)
        phone_hours = self.hours_since_phone_change(reference_time)

        if email_hours is not None and email_hours < threshold_hours:
            return True
        if phone_hours is not None and phone_hours < threshold_hours:
            return True
        return False


class EvaluationMetadata(BaseModel):
    """Metadata for model evaluation (not used in training).

    Tracks temporal relationships between transactions and fraud events
    to enable proper train/test splitting and model evaluation.
    """

    user_id: str = Field(
        ...,
        description="Persistent user identifier across transactions",
        examples=["user_abc123", "user_xyz789"],
    )

    record_id: str = Field(
        ...,
        description="FK to generated_records",
        examples=["rec_abc123"],
    )

    sequence_number: int = Field(
        ...,
        ge=1,
        description="Transaction order for this user (1-indexed)",
        examples=[1, 5, 12],
    )

    fraud_confirmed_at: datetime | None = Field(
        default=None,
        description="When fraud was confirmed/reported for this user",
        examples=[None, datetime(2024, 1, 20, 9, 0, 0)],
    )

    is_pre_fraud: bool = Field(
        default=True,
        description="Transaction occurred before fraud detection",
        examples=[True, False],
    )

    days_to_fraud: float | None = Field(
        default=None,
        description="Days until fraud event (negative if after, null if no fraud)",
        examples=[None, 5.5, -2.0],
    )

    is_train_eligible: bool = Field(
        default=True,
        description="Whether this record can be used for training",
        examples=[True, False],
    )


class GeneratedRecord(BaseModel):
    """Complete generated record combining all metrics and PII."""

    # Identifiers
    record_id: str = Field(
        ...,
        description="Unique identifier for this record",
        examples=["rec_abc123", "rec_xyz789"],
    )

    user_id: str = Field(
        ...,
        description="Persistent user identifier linking transactions",
        examples=["user_abc123", "user_xyz789"],
    )

    # PII fields
    full_name: str = Field(
        ...,
        description="Full name of the account holder",
        examples=["John Smith", "Jane Doe"],
    )

    email: str = Field(
        ...,
        description="Email address of the account holder",
        examples=["john.smith@example.com"],
    )

    phone: str = Field(
        ...,
        description="Phone number of the account holder",
        examples=["+1-555-123-4567"],
    )

    # Timestamps
    transaction_timestamp: datetime = Field(
        ...,
        description="Timestamp when the transaction occurred",
        examples=[datetime(2024, 1, 15, 14, 30, 0)],
    )

    is_off_hours_txn: bool = Field(
        default=False,
        description=(
            "Whether transaction occurred during off-hours (10pm-6am) [cite: 73]"
        ),
        examples=[False, True],
    )

    # Composite models
    account: AccountSnapshot = Field(
        ...,
        description="Account snapshot at time of transaction",
    )

    behavior: BehaviorMetrics = Field(
        ...,
        description="Behavioral metrics for the account",
    )

    connection: ConnectionMetrics = Field(
        ...,
        description="Connection metrics for the account",
    )

    transaction: TransactionEvaluation = Field(
        ...,
        description="Transaction evaluation metrics",
    )

    identity_changes: IdentityChangeInfo = Field(
        default_factory=IdentityChangeInfo,
        description="Identity change tracking for ATO detection [cite: 88]",
    )

    # Labels
    is_fraudulent: bool = Field(
        default=False,
        description="Whether this record represents fraudulent activity",
        examples=[False, True],
    )

    fraud_type: str | None = Field(
        default=None,
        description="Type of fraud if fraudulent",
        examples=[None, "liquidity_crunch", "link_burst", "ato"],
    )

    @property
    def avg_transaction_amount(self) -> Decimal:
        """Calculate implied average transaction from ratio."""
        if self.transaction.amount_to_avg_ratio == 0:
            return Decimal("0")
        return Decimal(
            str(float(self.transaction.amount) / self.transaction.amount_to_avg_ratio)
        ).quantize(Decimal("0.01"))
