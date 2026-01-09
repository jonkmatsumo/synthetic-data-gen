"""SQLAlchemy models mirroring Pydantic models."""

from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    Numeric,
    String,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    pass


class GeneratedRecordDB(Base):
    """SQLAlchemy model for generated records."""

    __tablename__ = "generated_records"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    record_id: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)

    # PII fields
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str] = mapped_column(String(255), nullable=False)
    phone: Mapped[str] = mapped_column(String(50), nullable=False)

    # Timestamps
    transaction_timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    is_off_hours_txn: Mapped[bool] = mapped_column(Boolean, default=False)

    # Account snapshot
    available_balance: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False)
    balance_to_transaction_ratio: Mapped[float] = mapped_column(Float, nullable=False)

    # Behavior metrics
    avg_available_balance_30d: Mapped[Decimal] = mapped_column(
        Numeric(18, 2), nullable=False
    )
    balance_volatility_z_score: Mapped[float] = mapped_column(Float, nullable=False)

    # Connection metrics
    bank_connections_count_24h: Mapped[int] = mapped_column(Integer, nullable=False)
    bank_connections_count_7d: Mapped[int] = mapped_column(Integer, nullable=False)
    bank_connections_avg_30d: Mapped[float] = mapped_column(Float, nullable=False)

    # Transaction evaluation
    amount: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False)
    amount_to_avg_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    merchant_risk_score: Mapped[int] = mapped_column(Integer, nullable=False)
    is_returned: Mapped[bool] = mapped_column(Boolean, default=False)

    # Identity changes
    email_changed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    phone_changed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Labels
    is_fraudulent: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    fraud_type: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    __table_args__ = (
        Index("ix_fraud_type", "fraud_type"),
        Index("ix_transaction_timestamp", "transaction_timestamp"),
    )

    def __repr__(self) -> str:
        return (
            f"<GeneratedRecordDB(record_id={self.record_id}, "
            f"is_fraudulent={self.is_fraudulent})>"
        )


# Separate normalized tables for more complex queries


class AccountSnapshotRecord(Base):
    """Normalized account snapshot records."""

    __tablename__ = "account_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    record_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    available_balance: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False)
    balance_to_transaction_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )


class BehaviorMetricsRecord(Base):
    """Normalized behavior metrics records."""

    __tablename__ = "behavior_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    record_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    avg_available_balance_30d: Mapped[Decimal] = mapped_column(
        Numeric(18, 2), nullable=False
    )
    balance_volatility_z_score: Mapped[float] = mapped_column(Float, nullable=False)
    is_high_risk: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )


class ConnectionMetricsRecord(Base):
    """Normalized connection metrics records."""

    __tablename__ = "connection_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    record_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    bank_connections_count_24h: Mapped[int] = mapped_column(Integer, nullable=False)
    bank_connections_count_7d: Mapped[int] = mapped_column(Integer, nullable=False)
    bank_connections_avg_30d: Mapped[float] = mapped_column(Float, nullable=False)
    is_24h_anomaly: Mapped[bool] = mapped_column(Boolean, default=False)
    is_7d_anomaly: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )


class TransactionEvaluationRecord(Base):
    """Normalized transaction evaluation records."""

    __tablename__ = "transaction_evaluations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    record_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    amount: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False)
    amount_to_avg_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    merchant_risk_score: Mapped[int] = mapped_column(Integer, nullable=False)
    is_returned: Mapped[bool] = mapped_column(Boolean, default=False)
    is_amount_anomaly: Mapped[bool] = mapped_column(Boolean, default=False)
    is_merchant_anomaly: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )


class IdentityChangeRecord(Base):
    """Normalized identity change records."""

    __tablename__ = "identity_changes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    record_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    email_changed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    phone_changed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    has_recent_change: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
