"""CLI entry point for synthetic data generation."""

from typing import Annotated

import typer

from synthetic_pipeline.db import (
    DatabaseSession,
    EvaluationMetadataDB,
    GeneratedRecordDB,
)
from synthetic_pipeline.generator import DataGenerator, FraudType
from synthetic_pipeline.logging import configure_logging, get_logger
from synthetic_pipeline.models import EvaluationMetadata, GeneratedRecord

app = typer.Typer(
    name="synthetic-data-gen",
    help="Synthetic data generation pipeline for fraud detection.",
    add_completion=False,
)


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


@app.command()
def seed(
    num_users: Annotated[
        int,
        typer.Option("--users", "-u", help="Number of unique users to generate"),
    ] = 100,
    fraud_rate: Annotated[
        float,
        typer.Option(
            "--fraud-rate",
            "-f",
            help="Fraction of users that should have fraud events (0.0-1.0)",
        ),
    ] = 0.05,
    seed_value: Annotated[
        int | None,
        typer.Option("--seed", "-s", help="Random seed for reproducibility"),
    ] = None,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", "-b", help="Batch size for database inserts"),
    ] = 500,
    database_url: Annotated[
        str | None,
        typer.Option("--database-url", envvar="DATABASE_URL", help="Database URL"),
    ] = None,
    drop_tables: Annotated[
        bool,
        typer.Option("--drop-tables", help="Drop existing tables before seeding"),
    ] = False,
    json_logs: Annotated[
        bool,
        typer.Option("--json-logs", help="Output logs in JSON format"),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose logging"),
    ] = False,
    legacy_mode: Annotated[
        bool,
        typer.Option(
            "--legacy",
            help="Use legacy generation (single transaction per user, no sequences)",
        ),
    ] = False,
) -> None:
    """Generate synthetic transaction profiles and seed the database.

    Generates user transaction sequences with both legitimate and fraudulent
    patterns, including evaluation metadata for proper train/test splitting.

    Example:
        synthetic-data-gen seed --users 100 --fraud-rate 0.05
    """
    # Configure logging
    log_level = "DEBUG" if verbose else "INFO"
    configure_logging(level=log_level, json_format=json_logs)
    log = get_logger("seed")

    # Validate inputs
    if not 0.0 <= fraud_rate <= 1.0:
        log.error("Invalid fraud rate", fraud_rate=fraud_rate)
        raise typer.BadParameter("Fraud rate must be between 0.0 and 1.0")

    if num_users < 1:
        log.error("Invalid user count", num_users=num_users)
        raise typer.BadParameter("User count must be at least 1")

    # Calculate counts
    num_fraud_users = int(num_users * fraud_rate)
    num_legitimate_users = num_users - num_fraud_users

    log.info(
        "Starting data generation",
        num_users=num_users,
        fraud_rate=fraud_rate,
        fraud_users=num_fraud_users,
        legitimate_users=num_legitimate_users,
        seed=seed_value,
        mode="legacy" if legacy_mode else "sequences",
    )

    # Initialize generator
    generator = DataGenerator(seed=seed_value)

    if legacy_mode:
        # Legacy mode: single transaction per user (backward compatible)
        log.info("Using legacy generation mode")

        legitimate_records = generator.generate_legitimate(count=num_legitimate_users)
        fraudulent_records: list[GeneratedRecord] = []

        if num_fraud_users > 0:
            fraud_types = list(FraudType)
            per_type = num_fraud_users // len(fraud_types)
            remainder = num_fraud_users % len(fraud_types)

            for i, fraud_type in enumerate(fraud_types):
                type_count = per_type + (1 if i < remainder else 0)
                if type_count > 0:
                    records = generator.generate_fraudulent(
                        fraud_type, count=type_count
                    )
                    fraudulent_records.extend(records)

        all_records = legitimate_records + fraudulent_records
        all_metadata: list[EvaluationMetadata] = []

        log.info(
            "Legacy generation complete",
            total_records=len(all_records),
            legitimate=len(legitimate_records),
            fraudulent=len(fraudulent_records),
        )
    else:
        # Sequence mode: multiple transactions per user with temporal tracking
        log.info("Generating user sequences with evaluation metadata")

        result = generator.generate_dataset_with_sequences(
            num_users=num_users,
            fraud_rate=fraud_rate,
        )
        all_records = result.records
        all_metadata = result.metadata

        # Count statistics
        fraud_records = [r for r in all_records if r.is_fraudulent]
        train_eligible = [m for m in all_metadata if m.is_train_eligible]
        unique_users = len(set(r.user_id for r in all_records))

        log.info(
            "Sequence generation complete",
            total_records=len(all_records),
            unique_users=unique_users,
            fraud_transactions=len(fraud_records),
            train_eligible_records=len(train_eligible),
            eval_only_records=len(all_metadata) - len(train_eligible),
        )

    # Log fraud type breakdown
    fraud_breakdown: dict[str, int] = {}
    for record in all_records:
        if record.is_fraudulent:
            fraud_type = record.fraud_type or "unknown"
            fraud_breakdown[fraud_type] = fraud_breakdown.get(fraud_type, 0) + 1

    if fraud_breakdown:
        log.info("Fraud type breakdown", **fraud_breakdown)

    # Database operations
    log.info("Connecting to database")
    db = DatabaseSession(database_url=database_url, echo=verbose)

    try:
        if drop_tables:
            log.warning("Dropping existing tables")
            db.drop_tables()

        log.info("Creating tables if not exist")
        db.create_tables()

        # Convert to DB models and insert
        log.info("Converting records to database models")
        db_records = [pydantic_to_db(r) for r in all_records]

        log.info(
            "Inserting records into database",
            record_count=len(db_records),
            batch_size=batch_size,
        )

        with db.get_session() as session:
            inserted = db.batch_insert(session, db_records, batch_size=batch_size)
            log.info("Records inserted", count=inserted)

            # Insert evaluation metadata if available
            if all_metadata:
                log.info("Inserting evaluation metadata", count=len(all_metadata))
                db_metadata = [metadata_to_db(m) for m in all_metadata]
                meta_inserted = db.batch_insert(
                    session, db_metadata, batch_size=batch_size
                )
                log.info("Metadata inserted", count=meta_inserted)

    except Exception as e:
        log.error(
            "Database operation failed", error=str(e), error_type=type(e).__name__
        )
        raise typer.Exit(code=1) from e

    # Final summary
    log.info(
        "Seeding complete",
        total_records=len(all_records),
        evaluation_metadata=len(all_metadata),
    )

    typer.echo(f"\nSuccessfully generated {len(all_records)} transaction records")
    if all_metadata:
        train_count = sum(1 for m in all_metadata if m.is_train_eligible)
        typer.echo(f"  Train-eligible: {train_count}")
        typer.echo(f"  Evaluation-only: {len(all_metadata) - train_count}")


@app.command()
def init_db(
    database_url: Annotated[
        str | None,
        typer.Option("--database-url", envvar="DATABASE_URL", help="Database URL"),
    ] = None,
    drop_tables: Annotated[
        bool,
        typer.Option("--drop-tables", help="Drop existing tables before creating"),
    ] = False,
) -> None:
    """Initialize the database schema without seeding data."""
    configure_logging()
    log = get_logger("init_db")

    log.info("Initializing database")
    db = DatabaseSession(database_url=database_url)

    if drop_tables:
        log.warning("Dropping existing tables")
        db.drop_tables()

    log.info("Creating tables")
    db.create_tables()

    log.info("Database initialization complete")
    typer.echo("Database initialized successfully")


@app.command()
def stats(
    database_url: Annotated[
        str | None,
        typer.Option("--database-url", envvar="DATABASE_URL", help="Database URL"),
    ] = None,
) -> None:
    """Show statistics about generated data in the database."""
    from sqlalchemy import func, select

    configure_logging()
    log = get_logger("stats")

    db = DatabaseSession(database_url=database_url)

    with db.get_session() as session:
        # Total records
        total = session.scalar(select(func.count(GeneratedRecordDB.id)))

        # Unique users
        unique_users = session.scalar(
            select(func.count(func.distinct(GeneratedRecordDB.user_id)))
        )

        # Fraud count
        fraud = session.scalar(
            select(func.count(GeneratedRecordDB.id)).where(
                GeneratedRecordDB.is_fraudulent.is_(True)
            )
        )

        # Fraud type breakdown
        fraud_types = session.execute(
            select(GeneratedRecordDB.fraud_type, func.count(GeneratedRecordDB.id))
            .where(GeneratedRecordDB.is_fraudulent.is_(True))
            .group_by(GeneratedRecordDB.fraud_type)
        ).all()

        # Evaluation metadata stats
        eval_total = session.scalar(select(func.count(EvaluationMetadataDB.id)))
        train_eligible = session.scalar(
            select(func.count(EvaluationMetadataDB.id)).where(
                EvaluationMetadataDB.is_train_eligible.is_(True)
            )
        )

    log.info(
        "Database statistics",
        total_records=total,
        unique_users=unique_users,
        fraudulent=fraud,
        legitimate=total - fraud if total else 0,
        fraud_rate=round(fraud / total, 4) if total else 0,
    )

    typer.echo("\nDatabase Statistics:")
    typer.echo(f"  Total records: {total}")
    typer.echo(f"  Unique users: {unique_users}")
    typer.echo(f"  Legitimate: {total - fraud if total else 0}")
    typer.echo(f"  Fraudulent: {fraud}")
    typer.echo(f"  Fraud rate: {round(fraud / total * 100, 2) if total else 0}%")

    if fraud_types:
        typer.echo("\nFraud Type Breakdown:")
        for fraud_type, count in fraud_types:
            typer.echo(f"  {fraud_type}: {count}")

    if eval_total:
        typer.echo("\nEvaluation Metadata:")
        typer.echo(f"  Total records: {eval_total}")
        typer.echo(f"  Train-eligible: {train_eligible}")
        typer.echo(f"  Evaluation-only: {eval_total - train_eligible}")


if __name__ == "__main__":
    app()
