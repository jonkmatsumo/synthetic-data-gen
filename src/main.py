"""CLI entry point for synthetic data generation."""

from typing import Annotated

import typer

from synthetic_pipeline.db import DatabaseSession, GeneratedRecordDB
from synthetic_pipeline.generator import DataGenerator, FraudType
from synthetic_pipeline.logging import configure_logging, get_logger
from synthetic_pipeline.models import GeneratedRecord

app = typer.Typer(
    name="synthetic-data-gen",
    help="Synthetic data generation pipeline for fraud detection.",
    add_completion=False,
)


def pydantic_to_db(record: GeneratedRecord) -> GeneratedRecordDB:
    """Convert a Pydantic GeneratedRecord to SQLAlchemy model."""
    return GeneratedRecordDB(
        record_id=record.record_id,
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


@app.command()
def seed(
    count: Annotated[
        int,
        typer.Option("--count", "-c", help="Total number of profiles to generate"),
    ] = 1000,
    fraud_rate: Annotated[
        float,
        typer.Option(
            "--fraud-rate",
            "-f",
            help="Fraction of profiles that should be fraudulent (0.0-1.0)",
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
) -> None:
    """Generate synthetic transaction profiles and seed the database.

    Generates a mix of legitimate and fraudulent profiles based on the
    specified fraud rate, then inserts them into PostgreSQL.

    Example:
        synthetic-data-gen seed --count 1000 --fraud-rate 0.05
    """
    # Configure logging
    log_level = "DEBUG" if verbose else "INFO"
    configure_logging(level=log_level, json_format=json_logs)
    log = get_logger("seed")

    # Validate inputs
    if not 0.0 <= fraud_rate <= 1.0:
        log.error("Invalid fraud rate", fraud_rate=fraud_rate)
        raise typer.BadParameter("Fraud rate must be between 0.0 and 1.0")

    if count < 1:
        log.error("Invalid count", count=count)
        raise typer.BadParameter("Count must be at least 1")

    # Calculate counts
    fraud_count = int(count * fraud_rate)
    legitimate_count = count - fraud_count

    log.info(
        "Starting data generation",
        total_count=count,
        fraud_rate=fraud_rate,
        fraud_count=fraud_count,
        legitimate_count=legitimate_count,
        seed=seed_value,
    )

    # Initialize generator
    generator = DataGenerator(seed=seed_value)

    # Generate legitimate records
    log.info("Generating legitimate profiles", count=legitimate_count)
    legitimate_records = generator.generate_legitimate(count=legitimate_count)
    log.debug("Legitimate profiles generated", count=len(legitimate_records))

    # Generate fraudulent records with mixed fraud types
    log.info("Generating fraudulent profiles", count=fraud_count)
    fraudulent_records: list[GeneratedRecord] = []

    if fraud_count > 0:
        # Distribute fraud types roughly evenly
        fraud_types = list(FraudType)
        per_type = fraud_count // len(fraud_types)
        remainder = fraud_count % len(fraud_types)

        for i, fraud_type in enumerate(fraud_types):
            type_count = per_type + (1 if i < remainder else 0)
            if type_count > 0:
                records = generator.generate_fraudulent(fraud_type, count=type_count)
                fraudulent_records.extend(records)
                log.debug(
                    "Generated fraud type",
                    fraud_type=fraud_type.value,
                    count=type_count,
                )

    # Combine all records
    all_records = legitimate_records + fraudulent_records

    log.info(
        "Generation complete",
        total_generated=len(all_records),
        legitimate=len(legitimate_records),
        fraudulent=len(fraudulent_records),
    )

    # Log fraud type breakdown
    fraud_breakdown = {}
    for record in fraudulent_records:
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
            log.info("Database insert complete", records_inserted=inserted)

    except Exception as e:
        log.error(
            "Database operation failed", error=str(e), error_type=type(e).__name__
        )
        raise typer.Exit(code=1) from e

    # Final summary
    log.info(
        "Seeding complete",
        total_profiles=len(all_records),
        legitimate_profiles=len(legitimate_records),
        fraud_profiles=len(fraudulent_records),
        fraud_rate_actual=round(len(fraudulent_records) / len(all_records), 4)
        if all_records
        else 0,
    )

    typer.echo(
        f"\nSuccessfully generated and inserted {len(all_records)} profiles "
        f"({len(fraudulent_records)} fraudulent, {len(legitimate_records)} legitimate)"
    )


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
        # Total count
        total = session.scalar(select(func.count(GeneratedRecordDB.id)))

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

    log.info(
        "Database statistics",
        total_records=total,
        fraudulent=fraud,
        legitimate=total - fraud if total else 0,
        fraud_rate=round(fraud / total, 4) if total else 0,
    )

    typer.echo("\nDatabase Statistics:")
    typer.echo(f"  Total records: {total}")
    typer.echo(f"  Legitimate: {total - fraud if total else 0}")
    typer.echo(f"  Fraudulent: {fraud}")
    typer.echo(f"  Fraud rate: {round(fraud / total * 100, 2) if total else 0}%")

    if fraud_types:
        typer.echo("\nFraud Type Breakdown:")
        for fraud_type, count in fraud_types:
            typer.echo(f"  {fraud_type}: {count}")


if __name__ == "__main__":
    app()
