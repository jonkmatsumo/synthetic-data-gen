"""Database session management."""

import os
from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from synthetic_pipeline.db.models import Base


def get_database_url() -> str:
    """Build database URL from environment variables."""
    user = os.getenv("POSTGRES_USER", "synthetic")
    password = os.getenv("POSTGRES_PASSWORD", "synthetic_dev_password")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "synthetic_data")

    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


class DatabaseSession:
    """Database session manager with connection pooling."""

    def __init__(self, database_url: str | None = None, echo: bool = False):
        """Initialize database session manager.

        Args:
            database_url: Database connection URL. Defaults to env vars.
            echo: Whether to echo SQL statements.
        """
        self.database_url = database_url or get_database_url()
        self.engine = create_engine(
            self.database_url,
            echo=echo,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
        )
        self.SessionLocal = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False,
        )

    def create_tables(self) -> None:
        """Create all tables in the database."""
        Base.metadata.create_all(bind=self.engine)

    def drop_tables(self) -> None:
        """Drop all tables in the database."""
        Base.metadata.drop_all(bind=self.engine)

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get a database session context manager.

        Yields:
            Database session that auto-commits on success or rolls back on error.
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def batch_insert(
        self,
        session: Session,
        records: list,
        batch_size: int = 1000,
    ) -> int:
        """Batch insert records into the database.

        Args:
            session: Database session.
            records: List of SQLAlchemy model instances.
            batch_size: Number of records per batch.

        Returns:
            Total number of records inserted.
        """
        total = 0
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            session.add_all(batch)
            session.flush()
            total += len(batch)
        return total
