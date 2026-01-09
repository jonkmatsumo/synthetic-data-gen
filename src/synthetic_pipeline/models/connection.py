"""Connection metrics model."""

from pydantic import BaseModel, Field


class ConnectionMetrics(BaseModel):
    """Metrics tracking bank connection patterns."""

    bank_connections_count_24h: int = Field(
        ...,
        ge=0,
        description=(
            "Number of bank connections in the last 24 hours. Anomaly threshold > 4"
        ),
        examples=[1, 3, 7],
    )

    bank_connections_count_7d: int = Field(
        ...,
        ge=0,
        description=(
            "Number of bank connections in the last 7 days. "
            "Anomaly if > 300% of 30-day average"
        ),
        examples=[5, 12, 45],
    )

    bank_connections_avg_30d: float = Field(
        default=0.0,
        ge=0.0,
        description="Average daily bank connections over the last 30 days",
        examples=[2.5, 1.0, 8.3],
    )

    @property
    def is_24h_anomaly(self) -> bool:
        """Check if 24h connection count exceeds anomaly threshold (> 4)."""
        return self.bank_connections_count_24h > 4

    @property
    def is_7d_anomaly(self) -> bool:
        """Check if 7d connection count exceeds 300% of 30d average."""
        if self.bank_connections_avg_30d == 0:
            return self.bank_connections_count_7d > 0
        expected_7d = self.bank_connections_avg_30d * 7
        return self.bank_connections_count_7d > (expected_7d * 3.0)
