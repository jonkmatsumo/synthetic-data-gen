"""Pydantic schemas for API request/response models."""

from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field


class Currency(str, Enum):
    """Supported currency codes."""

    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    CAD = "CAD"
    AUD = "AUD"


class SignalRequest(BaseModel):
    """Request schema for signal evaluation endpoint."""

    user_id: str = Field(
        ...,
        description="Unique identifier for the user",
        examples=["user_abc123"],
    )
    amount: Decimal = Field(
        ...,
        gt=0,
        description="Transaction amount",
        examples=[150.00],
    )
    currency: Currency = Field(
        default=Currency.USD,
        description="Currency code",
    )
    client_transaction_id: str = Field(
        ...,
        description="Client-provided transaction identifier for idempotency",
        examples=["txn_xyz789"],
    )


class RiskComponent(BaseModel):
    """Individual risk factor contributing to the score."""

    key: str = Field(
        ...,
        description="Machine-readable identifier for the risk factor",
        examples=["velocity", "history", "amount_ratio"],
    )
    label: str = Field(
        ...,
        description="Human-readable description of the risk factor",
        examples=["high_transaction_velocity", "insufficient_history"],
    )


class SignalResponse(BaseModel):
    """Response schema for signal evaluation endpoint."""

    request_id: str = Field(
        ...,
        description="Unique identifier for this evaluation request",
        examples=["req_123xyz"],
    )
    score: int = Field(
        ...,
        ge=1,
        le=99,
        description="Risk score from 1 (lowest risk) to 99 (highest risk)",
        examples=[85],
    )
    risk_components: list[RiskComponent] = Field(
        default_factory=list,
        description="List of risk factors contributing to the score",
    )
    model_version: str = Field(
        ...,
        description="Version of the model used for evaluation",
        examples=["v1.0.0"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "request_id": "req_123xyz",
                    "score": 85,
                    "risk_components": [
                        {"key": "velocity", "label": "high_transaction_velocity"},
                        {"key": "history", "label": "insufficient_history"},
                    ],
                    "model_version": "v1.0.0",
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""

    status: str = Field(default="healthy")
    model_loaded: bool = Field(default=False)
    version: str = Field(default="0.1.0")


class TrainRequest(BaseModel):
    """Request schema for model training endpoint."""

    max_depth: int = Field(
        default=6,
        ge=2,
        le=12,
        description="Maximum depth of XGBoost trees",
    )
    training_window_days: int = Field(
        default=30,
        ge=7,
        le=90,
        description="Number of days for training window",
    )


class TrainResponse(BaseModel):
    """Response schema for model training endpoint."""

    success: bool = Field(..., description="Whether training completed successfully")
    run_id: str | None = Field(None, description="MLflow run ID if successful")
    error: str | None = Field(None, description="Error message if training failed")
