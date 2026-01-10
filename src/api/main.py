"""FastAPI application for fraud signal evaluation.

This API provides idempotent risk assessment for transactions.
It does not modify transaction state - it only provides an evaluation.
"""

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from api.model_manager import get_model_manager
from api.schemas import (
    ClearDataResponse,
    GenerateDataRequest,
    GenerateDataResponse,
    HealthResponse,
    SignalRequest,
    SignalResponse,
    TrainRequest,
    TrainResponse,
)
from api.services import get_evaluator

if TYPE_CHECKING:
    from synthetic_pipeline.db.models import EvaluationMetadataDB, GeneratedRecordDB
    from synthetic_pipeline.models import EvaluationMetadata, GeneratedRecord

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _pydantic_to_db(record: "GeneratedRecord") -> "GeneratedRecordDB":
    """Convert a Pydantic GeneratedRecord to SQLAlchemy model."""
    from synthetic_pipeline.db.models import GeneratedRecordDB

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


def _metadata_to_db(meta: "EvaluationMetadata") -> "EvaluationMetadataDB":
    """Convert a Pydantic EvaluationMetadata to SQLAlchemy model."""
    from synthetic_pipeline.db.models import EvaluationMetadataDB

    return EvaluationMetadataDB(
        user_id=meta.user_id,
        record_id=meta.record_id,
        sequence_number=meta.sequence_number,
        fraud_confirmed_at=meta.fraud_confirmed_at,
        is_pre_fraud=meta.is_pre_fraud,
        days_to_fraud=meta.days_to_fraud,
        is_train_eligible=meta.is_train_eligible,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - load model on startup."""
    # Startup: Load the production model
    logger.info("Starting up - loading production model...")
    manager = get_model_manager()
    success = manager.load_production_model()

    if success:
        logger.info(
            f"Model loaded successfully: version={manager.model_version}, "
            f"source={manager.model_source}"
        )
    else:
        logger.warning("No model loaded - API will use rule-based evaluation only")

    yield

    # Shutdown: cleanup if needed
    logger.info("Shutting down...")


app = FastAPI(
    title="Fraud Signal API",
    description="Risk signal evaluation for fraud detection",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Check API health status.

    Returns:
        HealthResponse with status and model information.
    """
    manager = get_model_manager()
    evaluator = get_evaluator()

    # Use model manager version if available, otherwise fall back to evaluator
    version = manager.model_version if manager.model_loaded else evaluator.model_version

    return HealthResponse(
        status="healthy",
        model_loaded=manager.model_loaded,
        version=version,
    )


@app.post("/reload-model", tags=["System"])
async def reload_model() -> dict:
    """Reload the production model from MLflow.

    Call this endpoint after promoting a new model to production
    to pick up the latest version without restarting the API.

    Returns:
        Dict with success status and model version.
    """
    manager = get_model_manager()
    success = manager.load_production_model()

    if success:
        logger.info(f"Model reloaded: version={manager.model_version}")
        return {
            "success": True,
            "model_loaded": True,
            "version": manager.model_version,
            "source": manager.model_source,
        }
    else:
        logger.warning("Model reload failed")
        return {
            "success": False,
            "model_loaded": False,
            "version": None,
            "source": "none",
        }


@app.post(
    "/evaluate/signal",
    response_model=SignalResponse,
    tags=["Evaluation"],
    summary="Evaluate fraud signal",
    description="""
Evaluate the fraud risk signal for a transaction.

This endpoint is **idempotent** - it only provides an assessment without
modifying any transaction state. The same input will produce consistent
scoring (deterministic per user_id).

The response includes:
- **score**: Risk score from 1 (lowest) to 99 (highest)
- **risk_components**: Factors contributing to the score
- **model_version**: Version of the scoring model

Scores are calibrated to match real-world fraud score distributions where:
- Scores 1-20 are very common (low risk)
- Scores 80+ are rare (high risk)
""",
    responses={
        200: {
            "description": "Successful evaluation",
            "content": {
                "application/json": {
                    "example": {
                        "request_id": "req_123xyz",
                        "score": 85,
                        "risk_components": [
                            {
                                "key": "velocity",
                                "label": "high_transaction_velocity",
                            },
                            {"key": "history", "label": "insufficient_history"},
                        ],
                        "model_version": "v1.0.0",
                    }
                }
            },
        },
        422: {"description": "Validation error"},
    },
)
async def evaluate_signal(request: SignalRequest) -> SignalResponse:
    """Evaluate fraud signal for a transaction.

    Args:
        request: Signal evaluation request with user_id, amount, currency,
            and client_transaction_id.

    Returns:
        SignalResponse with risk score and contributing factors.
    """
    try:
        evaluator = get_evaluator()
        return evaluator.evaluate(request)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {e!s}",
        ) from e


@app.post(
    "/data/generate",
    response_model=GenerateDataResponse,
    tags=["Data Management"],
    summary="Generate synthetic data",
    description="Generate synthetic transaction data with configurable fraud rate.",
)
async def generate_data(request: GenerateDataRequest) -> GenerateDataResponse:
    """Generate synthetic transaction data.

    Args:
        request: Generation request with num_users, fraud_rate, and drop_existing.

    Returns:
        GenerateDataResponse with counts of generated records.
    """
    try:
        from pipeline.materialize_features import FeatureMaterializer
        from synthetic_pipeline.db.models import Base
        from synthetic_pipeline.db.session import DatabaseSession
        from synthetic_pipeline.generator import DataGenerator

        # Generate data
        generator = DataGenerator()
        result = generator.generate_dataset_with_sequences(
            num_users=request.num_users,
            fraud_rate=request.fraud_rate,
        )

        # Count fraud records
        fraud_count = sum(1 for r in result.records if r.is_fraudulent)

        # Connect to database
        db_session = DatabaseSession()

        with db_session.get_session() as session:
            if request.drop_existing:
                # Drop and recreate tables
                Base.metadata.drop_all(db_session.engine)
                Base.metadata.create_all(db_session.engine)
            else:
                # Just ensure tables exist
                Base.metadata.create_all(db_session.engine)

            # Convert and insert records
            db_records = [_pydantic_to_db(record) for record in result.records]
            session.bulk_save_objects(db_records)

            # Insert metadata
            meta_records = [_metadata_to_db(meta) for meta in result.metadata]
            session.bulk_save_objects(meta_records)

            session.commit()

        # Materialize features
        materializer = FeatureMaterializer()
        materialize_stats = materializer.materialize_all()
        features_count = materialize_stats.get("total_processed", 0)

        return GenerateDataResponse(
            success=True,
            total_records=len(result.records),
            fraud_records=fraud_count,
            features_materialized=features_count,
        )

    except Exception as e:
        logger.exception("Data generation failed")
        return GenerateDataResponse(success=False, error=str(e))


@app.delete(
    "/data/clear",
    response_model=ClearDataResponse,
    tags=["Data Management"],
    summary="Clear all data",
    description="Delete all records from the database tables.",
)
async def clear_data() -> ClearDataResponse:
    """Clear all data from the database.

    Returns:
        ClearDataResponse with list of cleared tables.
    """
    try:
        from synthetic_pipeline.db.models import Base
        from synthetic_pipeline.db.session import DatabaseSession

        db_session = DatabaseSession()

        # Get table names before dropping
        table_names = [table.name for table in Base.metadata.sorted_tables]

        # Drop all tables
        Base.metadata.drop_all(db_session.engine)

        # Recreate empty tables
        Base.metadata.create_all(db_session.engine)

        return ClearDataResponse(
            success=True,
            tables_cleared=table_names,
        )

    except Exception as e:
        logger.exception("Data clearing failed")
        return ClearDataResponse(success=False, error=str(e))


@app.post(
    "/train",
    response_model=TrainResponse,
    tags=["Training"],
    summary="Train a new model",
    description="Train a new XGBoost model with the specified hyperparameters.",
)
async def train_model_endpoint(request: TrainRequest) -> TrainResponse:
    """Train a new model with specified parameters.

    Args:
        request: Training request with max_depth and training_window_days.

    Returns:
        TrainResponse with success status and run_id or error.
    """
    try:
        from model.train import train_model

        run_id = train_model(
            max_depth=request.max_depth,
            training_window_days=request.training_window_days,
        )
        return TrainResponse(success=True, run_id=run_id)
    except ValueError as e:
        return TrainResponse(success=False, error=str(e))
    except Exception as e:
        logger.exception("Training failed")
        return TrainResponse(success=False, error=str(e))


@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )
