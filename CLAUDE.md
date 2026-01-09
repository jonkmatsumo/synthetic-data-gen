# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Synthetic data generation pipeline for fraud detection model training. Generates realistic transaction profiles with configurable fraud patterns, using PostgreSQL for persistence and SQL window functions for point-in-time correct feature engineering.

## Common Commands

```bash
# Install dependencies (uses uv package manager)
make install

# Run tests with coverage
make test

# Linting
make lint         # Check only
make lint-fix     # Auto-fix issues

# Generate data (requires PostgreSQL running)
docker compose up -d db
uv run python src/main.py seed --users 100 --fraud-rate 0.05

# Other CLI commands
uv run python src/main.py init-db   # Initialize schema only
uv run python src/main.py stats     # Show database statistics
```

## Architecture

### Data Generation Layers

**DataGenerator** (`src/synthetic_pipeline/generator.py`): Core generator with two modes:
- Legacy mode: Single transaction per user
- Sequences mode: Multiple transactions with temporal tracking and evaluation metadata

**Stateful Generator** (`src/generator/core.py`): `UserSimulator` maintains persistent state across transactions, enabling complex fraud profiles:
- `BustOutProfile`: 20-50 legitimate transactions → sudden spike (>500% avg)
- `SleeperProfile`: 30+ days dormancy → link burst → high-value withdrawal
- `LabelDelaySimulator`: Log-normal delay distribution for realistic fraud detection timing

### Database Layer

**DatabaseSession** (`src/synthetic_pipeline/db/session.py`): Connection pooling with batch insert optimization.

**SQLAlchemy Models** (`src/synthetic_pipeline/db/models.py`):
- `GeneratedRecordDB`: Main transaction table (24 columns, indexed on user_id, is_fraudulent, fraud_type)
- `EvaluationMetadataDB`: Temporal metadata for train/test splitting (NOT training features)
- `FeatureSnapshotDB`: Point-in-time correct features with JSONB for experimental signals

### Feature Store

**FeatureMaterializer** (`src/pipeline/materialize_features.py`): SQL window functions compute features without future data leakage:
- `velocity_24h`: Transaction count using `RANGE BETWEEN INTERVAL '24 hours' PRECEDING`
- `amount_to_avg_ratio_30d`, `balance_volatility_z_score`: Rolling aggregations
- `experimental_signals` (JSONB): Flexible schema for new features

### Pydantic Models (`src/synthetic_pipeline/models/`)

Hierarchical composition: `AccountSnapshot` → `BehaviorMetrics` → `ConnectionMetrics` → `TransactionEvaluation` → `GeneratedRecord`

### Graph Network

**GraphNetworkGenerator** (`src/synthetic_pipeline/graph.py`): Relationship data between users, devices, accounts, and IPs. Fraud patterns include device sharing rings, IP recycling, and fund transfer cycles.

## Fraud Pattern Types

| Pattern | Location | Key Indicators |
|---------|----------|----------------|
| Liquidity Crunch | DataGenerator | balance z-score < -2.5, returned=True |
| Link Burst | DataGenerator | 5-15 bank connections in 24h |
| ATO | DataGenerator | amount_ratio > 5.0, off-hours, identity change < 72h |
| Bust-Out | UserSimulator | Gradual buildup → sudden spike |
| Sleeper ATO | UserSimulator | Long dormancy → burst → withdrawal |

## Key Conventions

- **Point-in-time correctness**: All feature engineering uses window functions with strict temporal boundaries to prevent data leakage
- **Evaluation metadata**: `evaluation_metadata` table is for model evaluation only, never used as training features
- **Batch inserts**: Large datasets use `DatabaseSession.batch_insert()` for performance
- **Ruff formatting**: Line length 88, rules E/F/I/N/W/UP
