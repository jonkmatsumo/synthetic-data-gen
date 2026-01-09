# synthetic-data-gen

Synthetic data generation pipeline for fraud detection model training. Generates realistic transaction profiles with configurable fraud patterns.

## How to Run

### Prerequisites

- Python 3.10+
- Docker and Docker Compose (for PostgreSQL)
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and pre-commit hooks
make install
```

### Start Database

```bash
# Start PostgreSQL container
docker compose up -d db

# Verify database is running
docker compose ps
```

### Generate Data

```bash
# Generate 1000 profiles with 5% fraud rate (default)
uv run python src/main.py seed

# Custom generation with user sequences
uv run python src/main.py seed --users 500 --fraud-rate 0.10

# Legacy mode (single transaction per user)
uv run python src/main.py seed --users 1000 --legacy

# With all options
uv run python src/main.py seed \
  --users 1000 \
  --fraud-rate 0.05 \
  --seed 42 \
  --batch-size 500 \
  --drop-tables \
  --verbose
```

### CLI Commands

| Command | Description |
|---------|-------------|
| `seed` | Generate synthetic profiles and insert into database |
| `init-db` | Initialize database schema without seeding |
| `stats` | Show statistics about generated data |

### Seed Command Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--users` | `-u` | 100 | Number of unique users to generate |
| `--fraud-rate` | `-f` | 0.05 | Fraction of users with fraud events (0.0-1.0) |
| `--seed` | `-s` | None | Random seed for reproducibility |
| `--batch-size` | `-b` | 500 | Batch size for database inserts |
| `--database-url` | | env | Database URL (or set `DATABASE_URL`) |
| `--drop-tables` | | False | Drop existing tables before seeding |
| `--json-logs` | | False | Output logs in JSON format |
| `--verbose` | `-v` | False | Enable verbose logging |
| `--legacy` | | False | Use legacy mode (single transaction per user) |

### Development

```bash
# Run tests
make test

# Run linting
make lint

# Fix lint issues
make lint-fix

# Clean build artifacts
make clean
```

## Table and Feature Schemas

### Generated Records Table (`generated_records`)

Main denormalized table containing all transaction data.

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key (auto-increment) |
| `record_id` | VARCHAR(50) | Unique record identifier |
| `user_id` | VARCHAR(50) | User identifier (indexed) |
| `full_name` | VARCHAR(255) | Account holder name |
| `email` | VARCHAR(255) | Email address |
| `phone` | VARCHAR(50) | Phone number |
| `transaction_timestamp` | DATETIME | Transaction time |
| `is_off_hours_txn` | BOOLEAN | True if 10pm-6am |
| `available_balance` | NUMERIC(18,2) | Current balance |
| `balance_to_transaction_ratio` | FLOAT | Balance / transaction amount |
| `avg_available_balance_30d` | NUMERIC(18,2) | 30-day average balance |
| `balance_volatility_z_score` | FLOAT | Volatility z-score (risk if < -2.5) |
| `bank_connections_count_24h` | INTEGER | Connections in 24h (anomaly if > 4) |
| `bank_connections_count_7d` | INTEGER | Connections in 7 days |
| `bank_connections_avg_30d` | FLOAT | 30-day average connections |
| `amount` | NUMERIC(18,2) | Transaction amount |
| `amount_to_avg_ratio` | FLOAT | Amount / average (anomaly if > 5.0) |
| `merchant_risk_score` | INTEGER | 0-100 (anomaly if > 80) |
| `is_returned` | BOOLEAN | Transaction returned flag |
| `email_changed_at` | DATETIME | Last email change time |
| `phone_changed_at` | DATETIME | Last phone change time |
| `is_fraudulent` | BOOLEAN | Fraud label (target) |
| `fraud_type` | VARCHAR(50) | Type: liquidity_crunch, link_burst, ato, bust_out, sleeper_ato |
| `created_at` | DATETIME | Record creation time |

### Evaluation Metadata Table (`evaluation_metadata`)

Tracks temporal relationships between transactions and fraud events for proper train/test splitting. **This table is for evaluation only - should NOT be used as training features.**

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key (auto-increment) |
| `user_id` | VARCHAR(50) | User identifier (indexed) |
| `record_id` | VARCHAR(50) | Links to generated_records (indexed) |
| `sequence_number` | INTEGER | Transaction order for user (1-indexed) |
| `fraud_confirmed_at` | DATETIME | When fraud was confirmed (nullable) |
| `is_pre_fraud` | BOOLEAN | Transaction occurred before fraud detection |
| `days_to_fraud` | FLOAT | Days until fraud event (negative if after) |
| `is_train_eligible` | BOOLEAN | Can be used for training (indexed) |
| `created_at` | DATETIME | Record creation time |

**Indexes:**
- `ix_eval_user_sequence` on `(user_id, sequence_number)`
- `ix_eval_train_eligible` on `(is_train_eligible, is_pre_fraud)`

### Feature Snapshots Table (`feature_snapshots`)

Feature store for ML training with point-in-time correct features computed using SQL window functions. **No future data leakage.**

| Column | Type | Description |
|--------|------|-------------|
| `snapshot_id` | INTEGER | Primary key (auto-increment) |
| `record_id` | VARCHAR(50) | FK to generated_records (unique, indexed) |
| `user_id` | VARCHAR(50) | User identifier (indexed) |
| `velocity_24h` | INTEGER | Transaction count in 24h window |
| `amount_to_avg_ratio_30d` | FLOAT | Current amount / 30-day rolling average |
| `balance_volatility_z_score` | FLOAT | (balance - avg) / stddev over 30 days |
| `experimental_signals` | JSONB | Flexible JSON for experimental features |
| `computed_at` | DATETIME | When features were computed |

**Window Function Logic (Point-in-Time Correct):**
```sql
-- velocity_24h
COUNT(*) OVER (
    PARTITION BY user_id
    ORDER BY transaction_timestamp
    RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND CURRENT ROW
)

-- balance_volatility_z_score
(balance - AVG(balance) OVER window) / STDDEV(balance) OVER window
-- where window = RANGE BETWEEN INTERVAL '30 days' PRECEDING AND CURRENT ROW
```

**Experimental Signals (JSONB):**
- `velocity_7d`: Transaction count in 7-day window
- `max_amount_30d`: Maximum amount in 30-day window
- `off_hours_count_7d`: Off-hours transactions in 7 days
- `bank_connections_24h`: Bank connection count
- `merchant_risk_score`: Merchant risk score

### Pydantic Models

#### AccountSnapshot

| Field | Type | Description |
|-------|------|-------------|
| `available_balance` | Decimal(18,2) | Current available balance |
| `balance_to_transaction_ratio` | Float | Balance to transaction ratio |

#### BehaviorMetrics

| Field | Type | Description |
|-------|------|-------------|
| `avg_available_balance_30d` | Decimal(18,2) | 30-day average balance |
| `balance_volatility_z_score` | Float | High risk if Z < -2.5 |

#### ConnectionMetrics

| Field | Type | Description |
|-------|------|-------------|
| `bank_connections_count_24h` | Integer | Anomaly threshold > 4 |
| `bank_connections_count_7d` | Integer | Anomaly if > 300% of 30d avg |
| `bank_connections_avg_30d` | Float | 30-day daily average |

#### TransactionEvaluation

| Field | Type | Description |
|-------|------|-------------|
| `amount` | Decimal(18,2) | Transaction amount |
| `amount_to_avg_ratio` | Float | Anomaly if > 5.0 |
| `merchant_risk_score` | Integer | 0-100, anomaly if > 80 |
| `is_returned` | Boolean | Target label |

#### GeneratedRecord

Composite model combining all above with PII and labels.

| Field | Type | Description |
|-------|------|-------------|
| `record_id` | String | Unique identifier |
| `user_id` | String | User identifier |
| `full_name` | String | Account holder name |
| `email` | String | Email address |
| `phone` | String | Phone number |
| `transaction_timestamp` | DateTime | Transaction time |
| `is_off_hours_txn` | Boolean | Off-hours flag (10pm-6am) |
| `account` | AccountSnapshot | Account metrics |
| `behavior` | BehaviorMetrics | Behavioral metrics |
| `connection` | ConnectionMetrics | Connection metrics |
| `transaction` | TransactionEvaluation | Transaction metrics |
| `identity_changes` | IdentityChangeInfo | Identity change tracking |
| `is_fraudulent` | Boolean | Fraud label |
| `fraud_type` | String | Fraud scenario type |

#### EvaluationMetadata

Metadata for model evaluation (not used in training).

| Field | Type | Description |
|-------|------|-------------|
| `user_id` | String | User identifier |
| `record_id` | String | Links to GeneratedRecord |
| `sequence_number` | Integer | Transaction order (1-indexed) |
| `fraud_confirmed_at` | DateTime | When fraud was confirmed |
| `is_pre_fraud` | Boolean | Before fraud detection |
| `days_to_fraud` | Float | Days until fraud event |
| `is_train_eligible` | Boolean | Can be used for training |

### Graph Models

#### Node Types

| Type | Properties |
|------|------------|
| `USER` | name, email, phone |
| `DEVICE` | fingerprint, device_type, os, user_agent |
| `ACCOUNT` | account_number, account_type, balance |
| `IP_ADDRESS` | address, is_vpn, country |

#### Edge Types

| Type | Description |
|------|-------------|
| `USES_DEVICE` | User -> Device relationship |
| `HAS_ACCOUNT` | User -> Account relationship |
| `USES_IP` | User -> IP Address relationship |
| `TRANSFERS_TO` | Account -> Account fund transfer |

### Fraud Patterns

#### Transaction-Based Patterns

| Pattern | Key Indicators |
|---------|----------------|
| **Liquidity Crunch** | Low balance, z-score < -2.5, is_returned=True |
| **Link Burst** | bank_connections_count_24h between 5-15 |
| **Account Takeover (ATO)** | amount_to_avg_ratio > 5.0, off-hours, identity change < 72h |

#### Stateful Fraud Profiles (src/generator/core.py)

| Profile | Behavior | Metadata |
|---------|----------|----------|
| **Bust-Out** | 20-50 small legitimate transactions, then sudden spike (>500% of avg) | Fraud event has `is_train_eligible=True` |
| **Sleeper (ATO)** | Dormant 30+ days, then Link Burst (3+ connections in 1 hour), then high-value debit | Burst events marked `is_pre_fraud=True` |

**Label Delay Simulation**: `fraud_confirmed_at` uses Log-Normal distribution (mean=5 days). If `fraud_confirmed_at > simulation_date`, record appears "clean" (undetected fraud).

#### Graph-Based Patterns

| Pattern | Key Indicators |
|---------|----------------|
| **Device Sharing** | >5 users per device in 7 days |
| **IP Recycling** | Multiple users sharing same IP/VPN |
| **Kiting Cycle** | Circular fund transfers (A->B->C->A) |

## File Structure

```
synthetic-data-gen/
├── src/
│   ├── main.py                          # CLI entry point
│   ├── generator/
│   │   ├── __init__.py                  # Stateful generator exports
│   │   └── core.py                      # UserSimulator, fraud profiles
│   ├── pipeline/
│   │   ├── __init__.py                  # Pipeline exports
│   │   └── materialize_features.py      # Feature materialization script
│   └── synthetic_pipeline/
│       ├── __init__.py                  # Package init
│       ├── generator.py                 # DataGenerator class
│       ├── graph.py                     # GraphNetworkGenerator class
│       ├── logging.py                   # Structured logging config
│       ├── db/
│       │   ├── __init__.py              # DB module exports
│       │   ├── models.py                # SQLAlchemy models
│       │   ├── session.py               # Database session management
│       │   └── migrations/
│       │       └── 001_create_feature_snapshots.sql
│       └── models/
│           ├── __init__.py              # Model exports
│           ├── account.py               # AccountSnapshot model
│           ├── behavior.py              # BehaviorMetrics model
│           ├── connection.py            # ConnectionMetrics model
│           ├── record.py                # GeneratedRecord model
│           └── transaction.py           # TransactionEvaluation model
├── tests/
│   ├── __init__.py
│   ├── test_core_generator.py           # Stateful generator tests
│   ├── test_feature_store.py            # Feature store tests
│   ├── test_generator.py                # DataGenerator tests
│   ├── test_graph.py                    # GraphNetworkGenerator tests
│   └── test_pipeline.py                 # Basic tests
├── config/                              # Configuration files
├── scripts/
│   └── wait-for-it.sh                   # DB readiness script
├── .github/
│   └── workflows/
│       └── ci.yml                       # GitHub Actions CI
├── .env                                 # Environment variables (gitignored)
├── .env.example                         # Environment template
├── .gitignore
├── .pre-commit-config.yaml              # Pre-commit hooks
├── docker-compose.yml                   # Docker services
├── Dockerfile                           # Generator container
├── Makefile                             # Build commands
├── pyproject.toml                       # Project config and dependencies
└── README.md                            # This file
```

### Key Files

| File | Purpose |
|------|---------|
| `src/main.py` | CLI with `seed`, `init-db`, `stats` commands |
| `src/generator/core.py` | Stateful UserSimulator, BustOutProfile, SleeperProfile |
| `src/pipeline/materialize_features.py` | Feature materialization with SQL window functions |
| `src/synthetic_pipeline/generator.py` | Transaction data generation with fraud patterns |
| `src/synthetic_pipeline/graph.py` | Graph relationship generation |
| `src/synthetic_pipeline/db/models.py` | SQLAlchemy ORM models (GeneratedRecordDB, FeatureSnapshotDB) |
| `src/synthetic_pipeline/db/session.py` | Database connection pooling and batch inserts |
| `src/synthetic_pipeline/db/migrations/*.sql` | SQL migration scripts |
| `src/synthetic_pipeline/models/*.py` | Pydantic validation models |
| `docker-compose.yml` | PostgreSQL and generator services |
| `pyproject.toml` | Dependencies and project metadata |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_USER` | synthetic | Database user |
| `POSTGRES_PASSWORD` | synthetic_dev_password | Database password |
| `POSTGRES_DB` | synthetic_data | Database name |
| `POSTGRES_HOST` | localhost | Database host |
| `POSTGRES_PORT` | 5432 | Database port |
| `DATABASE_URL` | (built from above) | Full connection URL |
