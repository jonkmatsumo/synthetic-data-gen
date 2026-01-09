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

# Custom generation
uv run python src/main.py seed --count 5000 --fraud-rate 0.10

# With all options
uv run python src/main.py seed \
  --count 10000 \
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
| `--count` | `-c` | 1000 | Total number of profiles to generate |
| `--fraud-rate` | `-f` | 0.05 | Fraction of fraudulent profiles (0.0-1.0) |
| `--seed` | `-s` | None | Random seed for reproducibility |
| `--batch-size` | `-b` | 500 | Batch size for database inserts |
| `--database-url` | | env | Database URL (or set `DATABASE_URL`) |
| `--drop-tables` | | False | Drop existing tables before seeding |
| `--json-logs` | | False | Output logs in JSON format |
| `--verbose` | `-v` | False | Enable verbose logging |

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

Main denormalized table containing all profile data.

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key (auto-increment) |
| `record_id` | VARCHAR(50) | Unique record identifier |
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
| `fraud_type` | VARCHAR(50) | Type: liquidity_crunch, link_burst, ato |
| `created_at` | DATETIME | Record creation time |

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

| Pattern | Key Indicators |
|---------|----------------|
| **Liquidity Crunch** | Low balance, z-score < -2.5, is_returned=True |
| **Link Burst** | bank_connections_count_24h between 5-15 |
| **Account Takeover (ATO)** | amount_to_avg_ratio > 5.0, off-hours, identity change < 72h |
| **Device Sharing** | >5 users per device in 7 days |
| **IP Recycling** | Multiple users sharing same IP/VPN |
| **Kiting Cycle** | Circular fund transfers (A->B->C->A) |

## File Structure

```
synthetic-data-gen/
├── src/
│   ├── main.py                          # CLI entry point
│   └── synthetic_pipeline/
│       ├── __init__.py                  # Package init
│       ├── generator.py                 # DataGenerator class
│       ├── graph.py                     # GraphNetworkGenerator class
│       ├── logging.py                   # Structured logging config
│       ├── db/
│       │   ├── __init__.py              # DB module exports
│       │   ├── models.py                # SQLAlchemy models
│       │   └── session.py               # Database session management
│       └── models/
│           ├── __init__.py              # Model exports
│           ├── account.py               # AccountSnapshot model
│           ├── behavior.py              # BehaviorMetrics model
│           ├── connection.py            # ConnectionMetrics model
│           ├── record.py                # GeneratedRecord model
│           └── transaction.py           # TransactionEvaluation model
├── tests/
│   ├── __init__.py
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
| `src/synthetic_pipeline/generator.py` | Transaction data generation with fraud patterns |
| `src/synthetic_pipeline/graph.py` | Graph relationship generation |
| `src/synthetic_pipeline/db/models.py` | SQLAlchemy ORM models |
| `src/synthetic_pipeline/db/session.py` | Database connection pooling and batch inserts |
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
