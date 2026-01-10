# Label Lag

End-to-end ML system for fraud detection with realistic label delay simulation. Generates synthetic transaction data, trains models with MLflow tracking, and serves predictions via API.

## Quick Start

```bash
# Start all services
docker compose up -d

# Generate synthetic data (inside generator container)
docker compose exec generator uv run python src/main.py seed --users 1000 --fraud-rate 0.05
```

## Services

| Service | URL | Description |
|---------|-----|-------------|
| Dashboard | http://localhost:8501 | Streamlit UI for risk analysis and model training |
| API | http://localhost:8000 | FastAPI fraud scoring endpoint |
| API Docs | http://localhost:8000/docs | Swagger UI |
| MLflow | http://localhost:5005 | Experiment tracking and model registry |
| MinIO | http://localhost:9001 | Object storage console (minioadmin/minioadmin) |
| PostgreSQL | localhost:5432 | Database |

All ports are configurable via `.env` file.

## Development

```bash
# Install dependencies locally
make install

# Run tests
make test

# Linting
make lint
make lint-fix
```

## Environment Variables

Copy `.env.example` to `.env` and adjust as needed:

```bash
# Database
POSTGRES_USER=synthetic
POSTGRES_PASSWORD=synthetic_dev_password
POSTGRES_DB=synthetic_data
DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:5432/${POSTGRES_DB}

# Service ports (change to avoid conflicts)
DB_PORT=5432
API_PORT=8000
DASHBOARD_PORT=8501
MLFLOW_PORT=5005
```

## Architecture

- **Synthetic Pipeline**: Generates realistic transaction profiles with configurable fraud patterns (bust-out, sleeper ATO, link burst)
- **Label Delay Simulation**: Log-normal distribution for `fraud_confirmed_at` timestamps, enabling point-in-time correct training
- **Feature Store**: SQL window functions compute features without future data leakage
- **Model Training**: XGBoost with MLflow tracking, metrics logging, and model registry
- **Serving**: Dynamic model loading from MLflow with automatic reloading on promotion
