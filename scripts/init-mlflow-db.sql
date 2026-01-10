-- Create MLflow database for experiment tracking
-- This script is executed on PostgreSQL container startup

CREATE DATABASE mlflow_db;

-- Grant privileges to the default user
GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO synthetic;
