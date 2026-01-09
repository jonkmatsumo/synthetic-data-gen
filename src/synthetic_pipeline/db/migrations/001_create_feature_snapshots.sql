-- Migration: Create feature_snapshots table
-- Description: Feature store for point-in-time correct ML features
-- Created: 2026-01-09

-- Create feature_snapshots table
CREATE TABLE IF NOT EXISTS feature_snapshots (
    snapshot_id SERIAL PRIMARY KEY,

    -- Foreign key to generated_records
    record_id VARCHAR(50) NOT NULL UNIQUE,

    -- User identifier for partitioning window functions
    user_id VARCHAR(50) NOT NULL,

    -- Velocity feature: transaction count in 24h window
    velocity_24h INTEGER NOT NULL,

    -- Amount ratio: current amount / 30-day rolling average
    amount_to_avg_ratio_30d FLOAT NOT NULL,

    -- Balance volatility z-score: (balance - avg) / stddev over 30d
    balance_volatility_z_score FLOAT NOT NULL,

    -- Flexible JSONB for experimental features
    experimental_signals JSONB DEFAULT NULL,

    -- Metadata
    computed_at TIMESTAMP NOT NULL DEFAULT NOW(),

    -- Foreign key constraint
    CONSTRAINT fk_feature_record
        FOREIGN KEY (record_id)
        REFERENCES generated_records(record_id)
        ON DELETE CASCADE
);

-- Create indexes for efficient queries
CREATE INDEX IF NOT EXISTS ix_feature_snapshots_record_id
    ON feature_snapshots(record_id);

CREATE INDEX IF NOT EXISTS ix_feature_snapshots_user_id
    ON feature_snapshots(user_id);

CREATE INDEX IF NOT EXISTS ix_feature_snapshots_computed_at
    ON feature_snapshots(computed_at);

-- GIN index for JSONB queries on experimental_signals
CREATE INDEX IF NOT EXISTS ix_feature_snapshots_experimental
    ON feature_snapshots USING GIN (experimental_signals);

-- Add comments for documentation
COMMENT ON TABLE feature_snapshots IS
    'Feature store snapshot for ML training with point-in-time correct features';

COMMENT ON COLUMN feature_snapshots.velocity_24h IS
    'COUNT(*) of transactions in 24h window for user (no future leakage)';

COMMENT ON COLUMN feature_snapshots.amount_to_avg_ratio_30d IS
    'Current amount / AVG(amount) over preceding 30d window';

COMMENT ON COLUMN feature_snapshots.balance_volatility_z_score IS
    '(balance - AVG(balance)) / STDDEV(balance) over preceding 30d window';

COMMENT ON COLUMN feature_snapshots.experimental_signals IS
    'Flexible JSON for experimental features like device_trust_score';
