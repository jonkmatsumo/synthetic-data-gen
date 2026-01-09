"""Tests for the feature store and materialization pipeline."""

from synthetic_pipeline.db.models import FeatureSnapshotDB


class TestFeatureSnapshotModel:
    """Tests for FeatureSnapshotDB model."""

    def test_model_tablename(self):
        """Test that model has correct table name."""
        assert FeatureSnapshotDB.__tablename__ == "feature_snapshots"

    def test_model_has_required_columns(self):
        """Test that model has all required columns."""
        columns = FeatureSnapshotDB.__table__.columns
        column_names = [c.name for c in columns]

        required_columns = [
            "snapshot_id",
            "record_id",
            "user_id",
            "velocity_24h",
            "amount_to_avg_ratio_30d",
            "balance_volatility_z_score",
            "experimental_signals",
            "computed_at",
        ]

        for col in required_columns:
            assert col in column_names, f"Missing column: {col}"

    def test_record_id_is_unique(self):
        """Test that record_id has unique constraint."""
        record_id_col = FeatureSnapshotDB.__table__.columns["record_id"]
        assert record_id_col.unique is True

    def test_record_id_foreign_key(self):
        """Test that record_id has foreign key to generated_records."""
        record_id_col = FeatureSnapshotDB.__table__.columns["record_id"]
        fk_targets = [fk.target_fullname for fk in record_id_col.foreign_keys]
        assert "generated_records.record_id" in fk_targets

    def test_experimental_signals_is_jsonb(self):
        """Test that experimental_signals is JSONB type."""
        exp_col = FeatureSnapshotDB.__table__.columns["experimental_signals"]
        # Check it's a JSONB type (PostgreSQL-specific)
        assert "JSONB" in str(exp_col.type)


class TestFeatureEngineeringSQL:
    """Tests for feature engineering SQL queries."""

    def test_sql_has_velocity_window(self):
        """Test that SQL includes velocity window function."""
        from pipeline.materialize_features import FEATURE_ENGINEERING_SQL

        # Check for 24h window function
        assert "RANGE BETWEEN INTERVAL '24 hours' PRECEDING" in FEATURE_ENGINEERING_SQL
        assert "COUNT(*) OVER" in FEATURE_ENGINEERING_SQL

    def test_sql_has_amount_ratio_window(self):
        """Test that SQL includes amount ratio window function."""
        from pipeline.materialize_features import FEATURE_ENGINEERING_SQL

        # Check for 30-day window for amount ratio
        assert "RANGE BETWEEN INTERVAL '30 days' PRECEDING" in FEATURE_ENGINEERING_SQL
        assert "AVG(gr.amount) OVER" in FEATURE_ENGINEERING_SQL

    def test_sql_has_volatility_window(self):
        """Test that SQL includes balance volatility window function."""
        from pipeline.materialize_features import FEATURE_ENGINEERING_SQL

        # Check for STDDEV calculation
        assert "STDDEV(gr.available_balance) OVER" in FEATURE_ENGINEERING_SQL

    def test_sql_has_point_in_time_correctness(self):
        """Test that SQL uses CURRENT ROW to prevent future leakage."""
        from pipeline.materialize_features import FEATURE_ENGINEERING_SQL

        # All windows should end at CURRENT ROW (no future leakage)
        assert "AND CURRENT ROW" in FEATURE_ENGINEERING_SQL

    def test_sql_partitions_by_user(self):
        """Test that SQL partitions window functions by user_id."""
        from pipeline.materialize_features import FEATURE_ENGINEERING_SQL

        assert "PARTITION BY gr.user_id" in FEATURE_ENGINEERING_SQL

    def test_sql_orders_by_timestamp(self):
        """Test that SQL orders window functions by transaction_timestamp."""
        from pipeline.materialize_features import FEATURE_ENGINEERING_SQL

        assert "ORDER BY gr.transaction_timestamp" in FEATURE_ENGINEERING_SQL

    def test_sql_excludes_existing_records(self):
        """Test that SQL excludes already-processed records."""
        from pipeline.materialize_features import FEATURE_ENGINEERING_SQL

        expected = "NOT IN (SELECT record_id FROM feature_snapshots)"
        assert expected in FEATURE_ENGINEERING_SQL

    def test_sql_builds_experimental_signals_json(self):
        """Test that SQL builds JSONB for experimental signals."""
        from pipeline.materialize_features import FEATURE_ENGINEERING_SQL

        assert "jsonb_build_object" in FEATURE_ENGINEERING_SQL


class TestFeatureMaterializer:
    """Tests for FeatureMaterializer class."""

    def test_materializer_imports(self):
        """Test that FeatureMaterializer can be imported."""
        from pipeline.materialize_features import FeatureMaterializer

        assert FeatureMaterializer is not None

    def test_materialize_function_imports(self):
        """Test that materialize_features function can be imported."""
        from pipeline.materialize_features import materialize_features

        assert materialize_features is not None

    def test_materializer_has_required_methods(self):
        """Test that FeatureMaterializer has required methods."""
        from pipeline.materialize_features import FeatureMaterializer

        required_methods = [
            "create_table",
            "get_pending_record_count",
            "compute_features_batch",
            "materialize_all",
            "get_feature_stats",
        ]

        for method in required_methods:
            assert hasattr(FeatureMaterializer, method), f"Missing method: {method}"


class TestMigrationSQL:
    """Tests for SQL migration script."""

    def test_migration_file_exists(self):
        """Test that migration file exists."""
        import os

        migration_path = (
            "src/synthetic_pipeline/db/migrations/001_create_feature_snapshots.sql"
        )
        assert os.path.exists(migration_path)

    def test_migration_creates_table(self):
        """Test that migration creates the table."""
        with open(
            "src/synthetic_pipeline/db/migrations/001_create_feature_snapshots.sql"
        ) as f:
            sql = f.read()

        assert "CREATE TABLE IF NOT EXISTS feature_snapshots" in sql

    def test_migration_has_foreign_key(self):
        """Test that migration includes foreign key constraint."""
        with open(
            "src/synthetic_pipeline/db/migrations/001_create_feature_snapshots.sql"
        ) as f:
            sql = f.read()

        assert "FOREIGN KEY (record_id)" in sql
        assert "REFERENCES generated_records(record_id)" in sql

    def test_migration_has_indexes(self):
        """Test that migration creates required indexes."""
        with open(
            "src/synthetic_pipeline/db/migrations/001_create_feature_snapshots.sql"
        ) as f:
            sql = f.read()

        assert "CREATE INDEX" in sql
        assert "ix_feature_snapshots_record_id" in sql
        assert "ix_feature_snapshots_user_id" in sql

    def test_migration_has_gin_index_for_jsonb(self):
        """Test that migration creates GIN index for JSONB column."""
        with open(
            "src/synthetic_pipeline/db/migrations/001_create_feature_snapshots.sql"
        ) as f:
            sql = f.read()

        assert "USING GIN (experimental_signals)" in sql
