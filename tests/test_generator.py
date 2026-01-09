"""Tests for the DataGenerator class."""

from decimal import Decimal

import pytest

from synthetic_pipeline.generator import DataGenerator, FraudType


@pytest.fixture
def generator() -> DataGenerator:
    """Create a seeded generator for reproducible tests."""
    return DataGenerator(seed=42)


class TestLegitimateGeneration:
    """Tests for legitimate transaction generation."""

    def test_generates_single_record(self, generator: DataGenerator):
        """Test generating a single legitimate record."""
        records = generator.generate_legitimate(count=1)
        assert len(records) == 1

    def test_generates_multiple_records(self, generator: DataGenerator):
        """Test generating multiple legitimate records."""
        records = generator.generate_legitimate(count=10)
        assert len(records) == 10

    def test_legitimate_volatility_in_range(self, generator: DataGenerator):
        """Test that volatility z-score stays between -1.0 and 1.0."""
        records = generator.generate_legitimate(count=100)
        for record in records:
            z_score = record.behavior.balance_volatility_z_score
            assert -1.0 <= z_score <= 1.0

    def test_legitimate_merchant_risk_low(self, generator: DataGenerator):
        """Test that merchant risk scores average around 20-30."""
        records = generator.generate_legitimate(count=100)
        avg_risk = sum(r.transaction.merchant_risk_score for r in records) / len(
            records
        )
        # Should be in 5-45 range, averaging around 25
        assert 15 <= avg_risk <= 35

    def test_legitimate_not_fraudulent(self, generator: DataGenerator):
        """Test that legitimate records are not marked as fraudulent."""
        records = generator.generate_legitimate(count=10)
        for record in records:
            assert record.is_fraudulent is False
            assert record.fraud_type is None


class TestFraudulentGeneration:
    """Tests for fraudulent transaction generation."""

    def test_liquidity_crunch_low_balance(self, generator: DataGenerator):
        """Test liquidity crunch has low available balance."""
        records = generator.generate_fraudulent(FraudType.LIQUIDITY_CRUNCH, count=10)
        for record in records:
            assert record.account.available_balance < Decimal("100")

    def test_liquidity_crunch_high_negative_zscore(self, generator: DataGenerator):
        """Test liquidity crunch has z-score < -2.5."""
        records = generator.generate_fraudulent(FraudType.LIQUIDITY_CRUNCH, count=10)
        for record in records:
            assert record.behavior.balance_volatility_z_score < -2.5
            assert record.behavior.is_high_risk_volatility is True

    def test_link_burst_high_connections(self, generator: DataGenerator):
        """Test link burst has 5-15 connections in 24h."""
        records = generator.generate_fraudulent(FraudType.LINK_BURST, count=10)
        for record in records:
            count_24h = record.connection.bank_connections_count_24h
            assert 5 <= count_24h <= 15
            assert record.connection.is_24h_anomaly is True

    def test_ato_high_amount_ratio(self, generator: DataGenerator):
        """Test ATO has amount_to_avg_ratio > 5.0."""
        records = generator.generate_fraudulent(FraudType.ATO, count=10)
        for record in records:
            assert record.transaction.amount_to_avg_ratio > 5.0
            assert record.transaction.is_amount_anomaly is True

    def test_ato_off_hours(self, generator: DataGenerator):
        """Test ATO transactions occur during off-hours."""
        records = generator.generate_fraudulent(FraudType.ATO, count=10)
        for record in records:
            assert record.is_off_hours_txn is True
            hour = record.transaction_timestamp.hour
            assert hour >= 22 or hour < 6

    def test_ato_recent_identity_change(self, generator: DataGenerator):
        """Test ATO has identity change within 72 hours."""
        records = generator.generate_fraudulent(FraudType.ATO, count=10)
        for record in records:
            has_change = record.identity_changes.has_recent_change(
                record.transaction_timestamp, threshold_hours=72
            )
            assert has_change is True

    def test_fraudulent_records_marked(self, generator: DataGenerator):
        """Test that all fraudulent records are properly marked."""
        for fraud_type in FraudType:
            records = generator.generate_fraudulent(fraud_type, count=5)
            for record in records:
                assert record.is_fraudulent is True
                assert record.fraud_type == fraud_type.value

    def test_string_fraud_type(self, generator: DataGenerator):
        """Test that string fraud types work."""
        records = generator.generate_fraudulent("ato", count=1)
        assert len(records) == 1
        assert records[0].fraud_type == "ato"
