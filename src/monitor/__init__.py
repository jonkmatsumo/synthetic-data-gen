"""Monitoring utilities for drift detection."""

__all__ = ["calculate_psi", "detect_drift"]


def calculate_psi(*args, **kwargs):
    """Calculate Population Stability Index. See detect_drift module."""
    from monitor.detect_drift import calculate_psi as _calculate_psi

    return _calculate_psi(*args, **kwargs)


def detect_drift(*args, **kwargs):
    """Run drift detection. See detect_drift module."""
    from monitor.detect_drift import detect_drift as _detect_drift

    return _detect_drift(*args, **kwargs)
