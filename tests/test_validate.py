from __future__ import annotations

import pytest

from alphabench.data.validate import DataQualityError, validate_panel


def test_valid_data_passes_with_no_issues(clean_panel):
    report = validate_panel(clean_panel, strict=True)
    assert report["issues"] == []
    assert report["rows"] == len(clean_panel)
    assert report["symbols"] == clean_panel["symbol"].nunique()


def test_duplicate_row_is_caught(dirty_panel):
    with pytest.raises(DataQualityError) as exc_info:
        validate_panel(dirty_panel, strict=True)
    assert "duplicate" in str(exc_info.value)


def test_high_low_violation_is_caught(dirty_panel):
    with pytest.raises(DataQualityError) as exc_info:
        validate_panel(dirty_panel, strict=True)
    assert "high < low" in str(exc_info.value)


def test_duplicate_and_high_low_reported_when_not_strict(dirty_panel):
    report = validate_panel(dirty_panel, strict=False)
    assert any("duplicate" in issue for issue in report["issues"])
    assert any("high < low" in issue for issue in report["issues"])


def test_large_single_day_move_warns_not_fails(jumpy_panel):
    report = validate_panel(jumpy_panel, strict=True)
    assert report["issues"] == []
    assert any("single-day moves" in w for w in report["warnings"])


def test_tiny_float_rounding_at_the_bound_is_not_flagged(clean_panel):
    """A close/high mismatch at float64 precision (~1e-16 relative) — the kind produced
    by rescaling OHLC by a corporate-action ratio — must not trip the bounds check."""
    df = clean_panel.copy()
    idx = df.index[0]
    df.loc[idx, "close"] = df.loc[idx, "high"] * (1 + 1e-15)
    report = validate_panel(df, strict=True)
    assert report["issues"] == []


def test_genuine_bounds_violation_is_still_caught(clean_panel):
    """A close meaningfully (1%) above high is real bad data, not rounding noise, and
    must still fail even with the small tolerance for float noise."""
    df = clean_panel.copy()
    idx = df.index[0]
    df.loc[idx, "close"] = df.loc[idx, "high"] * 1.01
    with pytest.raises(DataQualityError) as exc_info:
        validate_panel(df, strict=True)
    assert "outside [low, high]" in str(exc_info.value)
