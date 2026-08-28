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
