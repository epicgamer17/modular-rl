import pytest
from compiler.validation import (
    ValidationIssue,
    ValidationReport,
    SEVERITY_ERROR,
    SEVERITY_WARN,
    SEVERITY_INFO,
)

pytestmark = pytest.mark.unit


def test_validation_report_empty() -> None:
    """Verifies that a new report has no errors and no issues."""
    report = ValidationReport()
    assert not report.has_errors(), "Empty report should not have errors"
    assert len(report.issues) == 0, "Empty report should have zero issues"


def test_validation_report_add_error() -> None:
    """Verifies that adding an ERROR issue makes has_errors() return True."""
    report = ValidationReport()
    issue = ValidationIssue(
        severity=SEVERITY_ERROR,
        code="ERR_TEST",
        node_id="node_123",
        message="Critical failure",
    )
    report.add(issue)
    assert report.has_errors(), "Report with an ERROR should return True for has_errors()"
    assert len(report.issues) == 1
    assert report.issues[0] == issue


def test_validation_report_add_non_error() -> None:
    """Verifies that adding non-error issues does not trigger has_errors()."""
    report = ValidationReport()

    warn_issue = ValidationIssue(
        severity=SEVERITY_WARN,
        code="WARN_TEST",
        node_id=None,
        message="Something might be wrong",
    )
    info_issue = ValidationIssue(
        severity=SEVERITY_INFO,
        code="INFO_TEST",
        node_id="node_456",
        message="For your information",
    )

    report.add(warn_issue)
    assert not report.has_errors(), "Report with only WARNING should not have errors"

    report.add(info_issue)
    assert not report.has_errors(), "Report with WARNING and INFO should not have errors"

    assert len(report.issues) == 2


def test_validation_issue_invalid_severity() -> None:
    """Verifies that creating an issue with an invalid severity raises an AssertionError."""
    with pytest.raises(AssertionError, match="Invalid severity"):
        ValidationIssue(
            severity="BOGUS",
            code="TEST",
            node_id=None,
            message="Invalid severity test",
        )


def test_validation_report_repr() -> None:
    """Verifies the string representation of the ValidationReport."""
    report = ValidationReport()
    report.add(
        ValidationIssue(SEVERITY_ERROR, "E1", None, "Error")
    )
    report.add(
        ValidationIssue(SEVERITY_WARN, "W1", None, "Warning")
    )
    
    repr_str = repr(report)
    assert "errors=1" in repr_str
    assert "warnings=1" in repr_str
    assert "infos=0" in repr_str
