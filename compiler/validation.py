"""
Validation system for the compiler.
Defines structures for reporting errors, warnings, and info messages.
"""

from dataclasses import dataclass
from typing import List, Optional

# Severity levels for validation issues
SEVERITY_ERROR = "ERROR"
SEVERITY_WARN = "WARN"
SEVERITY_INFO = "INFO"


@dataclass(frozen=True)
class ValidationIssue:
    """Represents a single validation issue found during compilation.

    Attributes:
        severity: The severity of the issue (ERROR, WARN, INFO).
        code: A unique identifier for the type of issue (e.g., 'UNUSED_NODE').
        node_id: Optional identifier of the graph node associated with the issue.
        message: A human-readable description of the issue.
    """
    severity: str
    code: str
    node_id: Optional[str]
    message: str

    def __post_init__(self) -> None:
        """Validate severity level."""
        valid_severities = {SEVERITY_ERROR, SEVERITY_WARN, SEVERITY_INFO}
        assert self.severity in valid_severities, (
            f"Invalid severity '{self.severity}'. Must be one of {valid_severities}"
        )


class ValidationReport:
    """Collects and manages validation issues found during compilation."""

    def __init__(self) -> None:
        """Initializes an empty validation report."""
        self.issues: List[ValidationIssue] = []

    def add(self, issue: ValidationIssue) -> None:
        """Adds a validation issue to the report.

        Args:
            issue: The validation issue to add.
        """
        self.issues.append(issue)

    def has_errors(self) -> bool:
        """Checks if the report contains any issues with ERROR severity.

        Returns:
            True if there is at least one error, False otherwise.
        """
        return any(issue.severity == SEVERITY_ERROR for issue in self.issues)

    def has_warnings(self) -> bool:
        """Checks if the report contains any issues with WARN severity.

        Returns:
            True if there is at least one warning, False otherwise.
        """
        return any(issue.severity == SEVERITY_WARN for issue in self.issues)

    def merge(self, other: "ValidationReport") -> None:
        """Merges another validation report into this one.

        Args:
            other: The report to merge.
        """
        self.issues.extend(other.issues)

    def get_issues_by_severity(self, severity: str) -> List[ValidationIssue]:
        """Filters issues by severity.

        Args:
            severity: The severity to filter by.

        Returns:
            A list of issues matching the specified severity.
        """
        return [issue for issue in self.issues if issue.severity == severity]

    def __repr__(self) -> str:
        """Returns a string representation of the report summary."""
        errors = len(self.get_issues_by_severity(SEVERITY_ERROR))
        warnings = len(self.get_issues_by_severity(SEVERITY_WARN))
        infos = len(self.get_issues_by_severity(SEVERITY_INFO))
        return (
            f"ValidationReport(errors={errors}, warnings={warnings}, infos={infos})"
        )
