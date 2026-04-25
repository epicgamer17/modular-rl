from .metadata import validate_metadata
from .connectivity import validate_structure
from .ports import validate_ports
from .handles import validate_handles
from compiler.validation import ValidationReport
from typing import Set, Optional

def validate_structural(
    graph,
    model_handles: Optional[Set[str]] = None,
    buffer_handles: Optional[Set[str]] = None,
    strict: bool = False
) -> ValidationReport:
    """Composed structural validation pass."""
    report = ValidationReport()
    report.merge(validate_metadata(graph, strict=strict))
    report.merge(validate_structure(graph))
    report.merge(validate_ports(graph))
    report.merge(validate_handles(graph, model_handles, buffer_handles))
    return report
