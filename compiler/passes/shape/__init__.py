from .inference import infer_shapes
from .validation import validate_shapes
from .gradient_analysis import analyze_gradients
from compiler.validation import ValidationReport, SEVERITY_WARN, SEVERITY_ERROR, ValidationIssue

def run_shape_analysis(graph) -> any: # returns Graph
    """Composed shape analysis pass."""
    return infer_shapes(graph)

def validate_shape_semantics(graph, context: str = "both") -> ValidationReport:
    """Validates shapes and analyzes gradients."""
    report = ValidationReport()
    report.merge(validate_shapes(graph))
    
    if context in ["learner", "both"]:
        grad_report = analyze_gradients(graph)
        for warn in grad_report.warnings:
            report.add(ValidationIssue(severity=SEVERITY_WARN, code="G001", node_id=None, message=warn))
        for err in grad_report.errors:
            report.add(ValidationIssue(severity=SEVERITY_ERROR, code="G001", node_id=None, message=err))
            
    return report
