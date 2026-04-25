from .rl_semantics import validate_rl_semantics
from .context import validate_context
from .domains import validate_domains
from .purity import validate_purity
from .gradients import validate_grad_semantics
from .serialization import validate_ir_purity
from compiler.validation import ValidationReport

def validate_semantic(graph, context: str = "both") -> ValidationReport:
    """Composed semantic validation pass."""
    report = ValidationReport()
    report.merge(validate_rl_semantics(graph))
    report.merge(validate_context(graph))
    report.merge(validate_domains(graph))
    report.merge(validate_purity(graph, context))
    report.merge(validate_grad_semantics(graph, context))
    report.merge(validate_ir_purity(graph))
    return report
