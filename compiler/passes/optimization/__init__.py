from .autodiff import autodiff
from .autobatch import vectorize_graph
from .parameters import collect_trainable_parameters
from .memory import optimize_memory
from compiler.optimizer import optimize_graph as _optimize_graph_base
from typing import Optional, Any

def run_transformations(
    graph,
    optimize: bool = True,
    autobatch: bool = False,
    autodiff_lowering: bool = True,
    context: str = "both",
    report: Optional[Any] = None
):
    """Composed transformation pass."""
    g = graph
    
    # 1. Autodiff Lowering
    if autodiff_lowering and context in ["learner", "both"]:
        g = autodiff(g, report=report)
        
    # 2. AutoBatching
    if autobatch:
        g = vectorize_graph(g)
        
    # 3. Optimization
    if optimize:
        # Use the base optimizer which internally calls memory optimizations
        g = _optimize_graph_base(g, report=report)
        
    # 4. Parameters
    g.parameters = collect_trainable_parameters(g)
    
    return g
