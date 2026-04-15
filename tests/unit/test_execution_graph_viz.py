import pytest
import re
from core.execution_graph import build_execution_graph
from core.contracts import Key, Observation, ValueEstimate, Scalar, WriteMode
from core.component import PipelineComponent
from typing import Set, Dict, Any

# Tier 1 Unit Test Marker
pytestmark = pytest.mark.unit

class SimpleMockComponent(PipelineComponent):
    """Minimal mock component for graph testing."""
    def __init__(self, name: str, requires: Set[Key], provides: Dict[Key, WriteMode]):
        self._name = name
        self._requires = requires
        self._provides = provides

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Dict[Key, WriteMode]:
        return self._provides

    def execute(self, blackboard: Any) -> Dict[str, Any]:
        return {}

def test_execution_graph_to_dot_basic():
    """Verify that to_dot returns a valid-looking DOT string for a simple pipeline."""
    k_obs = Key("data.obs", Observation)
    k_val = Key("losses.value", ValueEstimate[Scalar])
    
    c1 = SimpleMockComponent("Producer", requires=set(), provides={k_obs: WriteMode.NEW})
    c2 = SimpleMockComponent("Consumer", requires={k_obs}, provides={k_val: WriteMode.NEW})
    
    graph = build_execution_graph(
        components=[c1, c2],
        initial_keys=set(),
        target_keys={k_val}
    )
    
    dot = graph.to_dot()
    
    assert "digraph ExecutionGraph" in dot
    assert "node_0" in dot
    assert "node_1" in dot
    assert "data.obs" in dot
    assert "node_0 -> node_1" in dot

def test_execution_graph_to_dot_with_pruning():
    """Verify that pruned nodes have dashed style in DOT."""
    k_target = Key("target", Observation)
    k_garbage = Key("garbage", Observation)
    
    c1 = SimpleMockComponent("Active", requires=set(), provides={k_target: WriteMode.NEW})
    c2 = SimpleMockComponent("Pruned", requires=set(), provides={k_garbage: WriteMode.NEW})
    
    # Prune by only targeting k_target
    graph = build_execution_graph(
        components=[c1, c2],
        initial_keys=set(),
        target_keys={k_target}
    )
    
    dot = graph.to_dot()
    
    # node_0 is Active, node_1 is Pruned
    assert "node_0" in dot
    assert "node_1" in dot
    
    # Active node should have filled style (and likely bold or solid)
    # Pruned node should have dashed style
    assert re.search(r'node_1 \[.*style="[^"]*dashed[^"]*"', dot)
    assert re.search(r'node_0 \[.*style="[^"]*bold[^"]*"', dot) # it's terminal

def test_execution_graph_to_dot_initial_keys():
    """Verify that initial keys are correctly visualized."""
    k_input = Key("input.data", Observation)
    c1 = SimpleMockComponent("Processor", requires={k_input}, provides={})
    
    graph = build_execution_graph(
        components=[c1],
        initial_keys={k_input},
        target_keys=None # No pruning
    )
    
    dot = graph.to_dot()
    
    assert "initial" in dot
    assert "input.data" in dot
    assert "initial -> node_0" in dot
