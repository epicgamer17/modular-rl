import pytest
from typing import Dict, Set, Any
from core.contracts import Key, Policy, Probs, Logits, WriteMode
from core.component import PipelineComponent
from core.execution_graph import build_execution_graph

pytestmark = pytest.mark.unit

class MockComponent(PipelineComponent):
    """Lighweight component for testing DAG wiring without side effects."""
    def __init__(self, name: str, requires: Set[Key], provides: Set[Key]):
        self._name = name
        self._requires = requires
        # Default all provides to WriteMode.NEW for testing
        self._provides = {k: WriteMode.NEW for k in provides}

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Dict[Key, WriteMode]:
        return self._provides

    def execute(self, blackboard: Any) -> Dict[str, Any]:
        return {}

def test_dag_compatibility_abstract_to_concrete():
    """Verify DAG builder allows a concrete provider to satisfy an abstract requirement."""
    key_concrete = Key("policy", Policy[Probs])
    key_abstract = Key("policy", Policy)
    
    comp1 = MockComponent("Provider", requires=set(), provides={key_concrete})
    comp2 = MockComponent("Consumer", requires={key_abstract}, provides=set())
    
    graph = build_execution_graph(
        components=[comp1, comp2],
        initial_keys=set(),
        target_keys=set()
    )
    assert len(graph.execution_order) == 2, "Both components should be active in the graph"

def test_dag_compatibility_concrete_to_abstract_fails_strictly():
    """Verify DAG builder rejects an abstract provider when a concrete requirement is specified (Strict Requirement)."""
    key_abstract = Key("policy", Policy)
    key_concrete = Key("policy", Policy[Probs])
    
    comp1 = MockComponent("Provider", requires=set(), provides={key_abstract})
    comp2 = MockComponent("Consumer", requires={key_concrete}, provides=set())
    
    with pytest.raises(RuntimeError, match="SEMANTIC MISMATCH"):
        build_execution_graph(
            components=[comp1, comp2],
            initial_keys=set(),
            target_keys=set()
        )

def test_dag_incompatibility_concrete_mismatch():
    """Verify DAG builder rejects mismatched concrete representations on the same path."""
    key_probs = Key("policy", Policy[Probs])
    key_logits = Key("policy", Policy[Logits])
    
    comp1 = MockComponent("Provider", requires=set(), provides={key_probs})
    comp2 = MockComponent("Consumer", requires={key_logits}, provides=set())
    
    with pytest.raises(RuntimeError, match="SEMANTIC MISMATCH"):
        build_execution_graph(
            components=[comp1, comp2],
            initial_keys=set(),
            target_keys=set()
        )
