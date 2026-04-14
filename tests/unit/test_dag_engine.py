import torch
import pytest
from typing import Dict, Any, Set
from core.blackboard import Blackboard
from core.component import PipelineComponent
from core.contracts import Key, SemanticType, Observation, Reward, ValueEstimate, Scalar, ShapeContract
from core.blackboard_engine import BlackboardEngine

# Tier 1 Unit Test Marker
pytestmark = pytest.mark.unit

class MockComponent(PipelineComponent):
    def __init__(self, name: str, requires: Set[Key], provides: Dict[Key, str]):
        self._name = name
        self._requires = requires
        self._provides = provides
        self.executed_count = 0

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Dict[Key, str]:
        return self._provides

    def execute(self, blackboard: Blackboard) -> Dict[str, Any]:
        self.executed_count += 1
        # Simple logic: provide the keys requested with dummy data if not present
        updates = {}
        for key in self.provides:
            if key.path not in updates:
                updates[key.path] = torch.zeros((1, 1))
        return updates

def test_dag_validation_success():
    """Verifies that a valid DAG passes validation and respects ordering."""
    k_obs = Key("data.obs", Observation)
    k_val = Key("losses.value", ValueEstimate[Scalar])
    
    c1 = MockComponent("Producer", requires=set(), provides={k_obs: "new"})
    c2 = MockComponent("Consumer", requires={k_obs}, provides={k_val: "new"})
    
    engine = BlackboardEngine(components=[c1, c2], device=torch.device("cpu"))
    
    # Assert correct topological order in execution graph
    assert engine.execution_graph.execution_order == (0, 1)

def test_dag_validation_missing_dependency():
    """Verifies build-time failure when a dependency is missing."""
    k_obs = Key("data.obs", Observation)
    c1 = MockComponent("Consumer", requires={k_obs}, provides={})
    
    with pytest.raises(RuntimeError, match="Missing dependencies in pipeline DAG"):
        BlackboardEngine(components=[c1], device=torch.device("cpu"))

def test_dag_validation_semantic_mismatch():
    """Verifies build-time failure when semantic types are incompatible."""
    k_obs = Key("data.obs", Observation)
    k_rew = Key("data.obs", Reward) # Same path, different semantic type
    
    c1 = MockComponent("Producer", requires=set(), provides={k_obs: "new"})
    c2 = MockComponent("Consumer", requires={k_rew}, provides={})
    
    with pytest.raises(RuntimeError, match="SEMANTIC MISMATCH"):
        BlackboardEngine(components=[c1, c2], device=torch.device("cpu"))

def test_dag_validation_shape_mismatch():
    """Verifies build-time failure when ShapeContracts conflict."""
    k_p = Key("data.x", Observation, shape=ShapeContract(ndim=2))
    k_c = Key("data.x", Observation, shape=ShapeContract(ndim=3))
    
    c1 = MockComponent("P", requires=set(), provides={k_p: "new"})
    c2 = MockComponent("C", requires={k_c}, provides={})
    
    with pytest.raises(RuntimeError, match="ndim mismatch"):
        BlackboardEngine(components=[c1, c2], device=torch.device("cpu"))

def test_engine_execution_flow():
    """Verifies fake batch execution and output existence on Blackboard."""
    k_obs = Key("data.obs", Observation)
    k_val = Key("losses.value", ValueEstimate[Scalar])
    
    c1 = MockComponent("P", requires=set(), provides={k_obs: "new"})
    c2 = MockComponent("C", requires={k_obs}, provides={k_val: "new"})
    
    engine = BlackboardEngine(components=[c1, c2], device=torch.device("cpu"))
    
    fake_batch = [{"dummy": torch.tensor([1])}]
    results = list(engine.step(fake_batch))
    
    assert len(results) == 1
    assert c1.executed_count == 1
    assert c2.executed_count == 1

def test_lazy_execution_skipping():
    """Verifies that lazy=True skips components whose outputs are already satisfied."""
    k_obs = Key("data.obs", Observation)
    k_val = Key("losses.value", ValueEstimate[Scalar])
    
    # c1 provides obs, c2 provides value (which is a terminal sink by default if it's in losses.*)
    c1 = MockComponent("P", requires=set(), provides={k_obs: "new"})
    c2 = MockComponent("C", requires={k_obs}, provides={k_val: "new"})
    
    engine = BlackboardEngine(components=[c1, c2], device=torch.device("cpu"), lazy=True)
    
    # Provide 'data.obs' in the batch directly
    obs_tensor = torch.randn(2, 4)
    fake_batch = [{"obs": obs_tensor}]
    
    results = list(engine.step(fake_batch))
    
    # c1 should be skipped because 'data.obs' is already in the batch (blackboard.data)
    assert c1.executed_count == 0
    assert c2.executed_count == 1
    assert results[0]["meta"]["lazy_skipped"] == 1

def test_blackboard_diffing():
    """Verifies that diff=True captures component-level changes."""
    k_obs = Key("data.obs", Observation)
    k_loss = Key("losses.value", ValueEstimate[Scalar])
    c1 = MockComponent("P", requires=set(), provides={k_obs: "new", k_loss: "new"})
    
    engine = BlackboardEngine(components=[c1], device=torch.device("cpu"), diff=True)
    
    fake_batch = [{"dummy": torch.tensor([0])}]
    results = list(engine.step(fake_batch))
    
    diffs = results[0]["meta"]["blackboard_diffs"]
    assert len(diffs) == 1
    assert "data.obs" in diffs[0].added

def test_dag_execution_graph_pruning():
    """Verifies backward-reachability pruning from terminal sinks."""
    # c1 provides data.x (not terminal), c2 provides losses.y (terminal)
    k_x = Key("data.x", Observation)
    k_y = Key("losses.y", ValueEstimate[Scalar])
    
    c1 = MockComponent("NonTerminal", requires=set(), provides={k_x: "new"})
    c2 = MockComponent("Terminal", requires={k_x}, provides={k_y: "new"})
    
    engine = BlackboardEngine(components=[c1, c2], device=torch.device("cpu"))
    
    # Both should be active (c1 is reachable from c2 which is terminal)
    assert engine.execution_graph.execution_order == (0, 1)

def test_dag_overwrite_mode_validation():
    """Verifies that 'overwrite' mode requires the key to already exist."""
    k_obs = Key("data.obs", Observation)
    k_loss = Key("losses.y", ValueEstimate[Scalar])
    # Component tries to overwrite a key that doesn't exist (no initial_keys provided)
    # But it also provides a terminal sink so it's not pruned
    c1 = MockComponent("Overwriter", requires=set(), provides={k_obs: "overwrite", k_loss: "new"})
    
    with pytest.raises(RuntimeError, match="STAGE OVERWRITE ERROR"):
        BlackboardEngine(components=[c1], device=torch.device("cpu"))
