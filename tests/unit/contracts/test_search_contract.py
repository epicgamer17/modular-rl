import torch
import numpy as np
import pytest
from core import Blackboard
from components.search.mcts_component import MCTSSearchComponent
from components.selectors.discrete import ActionSelectorComponent
from components.memory.buffer import SequenceBufferComponent

# Module-level marker for unit tests
pytestmark = pytest.mark.unit

class MockAgentNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.training = True

class MockSearchEngine:
    def __init__(self, num_actions=4):
        self.num_actions = num_actions
        self.num_simulations = 50

    def run(self, obs, info, agent_network):
        # Return (root_value, exploratory_policy, target_policy, best_action, search_metadata)
        root_value = torch.tensor([1.5])
        exploratory_policy = torch.tensor([0.1, 0.4, 0.3, 0.2])
        target_policy = torch.tensor([0.0, 0.5, 0.5, 0.0])
        best_action = 1
        search_metadata = {"simulations": 50}
        return root_value, exploratory_policy, target_policy, best_action, search_metadata

class MockReplayBuffer:
    def __init__(self):
        self.stored_data = []
        self.buffers = {"policies": np.zeros((1, 4))}

    def store_aggregate(self, sequence):
        # In a real buffer, this would store the whole sequence.
        # For this mock, we'll extract the data from the sequence object.
        for i in range(len(sequence.action_history)):
             self.stored_data.append({
                 "policy": sequence.policy_history[i],
                 "value": sequence.value_history[i]
             })
    
    @property
    def size(self):
        return len(self.stored_data)

def test_search_selector_buffer_alignment():
    """
    Verifies the absolute contract hand-off:
    MCTS -> (search_*) -> Selector -> (metadata) -> Buffer
    """
    torch.manual_seed(42)
    
    # 1. Setup
    blackboard = Blackboard()
    blackboard.data["obs"] = torch.randn(1, 10)
    blackboard.data["info"] = {}
    
    agent_net = MockAgentNetwork()
    search_eng = MockSearchEngine()
    buffer = MockReplayBuffer()
    
    mcts_comp = MCTSSearchComponent(search_eng, agent_net)
    # Selector only cares about selection
    selector_comp = ActionSelectorComponent(input_key="search_policy")
    # Buffer handles target extraction
    buffer_comp = SequenceBufferComponent(
        buffer, 
        num_players=1,
        target_policy_key="search_target_policy",
        target_value_key="search_value"
    )
    
    # 2. Execute Search
    mcts_comp.execute(blackboard)
    
    # Assert Search Contract Keys
    assert "search_policy" in blackboard.predictions
    assert "search_target_policy" in blackboard.predictions
    assert "search_value" in blackboard.predictions
    
    # 3. Execute Selector
    selector_comp.execute(blackboard)
    
    # Assert Selector DOES NOT bridge targets anymore
    meta = blackboard.meta["action_metadata"]
    assert torch.allclose(meta["policy"], torch.tensor([0.1, 0.4, 0.3, 0.2]))
    assert "target_policies" not in meta, "Selector should no longer write target_policies"
    assert "value" not in meta, "Selector should no longer write value"
    
    # 4. Execute Buffer
    # Mock some data for step
    blackboard.data["reward"] = torch.tensor([1.0])
    blackboard.data["action"] = torch.tensor([1])
    blackboard.data["done"] = True
    blackboard.meta["next_info"] = {}
    blackboard.data["next_obs"] = torch.randn(1, 10)
    
    buffer_comp.execute(blackboard)
    
    # Assert Buffer Storage alignment
    assert len(buffer.stored_data) == 1
    stored = buffer.stored_data[0]
    # Buffer should have pulled directly from blackboard
    assert np.allclose(stored["policy"], np.array([0.0, 0.5, 0.5, 0.0])), "Buffer must pull search_target_policy from blackboard"
    assert stored["value"] == 1.5, "Buffer must pull search_value from blackboard"

if __name__ == "__main__":
    pytest.main([__file__])
