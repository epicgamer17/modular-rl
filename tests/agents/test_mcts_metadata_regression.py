import pytest
import torch
import numpy as np
from types import SimpleNamespace
from agents.workers.actors import GymActor
from agents.action_selectors.selectors import ArgmaxSelector
from agents.action_selectors.policy_sources import SearchPolicySource
from replay_buffers.sequence import Sequence

pytestmark = pytest.mark.unit

class MockSearch:
    def run(self, obs, info, agent_network, trajectory_action=None, exploration=True):
        return (
            0.75, # root_value
            torch.tensor([0.2, 0.8]), # exploratory_policy
            torch.tensor([0.1, 0.9]), # target_policy
            1, # best_action
            {"mcts_simulations": 123, "mcts_search_time": 0.456} # search_metadata
        )

class MockNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_shape = (4,)
        self.num_actions = 2
    def obs_inference(self, obs):
        return None

class MockBuffer:
    def __init__(self):
        self.stored_sequences = []
    def store_aggregate(self, sequence):
        self.stored_sequences.append(sequence)

def test_mcts_metadata_merging_regression():
    """
    Check that MCTS value, root_value, and search_metadata are correctly
    merged into the actor's metadata and preserved in the sequence stats.
    """
    config = SimpleNamespace(
        compilation=SimpleNamespace(enabled=False),
        num_envs_per_worker=1
    )
    
    net = MockNetwork()
    search = MockSearch()
    policy_source = SearchPolicySource(search, net, config)
    selector = ArgmaxSelector()
    buffer = MockBuffer()
    
    class MockEnv:
        def __init__(self):
            # BaseActor expects an env with reset/step or last (for PZ)
            pass
        def reset(self, seed=None, options=None):
            return np.zeros(4, dtype=np.float32), {"legal_moves": [0, 1]}
        def step(self, action):
            return np.zeros(4, dtype=np.float32), 1.0, True, False, {"legal_moves": [0, 1]}

    actor = GymActor(
        env_factory=MockEnv,
        agent_network=net,
        action_selector=selector,
        replay_buffer=buffer,
        config=config,
        policy_source=policy_source
    )
    
    # Run one episode
    stats = actor.play_sequence()
    
    # 1. Verify play_sequence returned stats
    assert stats["mcts_simulations"] == 123
    assert stats["mcts_search_time"] == 0.456
    
    # 2. Verify sequence stored in buffer has correct stats
    stored_seq = buffer.stored_sequences[0]
    assert stored_seq.stats["mcts_simulations"] == 123
    assert stored_seq.stats["mcts_search_time"] == 0.456
    
    # 3. Verify policy in sequence is a tensor (not a Categorical)
    # Actor prefers target_policies over exploratory policy for the buffer.
    assert len(stored_seq.policy_history) == 1
    policy = stored_seq.policy_history[0]
    assert torch.is_tensor(policy)
    assert torch.allclose(policy, torch.tensor([0.1, 0.9]))
    
    # 4. Verify root_value/value in metadata
    # The value_history contains the values retrieved from metadata
    assert len(stored_seq.value_history) == 1
    assert stored_seq.value_history[0] == 0.75

if __name__ == "__main__":
    print("Running MCTS metadata merging regression test...")
    try:
        test_mcts_metadata_merging_regression()
        print("Test PASSED!")
    except Exception as e:
        print(f"Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
