import pytest
import torch
import numpy as np
from search.aos_search.search_algorithm import ModularSearch
from configs.agents.muzero import MuZeroConfig

pytestmark = pytest.mark.integration


class MockAOSPipeline:
    """Mocks the compiled search pipeline to return fake vectorized outputs."""

    def __init__(self, batch_size=2):
        self.batch_size = batch_size

    class FakeSearchOutput:
        def __init__(self, b):
            self.root_values = torch.tensor([1.0, 2.0])
            self.exploratory_policy = torch.ones((b, 2)) * 0.5
            self.target_policy = torch.ones((b, 2)) * 0.5
            self.best_actions = torch.tensor([0, 1])

    def __call__(self, obs, info, to_play, net, trajectory_actions=None):
        return self.FakeSearchOutput(self.batch_size)


def test_aos_search_vectorized_unbatching(
    make_muzero_config_dict, cartpole_game_config
):
    torch.manual_seed(42)
    np.random.seed(42)

    config_dict = make_muzero_config_dict()
    config = MuZeroConfig(config_dict, cartpole_game_config)
    config.compilation.enabled = False  # Disable compilation for testing

    search = ModularSearch(config, torch.device("cpu"), num_actions=2)

    # Override the internal pipeline with our mock
    search._run_mcts = MockAOSPipeline(batch_size=2)

    # Create a batch of 2 observations
    batched_obs = torch.randn(2, 4)
    batched_info = [
        {"legal_moves": [0, 1], "player": 0},
        {"legal_moves": [0, 1], "player": 0},
    ]

    # Run the vectorized unbatching loop
    root_values, exp_policies, tgt_policies, best_actions, metadata = (
        search.run_vectorized(
            batched_obs=batched_obs,
            batched_info=batched_info,
            agent_network=None,  # Handled by mock
        )
    )

    # Verify everything unbatched perfectly into standard Python lists
    assert isinstance(root_values, list)
    assert len(root_values) == 2
    assert root_values == [1.0, 2.0]

    assert isinstance(best_actions, list)
    assert best_actions == [0, 1]

    assert len(metadata) == 2
    assert "network_value" in metadata[0]
