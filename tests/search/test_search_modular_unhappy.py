import pytest
import torch
import numpy as np
from search.search_py.modular_search import ModularSearch
from configs.agents.muzero import MuZeroConfig
from modules.world_models.inference_output import InferenceOutput

pytestmark = pytest.mark.unit


class MockEmptyDistribution:
    """A strictly typed mock to represent an empty distribution output."""

    def __init__(self):
        self.logits = None
        self.probs = None


class MockBadSearchNetwork(torch.nn.Module):
    """A dummy network that intentionally returns invalid search outputs."""

    def __init__(self):
        super().__init__()

    def obs_inference(self, obs):
        # Return an InferenceOutput missing both logits and probs
        return InferenceOutput(
            value=torch.tensor([0.0]),
            policy=MockEmptyDistribution(),
            network_state=None,
        )


def test_modular_search_missing_policy_probs(
    make_muzero_config_dict, cartpole_game_config
):
    torch.manual_seed(42)
    np.random.seed(42)

    config = MuZeroConfig(make_muzero_config_dict(), cartpole_game_config)

    # Initialize ModularSearch. We pass None to strategies because it will
    # crash in the inference phase before these are ever called.
    search = ModularSearch(
        config=config,
        device=torch.device("cpu"),
        num_actions=2,
        root_selection_strategy=None,
        decision_selection_strategy=None,
        chance_selection_strategy=None,
        root_target_policy=None,
        root_exploratory_policy=None,
        prior_injectors=[],
        root_searchset=None,
        internal_searchset=None,
        pruning_method=None,
        internal_pruning_method=None,
        backpropagator=None,
    )

    bad_net = MockBadSearchNetwork()
    dummy_obs = torch.randn(1, 4)
    dummy_info = {"legal_moves": [[0, 1]], "player_id": 0}

    # The search should immediately identify the missing probabilities and safely crash
    with pytest.raises(
        ValueError, match="Search requires a policy distribution with logits/probs"
    ):
        search.run(
            observation=dummy_obs, info=dummy_info, agent_network=bad_net
        )
