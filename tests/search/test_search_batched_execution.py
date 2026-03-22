import pytest

pytestmark = pytest.mark.integration

import torch
import torch.nn as nn
from modules.models.agent_network import AgentNetwork
from configs.agents.muzero import MuZeroConfig
from configs.games.tictactoe import TicTacToeConfig
from search import ModularSearch
from search.search_selectors import TopScoreSelection, SamplingSelection
from search.scoring_methods import UCBScoring, PriorScoring
from search.backpropogation import AverageDiscountedReturnBackpropagator
from search.initial_searchsets import SelectAll
from search.root_policies import VisitFrequencyPolicy
from search.pruners import NoPruning


def test_batched_search():
    import numpy as np

    torch.manual_seed(42)
    np.random.seed(42)
    # 1. Setup minimal MuZero components
    game_config = TicTacToeConfig()
    config_dict = {
        "num_simulations": 10,
        "search_batch_size": 5,  # Force batching
        "unroll_steps": 2,
        "gumbel": False,
        "use_virtual_mean": False,
        "backbone": {"type": "dense", "hidden_dim": 32},
        "value_head": {},
        "reward_head": {},
        "policy_head": {},
        "policy_loss_function": nn.CrossEntropyLoss(),
        "action_selector": {"base": {"type": "categorical", "kwargs": {}}},
    }
    config = MuZeroConfig(config_dict, game_config)
    device = torch.device("cpu")

    # Corrected AgentNetwork initialization
    num_actions = 9
    input_shape = (5, 3, 3)  # Handled by TicTacToe env wrappers
    # Create shared network
    model = AgentNetwork(config, input_shape, num_actions)

    # 2. Setup Search Algorithm
    search = ModularSearch(
        config=config,
        device=device,
        num_actions=num_actions,
    )

    # 3. Create dummy observation and inference functions
    obs = torch.zeros((1, *input_shape))  # Single observation with batch dim
    info = {"legal_moves": [list(range(num_actions))], "player": 0}

    def predict_initial_inference(state):
        return model.obs_inference(state)

    def predict_recurrent_inference(network_state, actions):
        return model.hidden_state_inference(network_state, actions)

    inference_fns = {
        "obs": predict_initial_inference,
        "hidden_state": predict_recurrent_inference,
    }

    # 4. Run Search - This should no longer raise ValueError
    print("Running batched search...")
    try:
        results = search.run(
            observation=obs,
            info=info,
            agent_network=model,
        )
        print("Batched search passed successfully!")
    except Exception as e:
        print(f"Batched search failed with error: {e}")
        import traceback

        traceback.print_exc()
        raise e


if __name__ == "__main__":
    test_batched_search()
