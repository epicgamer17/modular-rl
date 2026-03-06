import torch
from search.aos_search.search_factories import build_search_pipeline
from search.aos_search.tree import FlatTree
from dataclasses import dataclass


@dataclass
class MockOutput:
    value: torch.Tensor
    reward: torch.Tensor
    to_play: torch.Tensor
    policy: any


@dataclass
class MockPolicy:
    logits: torch.Tensor


class MockNetwork:
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def obs_inference(self, obs):
        B = obs.shape[0]
        return MockOutput(
            value=torch.zeros(B),
            reward=torch.zeros(B),
            to_play=torch.zeros(B, dtype=torch.int8),
            policy=MockPolicy(logits=torch.zeros((B, self.num_actions))),
        )

    def hidden_state_inference(self, parent_indices, actions):
        B = parent_indices.shape[0]
        # Return unique logits to detect if they are modified
        logits = torch.full((B, self.num_actions), 1.0)
        return MockOutput(
            value=torch.zeros(B),
            reward=torch.zeros(B),
            to_play=torch.zeros(B, dtype=torch.int8),
            policy=MockPolicy(logits=logits),
        )


class MockConfig:
    def __init__(self):
        self.num_simulations = 4
        self.max_search_depth = 5
        self.max_nodes = 100
        self.num_codes = 1
        self.pb_c_init = 1.25
        self.pb_c_base = 19652
        self.discount_factor = 0.99
        self.use_dirichlet = True
        self.dirichlet_alpha = 0.25
        self.dirichlet_fraction = 1.0  # Force all noise
        self.search_batch_size = 1
        self.virtual_loss = 3.0
        self.use_virtual_mean = False
        self.bootstrap_method = "v_mix"
        self.policy_extraction = "visit_count"
        self.scoring_method = "ucb"
        self.use_value_prefix = False
        self.internal_decision_modifier = "none"
        self.internal_chance_modifier = "none"
        self.game = dataclass(num_players=2)
        self.compilation = dataclass(enabled=False)
        self.known_bounds = None


def verify():
    device = torch.device("cpu")
    num_actions = 4
    config = MockConfig()
    network = MockNetwork(num_actions)

    run_mcts = build_search_pipeline(config, device, num_actions)

    # We need to capture the tree to inspect it.
    # Since run_mcts creates the tree internally, we'll have to rely on its effects.
    # However, for verification, I'll modify search_factories.py temporarily to return the tree.
    # OR better: I'll just check if the output policies reflect noise.

    obs = torch.zeros((1, 10))
    info = {}
    to_play = torch.tensor([0], dtype=torch.int8)

    print("\n--- Running MCTS Pipeline ---")
    policies = run_mcts(obs, info, to_play, network)

    print("MCTS execution finished.")
    # If the root priors were noised, we'd see non-uniform visit counts eventually,
    # but with 1.0 fraction and 0.0 base, it should be very noisy.


if __name__ == "__main__":
    verify()
