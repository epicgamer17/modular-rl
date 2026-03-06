import torch
import math
from search.aos_search.search_factories import build_search_pipeline
from dataclasses import dataclass


class MockConfig:
    def __init__(self):
        # Mandatory fields for build_search_pipeline
        self.num_simulations = 4
        self.max_search_depth = 5
        self.max_nodes = 100
        self.num_codes = 1
        self.pb_c_init = 1.25
        self.pb_c_base = 19652
        self.discount_factor = 0.99
        self.use_dirichlet = True
        self.dirichlet_alpha = 0.25
        self.dirichlet_fraction = 1.0  # Force all noise for detection
        self.search_batch_size = 1
        self.virtual_loss = 3.0
        self.use_virtual_mean = False
        self.bootstrap_method = "v_mix"
        self.policy_extraction = "visit_count"
        self.scoring_method = "ucb"
        self.use_value_prefix = False
        self.internal_decision_modifier = "none"
        self.internal_chance_modifier = "none"
        self.backprop_method = "average"
        self.use_sequential_halving = False
        self.gumbel_m = 16
        self.gumbel_cvisit = 50.0
        self.gumbel_cscale = 1.0
        self.known_bounds = None

        @dataclass
        class Game:
            num_players = 2
            num_actions = 4

        self.game = Game()


def verify_root_noise():
    device = torch.device("cpu")
    num_actions = 4
    config = MockConfig()

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
            # Return uniform logits (all 0s)
            return MockOutput(
                value=torch.zeros(B),
                reward=torch.zeros(B),
                to_play=torch.zeros(B, dtype=torch.int8),
                policy=MockPolicy(logits=torch.zeros((B, self.num_actions))),
            )

        def hidden_state_inference(self, parent_indices, actions):
            B = parent_indices.shape[0]
            return MockOutput(
                value=torch.zeros(B),
                reward=torch.zeros(B),
                to_play=torch.zeros(B, dtype=torch.int8),
                policy=MockPolicy(logits=torch.zeros((B, self.num_actions))),
            )

    network = MockNetwork(num_actions)

    # Monkeypatch FlatTree.allocate to capture the tree
    from search.aos_search.tree import FlatTree

    original_allocate = FlatTree.allocate
    captured_tree = []

    def mock_allocate(*args, **kwargs):
        tree = original_allocate(*args, **kwargs)
        captured_tree.append(tree)
        return tree

    FlatTree.allocate = mock_allocate

    # Build search pipeline
    run_mcts = build_search_pipeline(config, device, num_actions)

    obs = torch.zeros((1, 10))
    info = {}  # No legal moves, so all actions valid
    to_play = torch.tensor([0], dtype=torch.int8)

    print("\n--- Running MCTS Pipeline to verify root noise ---")
    run_mcts(obs, info, to_play, network)

    if not captured_tree:
        print("FAILURE: FlatTree was not allocated")
        return

    tree = captured_tree[0]
    root_priors = tree.children_prior_logits[0, 0, :num_actions]
    print(f"Root priors (Node 0): {root_priors}")

    # Check if priors are noised. If they are noised from uniform zero,
    # they should be log(noise_p) which won't be uniform zero.
    if not torch.allclose(root_priors, torch.zeros(num_actions)):
        print("SUCCESS: Root priors were correctly noised outside the simulation loop")
    else:
        print("FAILURE: Root priors were NOT noised (they stayed uniform zeros)")


if __name__ == "__main__":
    verify_root_noise()
