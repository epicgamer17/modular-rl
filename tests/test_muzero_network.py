import torch
import gymnasium as gym
from configs.agents.muzero import MuZeroConfig
from modules.world_models.muzero_world_model import MuzeroWorldModel
from modules.agent_nets.muzero import AgentNetwork
from configs.games.game import GameConfig


class MockEnv:
    def __init__(self):
        self.action_space = gym.spaces.Discrete(5)

    def close(self):
        pass


def test_muzero_network_structure():
    game = GameConfig(
        max_score=100,
        min_score=0,
        is_discrete=True,
        is_image=False,
        is_deterministic=False,
        has_legal_moves=False,
        perfect_information=True,
        multi_agent=False,
        num_players=1,
        num_actions=5,
        make_env=lambda: MockEnv(),
    )

    config_dict = {
        "world_model_cls": MuzeroWorldModel,
        "stochastic": True,
        "num_chance": 10,
        "prediction_backbone": {"type": "identity"},
        "representation_backbone": {"type": "identity"},
        "dynamics_backbone": {"type": "identity"},
        "afterstate_dynamics_backbone": {"type": "identity"},
        "chance_encoder_backbone": {"type": "identity"},
        "value_head": {"output_strategy": {"type": "scalar"}},
    }

    print("Initializing MuZero Config...")
    config = MuZeroConfig(config_dict, game)

    print("Initializing Network...")
    input_shape = (4,)

    net = Network(config, config.game.num_actions, input_shape)

    print("Parameters check:")
    print(f"Prediction Value Head: {net.prediction.value_head}")
    print(f"Prediction Policy Head: {net.prediction.policy_head}")

    if hasattr(net.prediction, "head"):
        print("FAILED: Prediction still has 'head' attribute.")
        raise AssertionError("PredictionHead wrapper should be removed.")

    print("Running forward pass (initial_inference)...")
    obs = torch.randn(1, *input_shape)
    value, policy, hidden = net.initial_inference(obs)
    print(f"Value shape: {value.shape}")
    print(f"Policy shape: {policy.shape}")

    if config.stochastic:
        print("Checking Afterstate Prediction...")
        # afterstate_prediction uses same Prediction class
        if hasattr(net.afterstate_prediction, "head"):
            print("FAILED: Afterstate Prediction still has 'head' attribute.")

        dummy_hidden = hidden
        dummy_action = torch.tensor([[0]])
        afterstate, as_value, as_policy = net.afterstate_recurrent_inference(
            dummy_hidden, dummy_action
        )
        print(f"Afterstate Value shape: {as_value.shape}")
        print(f"Afterstate Policy (Chance) shape: {as_policy.shape}")

    print("Success!")


if __name__ == "__main__":
    test_muzero_network_structure()
