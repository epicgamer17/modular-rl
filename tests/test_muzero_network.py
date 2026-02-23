import torch
import gymnasium as gym
from configs.agents.muzero import MuZeroConfig
from modules.world_models.muzero_world_model import MuzeroWorldModel
from modules.agent_nets.modular import ModularAgentNetwork
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
        "action_selector": {"base": {"type": "categorical", "kwargs": {}}},
    }

    print("Initializing MuZero Config...")
    config = MuZeroConfig(config_dict, game)

    print("Initializing Network...")
    input_shape = (4,)
    config_dict = {
        "world_model_cls": MuzeroWorldModel,
        "residual_layers": [],
        "representation_dense_layer_widths": [64],
        "dynamics_dense_layer_widths": [64],
        "actor_dense_layer_widths": [64],
        "critic_dense_layer_widths": [64],
        "reward_dense_layer_widths": [64],
        "actor_conv_layers": [],
        "critic_conv_layers": [],
        "reward_conv_layers": [],
        "to_play_conv_layers": [],
        "num_simulations": 2,
        "minibatch_size": 2,
        "support_range": 5,  # Smaller support for testing
    }

    net = ModularAgentNetwork(config, input_shape, config.game.num_actions)

    print("Parameters check:")
    print(f"Prediction Value Head: {net.value_head}")
    print(f"Prediction Policy Head: {net.policy_head}")

    if hasattr(net, "prediction"):
        print("FAILED: Network still has 'prediction' attribute.")

    print("Running forward pass (obs_inference)...")
    obs = torch.randn(1, *input_shape)
    outputs = net.obs_inference(obs)
    value = outputs.value
    policy = outputs.policy
    hidden = outputs.network_state
    # Value is scalar/tensor, Policy is Distribution
    if isinstance(value, torch.Tensor):
        print(f"Value shape: {value.shape}")

    # Check Recurrent Inference (New Signature)
    print("Checking Recurrent Inference (New Signature)...")

    if config.stochastic:
        dummy_action_rec = torch.nn.functional.one_hot(
            torch.tensor([0]), num_classes=10
        ).float()
    else:
        dummy_action_rec = torch.nn.functional.one_hot(
            torch.tensor([0]), num_classes=5
        ).float()

    rec_out = net.hidden_state_inference(outputs.network_state, dummy_action_rec)
    if isinstance(rec_out.reward, torch.Tensor):
        print(f"Recurrent Reward shape: {rec_out.reward.shape}")

    if config.stochastic:
        print("Checking Afterstate Prediction...")
        if hasattr(net, "afterstate_prediction") and hasattr(
            net.afterstate_prediction, "head"
        ):
            print("FAILED: Afterstate Prediction still has 'head' attribute.")

        dummy_hidden = hidden
        dummy_action = torch.tensor([[0]])

        as_out = net.afterstate_inference(dummy_hidden, dummy_action)
        as_value = as_out.value
        as_policy = as_out.policy  # Chance distribution

        if isinstance(as_value, torch.Tensor):
            print(f"Afterstate Value shape: {as_value.shape}")
        # as_policy is Distribution

    print("Success!")


def test_learner_inference():
    print("\n--- Testing Learner Inference ---")
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
        "action_selector": {"base": {"type": "categorical", "kwargs": {}}},
    }

    config = MuZeroConfig(config_dict, game)
    input_shape = (4,)
    net = ModularAgentNetwork(config, input_shape, config.game.num_actions)

    # Mock batch
    batch_size = 2
    unroll_steps = 3
    batch = {
        "observations": torch.randn(batch_size, *input_shape),
        "actions": torch.randint(0, 5, (batch_size, unroll_steps)),
        "unroll_observations": torch.randn(batch_size, unroll_steps + 1, *input_shape),
    }

    print("Running learner_inference...")
    learning_output = net.learner_inference(batch)

    print(f"Values shape: {learning_output.values.shape}")
    print(f"Policies shape: {learning_output.policies.shape}")
    print(f"Rewards shape: {learning_output.rewards.shape}")
    print(f"Latents shape: {learning_output.latents.shape}")

    # Expected shapes:
    # State-aligned tensors have T+1 steps; rewards are transition-aligned (T steps).
    assert learning_output.values.shape == (batch_size, unroll_steps + 1, 1)
    assert learning_output.policies.shape == (batch_size, unroll_steps + 1, 5)
    assert learning_output.rewards.shape == (
        batch_size,
        unroll_steps,
        1,
    ), f"Expected rewards shape (B, T, 1), got {learning_output.rewards.shape}"
    assert learning_output.latents.shape == (batch_size, unroll_steps + 1, 4)

    if config.stochastic:
        print(f"Latents Afterstates shape: {learning_output.latents_afterstates.shape}")
        print(f"Chance Logits shape: {learning_output.chance_logits.shape}")
        print(f"Chance Values shape: {learning_output.chance_values.shape}")
        # Stochastic tensors are transition-aligned (T steps, same as rewards)
        assert learning_output.latents_afterstates.shape == (
            batch_size,
            unroll_steps,
            4,
        )
        assert learning_output.chance_logits.shape == (batch_size, unroll_steps, 10)
        assert learning_output.chance_values.shape == (batch_size, unroll_steps, 1)

    print("learner_inference Test Success!")


if __name__ == "__main__":
    test_muzero_network_structure()
    test_learner_inference()
