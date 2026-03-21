import copy

import pytest

pytestmark = pytest.mark.integration

import gymnasium as gym
import torch

from configs.agents.muzero import MuZeroConfig
from modules.agent_nets.modular import ModularAgentNetwork
from modules.world_models.modular_world_model import ModularWorldModel


class MockEnv:
    def __init__(self):
        self.action_space = gym.spaces.Discrete(5)

    def close(self):
        pass


def _build_muzero_test_config(
    rainbow_cartpole_replay_config, make_cartpole_config, **overrides
):
    game = make_cartpole_config(
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
        make_env=MockEnv,
    )

    config_dict = copy.deepcopy(rainbow_cartpole_replay_config.config_dict)
    config_dict.update(
        {
            "world_model_cls": ModularWorldModel,
            "stochastic": True,
            "num_chance": 10,
            "prediction_backbone": {"type": "identity"},
            "representation_backbone": {"type": "identity"},
            "dynamics_backbone": {"type": "identity"},
            "afterstate_dynamics_backbone": {"type": "identity"},
            "chance_encoder_backbone": {"type": "identity"},
            "value_head": {"output_strategy": {"type": "scalar"}},
            "minibatch_size": 2,
            "num_simulations": 2,
        }
    )
    config_dict.update(overrides)
    return MuZeroConfig(config_dict, game)


def test_muzero_network_structure(
    rainbow_cartpole_replay_config, make_cartpole_config
):
    config = _build_muzero_test_config(
        rainbow_cartpole_replay_config, make_cartpole_config
    )

    print("Initializing Network...")
    input_shape = (4,)
    net = ModularAgentNetwork(config, input_shape, config.game.num_actions)

    print("Parameters check:")
    print(f"Prediction Value Head: {net.components['behavior_heads']['state_value']}")
    print(f"Prediction Policy Head: {net.components['behavior_heads']['policy_logits']}")

    if hasattr(net, "prediction"):
        print("FAILED: Network still has 'prediction' attribute.")

    print("Running forward pass (obs_inference)...")
    obs = torch.randn(1, *input_shape)
    outputs = net.obs_inference(obs)
    value = outputs.value
    recurrent_state = outputs.recurrent_state
    if isinstance(value, torch.Tensor):
        print(f"Value shape: {value.shape}")

    print("Checking Recurrent Inference (New Signature)...")
    if config.stochastic:
        dummy_action_rec = torch.nn.functional.one_hot(
            torch.tensor([0]), num_classes=config.num_chance
        ).float()
    else:
        dummy_action_rec = torch.nn.functional.one_hot(
            torch.tensor([0]), num_classes=config.game.num_actions
        ).float()

    rec_out = net.hidden_state_inference(outputs.recurrent_state, dummy_action_rec)
    if isinstance(rec_out.reward, torch.Tensor):
        print(f"Recurrent Reward shape: {rec_out.reward.shape}")

    if config.stochastic:
        print("Checking Afterstate Prediction...")
        if hasattr(net, "afterstate_prediction") and hasattr(
            net.afterstate_prediction, "head"
        ):
            print("FAILED: Afterstate Prediction still has 'head' attribute.")

        dummy_action = torch.tensor([[0]])
        as_out = net.afterstate_inference(hidden, dummy_action)
        as_value = as_out.value
        if isinstance(as_value, torch.Tensor):
            print(f"Afterstate Value shape: {as_value.shape}")

    print("Success!")


def test_learner_inference(rainbow_cartpole_replay_config, make_cartpole_config):
    print("\n--- Testing Learner Inference ---")
    config = _build_muzero_test_config(
        rainbow_cartpole_replay_config, make_cartpole_config
    )
    input_shape = (4,)
    net = ModularAgentNetwork(config, input_shape, config.game.num_actions)

    batch_size = 2
    unroll_steps = 3
    batch = {
        "observations": torch.randn(batch_size, *input_shape),
        "actions": torch.randint(0, config.game.num_actions, (batch_size, unroll_steps)),
        "unroll_observations": torch.randn(batch_size, unroll_steps + 1, *input_shape),
    }

    print("Running learner_inference...")
    learning_output = net.learner_inference(batch)

    print(f"Values shape: {learning_output['state_value'].shape}")
    print(f"Policies shape: {learning_output['policy_logits'].shape}")
    print(f"Rewards shape: {learning_output['reward_logits'].shape}")
    print(f"Latents shape: {learning_output['latents'].shape}")

    assert learning_output["state_value"].shape == (batch_size, unroll_steps + 1, 1)
    assert learning_output["policy_logits"].shape == (
        batch_size,
        unroll_steps + 1,
        config.game.num_actions,
    )
    assert learning_output["reward_logits"].shape == (
        batch_size,
        unroll_steps + 1,
        1,
    ), f"Expected rewards shape (B, T+1, 1), got {learning_output['reward_logits'].shape}"
    assert learning_output["latents"].shape == (batch_size, unroll_steps + 1, 4)

    if config.stochastic:
        print(f"Latents Afterstates shape: {learning_output['latents_afterstates'].shape}")
        print(f"Chance Logits shape: {learning_output['chance_logits'].shape}")
        print(f"Chance Values shape: {learning_output['afterstate_value'].shape}")
        assert learning_output["latents_afterstates"].shape == (
            batch_size,
            unroll_steps,
            4,
        )
        assert learning_output["chance_logits"].shape == (
            batch_size,
            unroll_steps + 1,
            config.num_chance,
        )
        assert learning_output["afterstate_value"].shape == (batch_size, unroll_steps + 1, 1)

    print("learner_inference Test Success!")
