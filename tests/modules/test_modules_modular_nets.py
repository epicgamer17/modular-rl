import copy

import pytest
import torch

from configs.agents.rainbow_dqn import RainbowConfig
from modules.agent_nets.modular import ModularAgentNetwork

pytestmark = pytest.mark.unit


def _build_small_rainbow_config(
    rainbow_cartpole_replay_config,
    make_cartpole_config,
    *,
    num_actions: int,
):
    game_config = make_cartpole_config(num_actions=num_actions)
    config_dict = copy.deepcopy(rainbow_cartpole_replay_config.config_dict)
    config_dict.update(
        {
            "dueling": False,
            "backbone": {"type": "identity"},
            "head": {
                "output_strategy": {"type": "scalar"},
                "hidden_widths": [8],
            },
        }
    )
    return RainbowConfig(config_dict, game_config)


def test_modules_modular_net_rainbow_forward_shapes(
    rainbow_cartpole_replay_config, make_cartpole_config
):
    torch.manual_seed(42)

    config = _build_small_rainbow_config(
        rainbow_cartpole_replay_config, make_cartpole_config, num_actions=3
    )
    net = ModularAgentNetwork(config, input_shape=(4,), num_actions=3)

    obs_batch = torch.randn(5, 4)
    actor_output = net.obs_inference(obs_batch)
    learner_output = net.learner_inference({"observations": obs_batch})

    assert actor_output.q_values is not None
    assert actor_output.value is not None
    assert actor_output.policy is not None
    assert actor_output.q_values.shape == (5, 3)
    assert actor_output.value.shape == (5,)
    assert actor_output.policy.mean.shape == (5, 3)

    assert learner_output.q_values is not None
    assert learner_output.q_logits is not None
    assert learner_output.q_values.shape == (5, 3)
    assert learner_output.q_logits.shape == (5, 3)


def test_modules_modular_net_rainbow_unsqueezes_single_observation(
    rainbow_cartpole_replay_config, make_cartpole_config
):
    config = _build_small_rainbow_config(
        rainbow_cartpole_replay_config, make_cartpole_config, num_actions=2
    )
    net = ModularAgentNetwork(config, input_shape=(4,), num_actions=2)

    single_obs = torch.randn(4)
    actor_output = net.obs_inference(single_obs)

    assert actor_output.q_values is not None
    assert actor_output.value is not None
    assert actor_output.q_values.shape == (1, 2)
    assert actor_output.value.shape == (1,)


def test_modules_modular_net_rejects_unsupported_config(make_cartpole_config):
    game_config = make_cartpole_config()

    with pytest.raises(ValueError, match="Unsupported config type"):
        ModularAgentNetwork(
            config={"invalid": "config"},
            input_shape=(4,),
            num_actions=game_config.num_actions,
        )
