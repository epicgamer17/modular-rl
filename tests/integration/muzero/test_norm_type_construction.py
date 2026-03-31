import pytest
import torch
import torch.nn as nn

from agents.factories.model import build_agent_network
from configs.agents.muzero import MuZeroConfig

pytestmark = pytest.mark.integration

_NORM_MODULE_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)


def _assert_norm_type(module: nn.Module, norm_type: str, expected_cls: type[nn.Module]) -> None:
    norm_layers = [submodule for submodule in module.modules() if isinstance(submodule, _NORM_MODULE_TYPES)]

    if norm_type == "none":
        assert not norm_layers, (
            f"Expected no normalization layers for norm_type='none', found "
            f"{[type(layer).__name__ for layer in norm_layers]}."
        )
        return

    assert norm_layers, f"Expected {expected_cls.__name__} layers, but found none."
    assert all(isinstance(layer, expected_cls) for layer in norm_layers), (
        f"Expected only {expected_cls.__name__} layers, found "
        f"{[type(layer).__name__ for layer in norm_layers]}."
    )


@pytest.mark.parametrize(
    ("norm_type", "expected_cls"),
    [("none", nn.Identity), ("batch", nn.BatchNorm1d), ("layer", nn.LayerNorm)],
)
def test_muzero_applies_norm_type_to_dense_backbone_slots_and_head_necks(
    make_muzero_config_dict,
    cartpole_game_config,
    norm_type,
    expected_cls,
):
    cfg = make_muzero_config_dict(
        representation_backbone={
            "type": "mlp",
            "widths": [16, 16],
            "norm_type": norm_type,
        },
        dynamics_backbone={
            "type": "mlp",
            "widths": [16, 16],
            "norm_type": norm_type,
        },
        prediction_backbone={
            "type": "mlp",
            "widths": [16, 16],
            "norm_type": norm_type,
        },
        value_head={
            "output_strategy": {"type": "scalar"},
            "neck": {"type": "mlp", "widths": [16], "norm_type": norm_type},
        },
        policy_head={
            "output_strategy": {"type": "categorical"},
            "neck": {"type": "mlp", "widths": [16], "norm_type": norm_type},
        },
        reward_head={
            "output_strategy": {"type": "scalar"},
            "neck": {"type": "mlp", "widths": [16], "norm_type": norm_type},
        },
    )
    config = MuZeroConfig(cfg, cartpole_game_config)

    network = build_agent_network(
        config=config,
        obs_dim=(4,),
        num_actions=cartpole_game_config.num_actions,
    )

    world_model = network.components["world_model"]
    _assert_norm_type(network.components["representation"], norm_type, expected_cls)
    _assert_norm_type(world_model.dynamics_pipeline.dynamics, norm_type, expected_cls)
    _assert_norm_type(network.components["prediction"], norm_type, expected_cls)
    _assert_norm_type(
        network.components["behavior_heads"]["state_value"].neck,
        norm_type,
        expected_cls,
    )
    _assert_norm_type(
        network.components["behavior_heads"]["policy_logits"].neck,
        norm_type,
        expected_cls,
    )
    _assert_norm_type(world_model.heads["reward_logits"].neck, norm_type, expected_cls)

    obs = torch.rand(2, 4)
    latent = torch.rand(2, 16)
    action = torch.zeros(2, dtype=torch.long)

    inference = network.obs_inference(obs)
    wm_out = world_model.recurrent_inference(latent, action)

    assert inference.policy.logits.shape == (2, cartpole_game_config.num_actions)
    assert inference.value.shape == (2,)
    assert wm_out.reward.shape == (2, 1)


@pytest.mark.parametrize(
    ("norm_type", "expected_cls"),
    [("none", nn.Identity), ("batch", nn.BatchNorm1d), ("layer", nn.LayerNorm)],
)
def test_muzero_applies_norm_type_to_stochastic_world_model_slots(
    make_muzero_config_dict,
    cartpole_game_config,
    norm_type,
    expected_cls,
):
    cfg = make_muzero_config_dict(
        stochastic=True,
        num_chance=4,
        representation_backbone={
            "type": "mlp",
            "widths": [16, 16],
            "norm_type": "none",
        },
        dynamics_backbone={
            "type": "mlp",
            "widths": [16, 16],
            "norm_type": norm_type,
        },
        afterstate_dynamics_backbone={
            "type": "mlp",
            "widths": [16, 16],
            "norm_type": norm_type,
        },
        chance_encoder_backbone={
            "type": "mlp",
            "widths": [16, 16],
            "norm_type": norm_type,
        },
        chance_probability_head={
            "output_strategy": {"type": "categorical"},
            "neck": {"type": "mlp", "widths": [16], "norm_type": norm_type},
        },
        reward_head={
            "output_strategy": {"type": "scalar"},
            "neck": {"type": "identity"},
        },
        value_head={
            "output_strategy": {"type": "scalar"},
            "neck": {"type": "identity"},
        },
        policy_head={
            "output_strategy": {"type": "categorical"},
            "neck": {"type": "identity"},
        },
    )
    config = MuZeroConfig(cfg, cartpole_game_config)

    network = build_agent_network(
        config=config,
        obs_dim=(4,),
        num_actions=cartpole_game_config.num_actions,
    )

    world_model = network.components["world_model"]
    stochastic_dynamics = world_model.dynamics_pipeline

    _assert_norm_type(stochastic_dynamics.afterstate_dynamics, norm_type, expected_cls)
    _assert_norm_type(stochastic_dynamics.dynamics, norm_type, expected_cls)
    _assert_norm_type(stochastic_dynamics.encoder, norm_type, expected_cls)
    _assert_norm_type(stochastic_dynamics.sigma_head.neck, norm_type, expected_cls)

    current_latent = torch.rand(2, 16)
    action = torch.zeros(2, dtype=torch.long)
    encoder_inputs = torch.rand(2, 1, 8)

    out = stochastic_dynamics(
        current_latent,
        action,
        encoder_inputs=encoder_inputs,
        k=0,
    )

    assert out["next_latent"].shape == (2, 16)
    assert out["chance_logits"].shape == (2, 4)
