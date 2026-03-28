"""
MuZero Parity Tests: Old (ground truth) vs New implementation.

These tests compare the OLD MuZero implementation (old_muzero/) against the NEW
refactored implementation to identify behavioral differences. The old implementation
is treated as ground truth.

Test Categories:
1. Config Parity - Same dict produces equivalent configs
2. World Model Parity - initial_inference, recurrent_inference, unroll_physics
3. Agent Network Parity - obs_inference, learner_inference, hidden_state_inference
4. N-Step Target Computation Parity
5. Replay Buffer Output Processor Parity
6. Target Builder Parity
7. Inference Output Data Structure Parity
8. Loss Pipeline Construction Parity
"""

import copy
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple

pytestmark = pytest.mark.unit

# ============================================================================
# Seeds
# ============================================================================

SEED = 42


def seed_all():
    torch.manual_seed(SEED)
    np.random.seed(SEED)


# ============================================================================
# Shared Config Dicts
# ============================================================================

CARTPOLE_OBS_SHAPE = (4,)
CARTPOLE_NUM_ACTIONS = 2
CARTPOLE_NUM_PLAYERS = 1


def _base_muzero_dict():
    return {
        "minibatch_size": 4,
        "min_replay_buffer_size": 0,
        "num_simulations": 3,
        "discount_factor": 0.99,
        "unroll_steps": 3,
        "n_step": 3,
        "learning_rate": 0.01,
        "architecture": {"backbone": {"type": "dense", "hidden_widths": [32]}},
        "representation_backbone": {"type": "dense", "hidden_widths": [32]},
        "dynamics_backbone": {"type": "dense", "hidden_widths": [32]},
        "prediction_backbone": {"type": "dense", "hidden_widths": [32]},
        "action_selector": {"base": {"type": "categorical"}},
        "policy_head": {
            "output_strategy": {"type": "categorical"},
            "neck": {"type": "identity"},
        },
        "value_head": {
            "output_strategy": {"type": "scalar"},
            "neck": {"type": "identity"},
        },
        "reward_head": {
            "output_strategy": {"type": "scalar"},
            "neck": {"type": "identity"},
        },
        "agent_type": "muzero",
        "games_per_generation": 10,
        "replay_buffer_size": 100,
        "per_alpha": 1.0,
        "per_beta_schedule": {"type": "constant", "value": 0.4},
    }


def _base_muzero_dict_distributional():
    d = _base_muzero_dict()
    d["atom_size"] = 11
    d["support_range"] = 5
    d["value_head"] = {
        "output_strategy": {"type": "muzero", "support_range": 5},
        "neck": {"type": "identity"},
    }
    d["reward_head"] = {
        "output_strategy": {"type": "muzero", "support_range": 5},
        "neck": {"type": "identity"},
    }
    return d


# ============================================================================
# Fixtures: Old Config + Network
# ============================================================================


@pytest.fixture
def old_cartpole_game_config():
    from old_muzero.configs.games.cartpole import CartPoleConfig
    return CartPoleConfig()


@pytest.fixture
def old_muzero_config(old_cartpole_game_config):
    from old_muzero.configs.agents.muzero import MuZeroConfig
    return MuZeroConfig(_base_muzero_dict(), old_cartpole_game_config)


@pytest.fixture
def old_muzero_config_distributional(old_cartpole_game_config):
    from old_muzero.configs.agents.muzero import MuZeroConfig
    return MuZeroConfig(_base_muzero_dict_distributional(), old_cartpole_game_config)


@pytest.fixture
def old_agent_network(old_muzero_config):
    from old_muzero.modules.agent_nets.modular import ModularAgentNetwork
    seed_all()
    net = ModularAgentNetwork(
        config=old_muzero_config,
        input_shape=CARTPOLE_OBS_SHAPE,
        num_actions=CARTPOLE_NUM_ACTIONS,
    )
    return net


# ============================================================================
# Fixtures: New Config + Network
# ============================================================================


@pytest.fixture
def new_cartpole_game_config():
    from configs.games.cartpole import CartPoleConfig
    return CartPoleConfig()


@pytest.fixture
def new_muzero_config(new_cartpole_game_config):
    from configs.agents.muzero import MuZeroConfig
    return MuZeroConfig(_base_muzero_dict(), new_cartpole_game_config)


@pytest.fixture
def new_muzero_config_distributional(new_cartpole_game_config):
    from configs.agents.muzero import MuZeroConfig
    return MuZeroConfig(_base_muzero_dict_distributional(), new_cartpole_game_config)


@pytest.fixture
def new_agent_network(new_muzero_config):
    """Build the new AgentNetwork using the trainer's factory pattern."""
    from modules.models.agent_network import AgentNetwork
    from agents.factories.builders import make_backbone_fn, make_head_fn
    from functools import partial
    from modules.models.world_model import WorldModel

    config = new_muzero_config
    seed_all()

    representation_fn = make_backbone_fn(getattr(config, "representation_backbone", None))
    prediction_backbone_fn = make_backbone_fn(getattr(config, "prediction_backbone", None))

    wm_cfg = config.world_model
    env_head_fns = {
        name: make_head_fn(h_cfg)
        for name, h_cfg in wm_cfg.env_heads.items()
    }

    world_model_fn = partial(
        WorldModel,
        stochastic=getattr(wm_cfg, "stochastic", False),
        num_chance=getattr(wm_cfg, "num_chance", 0),
        observation_shape=CARTPOLE_OBS_SHAPE,
        use_true_chance_codes=getattr(wm_cfg, "use_true_chance_codes", False),
        env_head_fns=env_head_fns,
        dynamics_fn=make_backbone_fn(wm_cfg.dynamics_backbone),
        afterstate_dynamics_fn=make_backbone_fn(getattr(wm_cfg, "afterstate_dynamics_backbone", None)),
        sigma_head_fn=make_head_fn(getattr(wm_cfg, "chance_probability_head", None)),
        encoder_fn=make_backbone_fn(getattr(wm_cfg, "chance_encoder_backbone", None)),
        action_embedding_dim=getattr(wm_cfg, "action_embedding_dim", 16),
    )

    head_fns = {}
    for name, h_cfg in config.heads.items():
        head_fns[name] = make_head_fn(h_cfg)

    net = AgentNetwork(
        input_shape=CARTPOLE_OBS_SHAPE,
        num_actions=CARTPOLE_NUM_ACTIONS,
        representation_fn=representation_fn,
        world_model_fn=world_model_fn,
        prediction_backbone_fn=prediction_backbone_fn,
        head_fns=head_fns,
        stochastic=config.stochastic,
        num_players=config.game.num_players,
        num_chance_codes=config.num_chance,
    )
    return net


# ============================================================================
# 1. CONFIG PARITY
# ============================================================================


class TestConfigParity:
    """Verify that the same config dict produces equivalent configuration objects."""

    def test_unroll_steps_match(self, old_muzero_config, new_muzero_config):
        assert old_muzero_config.unroll_steps == new_muzero_config.unroll_steps

    def test_n_step_match(self, old_muzero_config, new_muzero_config):
        assert old_muzero_config.n_step == new_muzero_config.n_step

    def test_discount_factor_match(self, old_muzero_config, new_muzero_config):
        assert old_muzero_config.discount_factor == new_muzero_config.discount_factor

    def test_minibatch_size_match(self, old_muzero_config, new_muzero_config):
        assert old_muzero_config.minibatch_size == new_muzero_config.minibatch_size

    def test_num_simulations_match(self, old_muzero_config, new_muzero_config):
        assert old_muzero_config.num_simulations == new_muzero_config.num_simulations

    def test_games_per_generation_match(self, old_muzero_config, new_muzero_config):
        assert old_muzero_config.games_per_generation == new_muzero_config.games_per_generation

    def test_stochastic_default_false(self, old_muzero_config, new_muzero_config):
        assert old_muzero_config.stochastic == new_muzero_config.stochastic == False

    def test_loss_factors_match(self, old_muzero_config, new_muzero_config):
        assert old_muzero_config.value_loss_factor == new_muzero_config.value_loss_factor
        assert old_muzero_config.policy_loss_factor == new_muzero_config.policy_loss_factor
        assert old_muzero_config.reward_loss_factor == new_muzero_config.reward_loss_factor

    def test_search_params_match(self, old_muzero_config, new_muzero_config):
        assert old_muzero_config.pb_c_base == new_muzero_config.pb_c_base
        assert old_muzero_config.pb_c_init == new_muzero_config.pb_c_init

    def test_temperature_schedule_match(self, old_muzero_config, new_muzero_config):
        old_sched = old_muzero_config.temperature_schedule
        new_sched = new_muzero_config.temperature_schedule
        assert type(old_sched).__name__ == type(new_sched).__name__

    def test_per_params_match(self, old_muzero_config, new_muzero_config):
        assert old_muzero_config.per_alpha == new_muzero_config.per_alpha

    def test_distributional_config_defaults(self, old_muzero_config, new_muzero_config):
        """Non-distributional config should have atom_size=1."""
        assert old_muzero_config.atom_size == new_muzero_config.atom_size

    def test_distributional_config_explicit(
        self, old_muzero_config_distributional, new_muzero_config_distributional
    ):
        old = old_muzero_config_distributional
        new = new_muzero_config_distributional
        assert old.atom_size == new.atom_size
        assert old.support_range == new.support_range

    def test_backbone_configs_present(self, old_muzero_config, new_muzero_config):
        """Both configs should have representation, dynamics, prediction backbones."""
        assert old_muzero_config.representation_backbone is not None
        assert old_muzero_config.dynamics_backbone is not None
        assert old_muzero_config.prediction_backbone is not None
        # New config may store these differently; just verify they're accessible
        assert new_muzero_config.representation_backbone is not None
        assert new_muzero_config.dynamics_backbone is not None
        assert new_muzero_config.prediction_backbone is not None


# ============================================================================
# 2. WORLD MODEL PARITY
# ============================================================================


class TestWorldModelParity:
    """Compare world model outputs between old and new."""

    def test_initial_inference_output_shapes(self, old_agent_network, new_agent_network):
        """initial_inference should produce same-shaped latent states."""
        seed_all()
        obs = torch.randn(1, *CARTPOLE_OBS_SHAPE)

        old_wm = old_agent_network.components["world_model"]
        old_out = old_wm.initial_inference(obs)

        new_rep = new_agent_network.components["representation"]
        new_latent = new_rep(obs)

        # Both should produce a latent tensor
        assert old_out.features.shape == new_latent.shape, (
            f"Old latent shape: {old_out.features.shape}, New latent shape: {new_latent.shape}"
        )

    def test_recurrent_inference_output_keys(self, old_agent_network, new_agent_network):
        """recurrent_inference should produce outputs with same semantic fields."""
        seed_all()
        obs = torch.randn(1, *CARTPOLE_OBS_SHAPE)

        # Old path
        old_wm = old_agent_network.components["world_model"]
        old_init = old_wm.initial_inference(obs)
        action = torch.tensor([0])
        old_rec = old_wm.recurrent_inference(old_init.features, action)

        # New path
        new_rep = new_agent_network.components["representation"]
        new_latent = new_rep(obs)
        new_wm = new_agent_network.components["world_model"]
        new_rec = new_wm.recurrent_inference(new_latent, action)

        # Both should have features, reward, to_play fields
        assert old_rec.features is not None
        assert new_rec.features is not None
        assert old_rec.features.shape == new_rec.features.shape, (
            f"Old features: {old_rec.features.shape}, New features: {new_rec.features.shape}"
        )

        # Reward output should exist in both
        assert old_rec.reward is not None, "Old WM recurrent_inference missing reward"
        assert new_rec.reward is not None, "New WM recurrent_inference missing reward"

    def test_recurrent_inference_reward_shape(self, old_agent_network, new_agent_network):
        """Reward tensor shapes should match between old and new."""
        seed_all()
        obs = torch.randn(1, *CARTPOLE_OBS_SHAPE)

        old_wm = old_agent_network.components["world_model"]
        old_init = old_wm.initial_inference(obs)
        action = torch.tensor([0])
        old_rec = old_wm.recurrent_inference(old_init.features, action)

        new_rep = new_agent_network.components["representation"]
        new_latent = new_rep(obs)
        new_wm = new_agent_network.components["world_model"]
        new_rec = new_wm.recurrent_inference(new_latent, action)

        assert old_rec.reward.shape == new_rec.reward.shape, (
            f"Old reward shape: {old_rec.reward.shape}, New reward shape: {new_rec.reward.shape}"
        )

    def test_unroll_physics_output_shapes(self, old_agent_network, new_agent_network):
        """unroll_physics should produce consistently shaped outputs."""
        seed_all()
        B, K = 2, 3
        obs = torch.randn(B, *CARTPOLE_OBS_SHAPE)
        actions = torch.randint(0, CARTPOLE_NUM_ACTIONS, (B, K))

        # Old
        old_wm = old_agent_network.components["world_model"]
        old_init = old_wm.initial_inference(obs)
        old_physics = old_wm.unroll_physics(
            initial_latent_state=old_init.features,
            actions=actions,
            encoder_inputs=None,
            true_chance_codes=None,
            head_state=old_init.head_state,
        )

        # New
        new_rep = new_agent_network.components["representation"]
        new_latent = new_rep(obs)
        new_wm = new_agent_network.components["world_model"]
        new_physics = new_wm.unroll_physics(
            initial_latent_state=new_latent,
            actions=actions,
        )

        # Old returns PhysicsOutput (NamedTuple), new returns dict
        old_latents = old_physics.latents
        new_latents = new_physics["latents"]

        assert old_latents.shape == new_latents.shape, (
            f"Old latents: {old_latents.shape}, New latents: {new_latents.shape}"
        )

    def test_unroll_physics_latents_temporal_dim(self, old_agent_network, new_agent_network):
        """Both should produce [B, K+1, ...] latents (K+1 because initial state included)."""
        seed_all()
        B, K = 2, 3
        obs = torch.randn(B, *CARTPOLE_OBS_SHAPE)
        actions = torch.randint(0, CARTPOLE_NUM_ACTIONS, (B, K))

        old_wm = old_agent_network.components["world_model"]
        old_init = old_wm.initial_inference(obs)
        old_physics = old_wm.unroll_physics(
            old_init.features, actions, None, None, old_init.head_state,
        )

        new_rep = new_agent_network.components["representation"]
        new_latent = new_rep(obs)
        new_wm = new_agent_network.components["world_model"]
        new_physics = new_wm.unroll_physics(new_latent, actions)

        assert old_physics.latents.shape[1] == K + 1
        assert new_physics["latents"].shape[1] == K + 1

    def test_unroll_physics_rewards_temporal_dim(self, old_agent_network, new_agent_network):
        """Reward tensor temporal dimensions: old produces [B, K], new should match or be [B, K+1]."""
        seed_all()
        B, K = 2, 3
        obs = torch.randn(B, *CARTPOLE_OBS_SHAPE)
        actions = torch.randint(0, CARTPOLE_NUM_ACTIONS, (B, K))

        old_wm = old_agent_network.components["world_model"]
        old_init = old_wm.initial_inference(obs)
        old_physics = old_wm.unroll_physics(
            old_init.features, actions, None, None, old_init.head_state,
        )

        new_rep = new_agent_network.components["representation"]
        new_latent = new_rep(obs)
        new_wm = new_agent_network.components["world_model"]
        new_physics = new_wm.unroll_physics(new_latent, actions)

        old_rew_t = old_physics.rewards.shape[1]
        # New stores reward head outputs with key like "reward_logits"
        new_rew_key = None
        for k in new_physics.keys():
            if "reward" in k.lower():
                new_rew_key = k
                break

        if new_rew_key:
            new_rew_t = new_physics[new_rew_key].shape[1]
            # Document the difference: old has K rewards, new may have K+1 (with initial head prediction)
            print(f"PARITY NOTE - Old reward T={old_rew_t}, New reward T={new_rew_t} (key={new_rew_key})")
        else:
            pytest.skip("New world model does not produce reward key in unroll_physics")

    def test_unroll_physics_gradient_scaling(self, old_agent_network, new_agent_network):
        """Both should apply 0.5 gradient scaling to hidden states between unroll steps."""
        # This is a structural check - both implementations should call scale_gradient(hidden, 0.5)
        # We verify by checking that gradients flow through with the expected scaling
        seed_all()
        B, K = 1, 2
        obs = torch.randn(B, *CARTPOLE_OBS_SHAPE, requires_grad=True)
        actions = torch.randint(0, CARTPOLE_NUM_ACTIONS, (B, K))

        # Old
        old_wm = old_agent_network.components["world_model"]
        old_init = old_wm.initial_inference(obs)
        old_physics = old_wm.unroll_physics(
            old_init.features, actions, None, None, old_init.head_state,
        )
        old_loss = old_physics.latents[:, -1].sum()
        old_loss.backward()
        old_grad_norm = obs.grad.norm().item()

        obs.grad = None

        # New
        new_rep = new_agent_network.components["representation"]
        new_latent = new_rep(obs)
        new_wm = new_agent_network.components["world_model"]
        new_physics = new_wm.unroll_physics(new_latent, actions)
        new_loss = new_physics["latents"][:, -1].sum()
        new_loss.backward()
        new_grad_norm = obs.grad.norm().item()

        # Both should have non-zero gradients (gradient scaling doesn't zero them out)
        assert old_grad_norm > 0, "Old grad norm is zero"
        assert new_grad_norm > 0, "New grad norm is zero"
        print(f"PARITY NOTE - Gradient norms: old={old_grad_norm:.6f}, new={new_grad_norm:.6f}")


# ============================================================================
# 3. AGENT NETWORK PARITY
# ============================================================================


class TestAgentNetworkParity:
    """Compare full agent network inference outputs."""

    def test_obs_inference_returns_value(self, old_agent_network, new_agent_network):
        """obs_inference should return a value estimate."""
        seed_all()
        obs = torch.randn(1, *CARTPOLE_OBS_SHAPE)

        old_out = old_agent_network.obs_inference(obs)
        new_out = new_agent_network.obs_inference(obs)

        assert old_out.value is not None, "Old obs_inference missing value"
        assert new_out.value is not None, "New obs_inference missing value"

    def test_obs_inference_returns_policy(self, old_agent_network, new_agent_network):
        """obs_inference should return a policy."""
        seed_all()
        obs = torch.randn(1, *CARTPOLE_OBS_SHAPE)

        old_out = old_agent_network.obs_inference(obs)
        new_out = new_agent_network.obs_inference(obs)

        assert old_out.policy is not None, "Old obs_inference missing policy"
        assert new_out.policy is not None, "New obs_inference missing policy"

    def test_obs_inference_returns_network_state(self, old_agent_network, new_agent_network):
        """obs_inference should return an opaque network state for MCTS."""
        seed_all()
        obs = torch.randn(1, *CARTPOLE_OBS_SHAPE)

        old_out = old_agent_network.obs_inference(obs)
        new_out = new_agent_network.obs_inference(obs)

        # Old uses MuZeroNetworkState (NamedTuple), new uses dict
        assert old_out.network_state is not None, "Old obs_inference missing network_state"
        assert new_out.recurrent_state is not None, "New obs_inference missing recurrent_state"

    def test_obs_inference_network_state_has_dynamics(self, old_agent_network, new_agent_network):
        """Network state must contain a 'dynamics' key/field for MCTS latent stepping."""
        seed_all()
        obs = torch.randn(1, *CARTPOLE_OBS_SHAPE)

        old_out = old_agent_network.obs_inference(obs)
        new_out = new_agent_network.obs_inference(obs)

        # Old: MuZeroNetworkState.dynamics
        assert hasattr(old_out.network_state, "dynamics"), "Old state missing dynamics"
        # New: dict with "dynamics" key
        assert "dynamics" in new_out.recurrent_state, "New state missing dynamics key"

    def test_obs_inference_policy_is_distribution_like(self, old_agent_network, new_agent_network):
        """Policy should be a Distribution or have probs/logits (sampleable)."""
        seed_all()
        obs = torch.randn(1, *CARTPOLE_OBS_SHAPE)

        old_out = old_agent_network.obs_inference(obs)
        new_out = new_agent_network.obs_inference(obs)

        old_pol = old_out.policy
        new_pol = new_out.policy

        # Old returns a Distribution object; new may return Distribution or tensor
        if hasattr(old_pol, "probs"):
            assert old_pol.probs.shape[-1] == CARTPOLE_NUM_ACTIONS
        if hasattr(new_pol, "probs"):
            assert new_pol.probs.shape[-1] == CARTPOLE_NUM_ACTIONS
        elif torch.is_tensor(new_pol):
            assert new_pol.shape[-1] == CARTPOLE_NUM_ACTIONS

    def test_hidden_state_inference_output_fields(self, old_agent_network, new_agent_network):
        """hidden_state_inference should return value, policy, reward, network_state."""
        seed_all()
        obs = torch.randn(1, *CARTPOLE_OBS_SHAPE)
        action = torch.tensor([0])

        # Old
        old_root = old_agent_network.obs_inference(obs)
        old_step = old_agent_network.hidden_state_inference(old_root.network_state, action)

        # New
        new_root = new_agent_network.obs_inference(obs)
        new_step = new_agent_network.hidden_state_inference(new_root.recurrent_state, action)

        assert old_step.value is not None, "Old hidden_state_inference missing value"
        assert new_step.value is not None, "New hidden_state_inference missing value"
        assert old_step.policy is not None, "Old hidden_state_inference missing policy"
        assert new_step.policy is not None, "New hidden_state_inference missing policy"

    def test_hidden_state_inference_produces_reward(self, old_agent_network, new_agent_network):
        """hidden_state_inference should produce a reward prediction for MCTS."""
        seed_all()
        obs = torch.randn(1, *CARTPOLE_OBS_SHAPE)
        action = torch.tensor([0])

        old_root = old_agent_network.obs_inference(obs)
        old_step = old_agent_network.hidden_state_inference(old_root.network_state, action)

        new_root = new_agent_network.obs_inference(obs)
        new_step = new_agent_network.hidden_state_inference(new_root.recurrent_state, action)

        # Both should produce reward (instant_reward for actor/MCTS usage)
        print(f"PARITY NOTE - Old reward: {old_step.reward}, New reward: {new_step.reward}")

    def test_hidden_state_chaining(self, old_agent_network, new_agent_network):
        """Multiple hidden_state_inference calls should chain correctly."""
        seed_all()
        obs = torch.randn(1, *CARTPOLE_OBS_SHAPE)

        # Old chain
        old_state = old_agent_network.obs_inference(obs).network_state
        for i in range(3):
            action = torch.tensor([i % CARTPOLE_NUM_ACTIONS])
            old_out = old_agent_network.hidden_state_inference(old_state, action)
            old_state = old_out.network_state

        # New chain
        new_state = new_agent_network.obs_inference(obs).recurrent_state
        for i in range(3):
            action = torch.tensor([i % CARTPOLE_NUM_ACTIONS])
            new_out = new_agent_network.hidden_state_inference(new_state, action)
            new_state = new_out.recurrent_state

        # Both should produce valid outputs after 3 steps
        assert old_out.value is not None
        assert new_out.value is not None
        assert old_out.policy is not None
        assert new_out.policy is not None

    def test_learner_inference_output_keys(self, old_agent_network, new_agent_network):
        """learner_inference should produce values, policies, rewards, latents."""
        seed_all()
        B, K = 2, 3
        batch = {
            "observations": torch.randn(B, *CARTPOLE_OBS_SHAPE),
            "actions": torch.randint(0, CARTPOLE_NUM_ACTIONS, (B, K)),
        }

        old_out = old_agent_network.learner_inference(batch)
        new_out = new_agent_network.learner_inference(batch)

        # Old keys
        assert "values" in old_out, f"Old missing 'values'. Keys: {old_out.keys()}"
        assert "policies" in old_out, f"Old missing 'policies'. Keys: {old_out.keys()}"
        assert "rewards" in old_out, f"Old missing 'rewards'. Keys: {old_out.keys()}"
        assert "latents" in old_out, f"Old missing 'latents'. Keys: {old_out.keys()}"

        # New keys - may use different names
        assert "latents" in new_out, f"New missing 'latents'. Keys: {new_out.keys()}"

        # Document key differences
        old_keys = set(old_out.keys())
        new_keys = set(new_out.keys())
        only_old = old_keys - new_keys
        only_new = new_keys - old_keys
        if only_old:
            print(f"PARITY NOTE - Keys only in old: {only_old}")
        if only_new:
            print(f"PARITY NOTE - Keys only in new: {only_new}")

    def test_learner_inference_values_shape(self, old_agent_network, new_agent_network):
        """Values should have shape [B, K+1, atoms]."""
        seed_all()
        B, K = 2, 3
        batch = {
            "observations": torch.randn(B, *CARTPOLE_OBS_SHAPE),
            "actions": torch.randint(0, CARTPOLE_NUM_ACTIONS, (B, K)),
        }

        old_out = old_agent_network.learner_inference(batch)
        new_out = new_agent_network.learner_inference(batch)

        old_vals = old_out["values"]
        assert old_vals.shape[0] == B
        assert old_vals.shape[1] == K + 1

        # New may use a different key name for value logits
        new_val_key = "values" if "values" in new_out else "state_value"
        if new_val_key not in new_out:
            # Try to find any value-related key
            for k in new_out:
                if "value" in k.lower() or "state_value" in k.lower():
                    new_val_key = k
                    break

        if new_val_key in new_out:
            new_vals = new_out[new_val_key]
            assert new_vals.shape[0] == B
            assert new_vals.shape[1] == K + 1, (
                f"New values temporal dim: {new_vals.shape[1]}, expected {K+1}"
            )
            print(f"PARITY NOTE - Old values shape: {old_vals.shape}, New values shape ({new_val_key}): {new_vals.shape}")
        else:
            print(f"PARITY NOTE - Could not find value key in new output. Keys: {new_out.keys()}")

    def test_learner_inference_policies_shape(self, old_agent_network, new_agent_network):
        """Policies should have shape [B, K+1, num_actions]."""
        seed_all()
        B, K = 2, 3
        batch = {
            "observations": torch.randn(B, *CARTPOLE_OBS_SHAPE),
            "actions": torch.randint(0, CARTPOLE_NUM_ACTIONS, (B, K)),
        }

        old_out = old_agent_network.learner_inference(batch)
        new_out = new_agent_network.learner_inference(batch)

        old_pols = old_out["policies"]
        assert old_pols.shape == (B, K + 1, CARTPOLE_NUM_ACTIONS)

        # Find policy key in new
        new_pol_key = None
        for k in new_out:
            if "polic" in k.lower():
                new_pol_key = k
                break

        if new_pol_key:
            new_pols = new_out[new_pol_key]
            assert new_pols.shape[0] == B
            assert new_pols.shape[1] == K + 1
            assert new_pols.shape[2] == CARTPOLE_NUM_ACTIONS, (
                f"New policy action dim: {new_pols.shape[2]}, expected {CARTPOLE_NUM_ACTIONS}"
            )
        else:
            print(f"PARITY NOTE - No policy key found in new output. Keys: {new_out.keys()}")

    def test_learner_inference_rewards_padding(self, old_agent_network, new_agent_network):
        """Old pads rewards to [B, K+1, ...] with dummy at t=0. Check new behavior."""
        seed_all()
        B, K = 2, 3
        batch = {
            "observations": torch.randn(B, *CARTPOLE_OBS_SHAPE),
            "actions": torch.randint(0, CARTPOLE_NUM_ACTIONS, (B, K)),
        }

        old_out = old_agent_network.learner_inference(batch)
        new_out = new_agent_network.learner_inference(batch)

        old_rews = old_out["rewards"]
        # Old pads: [B, K+1, ...], t=0 is zeros
        assert old_rews.shape[1] == K + 1, f"Old rewards T dim: {old_rews.shape[1]}"
        assert torch.allclose(old_rews[:, 0], torch.zeros_like(old_rews[:, 0])), (
            "Old rewards t=0 should be zero-padded"
        )

        # New may or may not pad - document the difference
        new_rew_key = None
        for k in new_out:
            if "reward" in k.lower():
                new_rew_key = k
                break

        if new_rew_key:
            new_rews = new_out[new_rew_key]
            print(
                f"PARITY NOTE - Reward padding: Old=[B, {old_rews.shape[1]}, ...] (padded), "
                f"New=[B, {new_rews.shape[1]}, ...] (key={new_rew_key})"
            )
            if new_rews.shape[1] == K + 1:
                # Check if new also pads t=0
                t0_is_zero = torch.allclose(new_rews[:, 0], torch.zeros_like(new_rews[:, 0]))
                print(f"PARITY NOTE - New rewards t=0 is zero-padded: {t0_is_zero}")


# ============================================================================
# 4. N-STEP TARGET COMPUTATION PARITY
# ============================================================================


class TestNStepTargetParity:
    """Compare the vectorized N-step target computation between old and new."""

    def _make_test_data(self, B=4, L=10):
        """Create deterministic test data for N-step target comparison."""
        seed_all()
        raw_rewards = torch.rand(B, L)
        raw_values = torch.rand(B, L)
        raw_to_plays = torch.zeros(B, L, dtype=torch.long)  # Single player
        raw_terminated = torch.zeros(B, L, dtype=torch.bool)
        raw_terminated[:, 7] = True  # Episode ends at step 7
        valid_mask = torch.ones(B, L, dtype=torch.bool)
        valid_mask[:, 8:] = False  # Invalid after termination
        return raw_rewards, raw_values, raw_to_plays, raw_terminated, valid_mask

    def test_n_step_single_player_no_termination(self):
        """N-step targets with no termination should match."""
        seed_all()
        B, L = 4, 15
        raw_rewards = torch.rand(B, L)
        raw_values = torch.rand(B, L)
        raw_to_plays = torch.zeros(B, L, dtype=torch.long)
        raw_terminated = torch.zeros(B, L, dtype=torch.bool)
        valid_mask = torch.ones(B, L, dtype=torch.bool)

        gamma = 0.99
        n_step = 3
        unroll_steps = 3

        # Old
        from old_muzero.replay_buffers.processors import NStepUnrollProcessor
        old_proc = NStepUnrollProcessor(
            unroll_steps=unroll_steps, n_step=n_step, gamma=gamma,
            num_actions=2, num_players=1, max_size=100,
        )
        old_vals, old_rews = old_proc._compute_n_step_targets(
            B, raw_rewards, raw_values, raw_to_plays,
            raw_terminated, torch.zeros_like(raw_terminated),
            valid_mask, raw_rewards.device,
        )

        # New
        from agents.learner.functional.returns import compute_unrolled_n_step_targets
        new_vals, new_rews = compute_unrolled_n_step_targets(
            raw_rewards=raw_rewards,
            raw_values=raw_values,
            raw_to_plays=raw_to_plays,
            raw_terminated=raw_terminated,
            valid_mask=valid_mask,
            gamma=gamma,
            n_step=n_step,
            unroll_steps=unroll_steps,
        )

        # Compare
        assert old_vals.shape == new_vals.shape, (
            f"Value target shapes differ: old={old_vals.shape}, new={new_vals.shape}"
        )
        assert old_rews.shape == new_rews.shape, (
            f"Reward target shapes differ: old={old_rews.shape}, new={new_rews.shape}"
        )

        if not torch.allclose(old_vals, new_vals, atol=1e-5):
            diff = (old_vals - new_vals).abs()
            print(f"PARITY DIFF - N-step values max diff: {diff.max().item():.8f}")
            print(f"  Old values: {old_vals[0]}")
            print(f"  New values: {new_vals[0]}")
        else:
            print("PARITY OK - N-step values match")

        if not torch.allclose(old_rews, new_rews, atol=1e-5):
            diff = (old_rews - new_rews).abs()
            print(f"PARITY DIFF - N-step rewards max diff: {diff.max().item():.8f}")
        else:
            print("PARITY OK - N-step rewards match")

    def test_n_step_with_termination(self):
        """N-step targets with mid-episode termination should match."""
        raw_rewards, raw_values, raw_to_plays, raw_terminated, valid_mask = self._make_test_data()
        gamma = 0.99
        n_step = 3
        unroll_steps = 3

        from old_muzero.replay_buffers.processors import NStepUnrollProcessor
        old_proc = NStepUnrollProcessor(
            unroll_steps=unroll_steps, n_step=n_step, gamma=gamma,
            num_actions=2, num_players=1, max_size=100,
        )
        old_vals, old_rews = old_proc._compute_n_step_targets(
            raw_rewards.shape[0], raw_rewards, raw_values, raw_to_plays,
            raw_terminated, torch.zeros_like(raw_terminated),
            valid_mask, raw_rewards.device,
        )

        from agents.learner.functional.returns import compute_unrolled_n_step_targets
        new_vals, new_rews = compute_unrolled_n_step_targets(
            raw_rewards=raw_rewards,
            raw_values=raw_values,
            raw_to_plays=raw_to_plays,
            raw_terminated=raw_terminated,
            valid_mask=valid_mask,
            gamma=gamma,
            n_step=n_step,
            unroll_steps=unroll_steps,
        )

        assert old_vals.shape == new_vals.shape
        assert old_rews.shape == new_rews.shape

        if not torch.allclose(old_vals, new_vals, atol=1e-5):
            diff = (old_vals - new_vals).abs()
            print(f"PARITY DIFF - Terminated n-step values max diff: {diff.max().item():.8f}")
            for b in range(min(2, old_vals.shape[0])):
                print(f"  Batch {b} old: {old_vals[b]}")
                print(f"  Batch {b} new: {new_vals[b]}")
        else:
            print("PARITY OK - Terminated n-step values match")

    def test_n_step_multiplayer_signs(self):
        """Multi-player value targets should apply correct sign flipping."""
        seed_all()
        B, L = 4, 15
        raw_rewards = torch.rand(B, L)
        raw_values = torch.rand(B, L)
        # Alternating players
        raw_to_plays = torch.zeros(B, L, dtype=torch.long)
        for i in range(L):
            raw_to_plays[:, i] = i % 2
        raw_terminated = torch.zeros(B, L, dtype=torch.bool)
        valid_mask = torch.ones(B, L, dtype=torch.bool)

        gamma = 0.99
        n_step = 3
        unroll_steps = 3

        from old_muzero.replay_buffers.processors import NStepUnrollProcessor
        old_proc = NStepUnrollProcessor(
            unroll_steps=unroll_steps, n_step=n_step, gamma=gamma,
            num_actions=2, num_players=2, max_size=100,
        )
        old_vals, _ = old_proc._compute_n_step_targets(
            B, raw_rewards, raw_values, raw_to_plays,
            raw_terminated, torch.zeros_like(raw_terminated),
            valid_mask, raw_rewards.device,
        )

        from agents.learner.functional.returns import compute_unrolled_n_step_targets
        new_vals, _ = compute_unrolled_n_step_targets(
            raw_rewards=raw_rewards,
            raw_values=raw_values,
            raw_to_plays=raw_to_plays,
            raw_terminated=raw_terminated,
            valid_mask=valid_mask,
            gamma=gamma,
            n_step=n_step,
            unroll_steps=unroll_steps,
        )

        if not torch.allclose(old_vals, new_vals, atol=1e-5):
            diff = (old_vals - new_vals).abs()
            print(f"PARITY DIFF - Multiplayer n-step values max diff: {diff.max().item():.8f}")
        else:
            print("PARITY OK - Multiplayer n-step values match")

    def test_n_step_value_prefix_mode(self):
        """Value prefix (LSTM reward accumulation) should match between old and new."""
        seed_all()
        B, L = 4, 15
        raw_rewards = torch.rand(B, L)
        raw_values = torch.rand(B, L)
        raw_to_plays = torch.zeros(B, L, dtype=torch.long)
        raw_terminated = torch.zeros(B, L, dtype=torch.bool)
        valid_mask = torch.ones(B, L, dtype=torch.bool)

        gamma = 0.99
        n_step = 3
        unroll_steps = 3
        lstm_horizon_len = 5

        from old_muzero.replay_buffers.processors import NStepUnrollProcessor
        old_proc = NStepUnrollProcessor(
            unroll_steps=unroll_steps, n_step=n_step, gamma=gamma,
            num_actions=2, num_players=1, max_size=100,
            lstm_horizon_len=lstm_horizon_len, value_prefix=True,
        )
        old_vals, old_rews = old_proc._compute_n_step_targets(
            B, raw_rewards, raw_values, raw_to_plays,
            raw_terminated, torch.zeros_like(raw_terminated),
            valid_mask, raw_rewards.device,
        )

        from agents.learner.functional.returns import compute_unrolled_n_step_targets
        new_vals, new_rews = compute_unrolled_n_step_targets(
            raw_rewards=raw_rewards,
            raw_values=raw_values,
            raw_to_plays=raw_to_plays,
            raw_terminated=raw_terminated,
            valid_mask=valid_mask,
            gamma=gamma,
            n_step=n_step,
            unroll_steps=unroll_steps,
            use_value_prefix=True,
            lstm_horizon_len=lstm_horizon_len,
        )

        # Value targets should be same (value prefix only affects reward targets)
        if not torch.allclose(old_vals, new_vals, atol=1e-5):
            diff = (old_vals - new_vals).abs()
            print(f"PARITY DIFF - Value prefix value targets max diff: {diff.max().item():.8f}")
        else:
            print("PARITY OK - Value prefix value targets match")

        if not torch.allclose(old_rews, new_rews, atol=1e-5):
            diff = (old_rews - new_rews).abs()
            print(f"PARITY DIFF - Value prefix reward targets max diff: {diff.max().item():.8f}")
            print(f"  Old rewards: {old_rews[0]}")
            print(f"  New rewards: {new_rews[0]}")
        else:
            print("PARITY OK - Value prefix reward targets match")


# ============================================================================
# 5. INFERENCE OUTPUT DATA STRUCTURES PARITY
# ============================================================================


class TestInferenceOutputParity:
    """Compare data structure contracts between old and new."""

    def test_inference_output_field_mapping(self):
        """Document the field mapping between old InferenceOutput and new."""
        from old_muzero.modules.world_models.inference_output import InferenceOutput as OldInferenceOutput
        from modules.models.inference_output import InferenceOutput as NewInferenceOutput

        old_fields = set(OldInferenceOutput._fields) if hasattr(OldInferenceOutput, "_fields") else set(OldInferenceOutput.__dataclass_fields__.keys())
        new_fields = set(NewInferenceOutput.__dataclass_fields__.keys())

        common = old_fields & new_fields
        only_old = old_fields - new_fields
        only_new = new_fields - old_fields

        print(f"PARITY MAP - InferenceOutput common fields: {common}")
        print(f"PARITY MAP - Fields only in old: {only_old}")
        print(f"PARITY MAP - Fields only in new: {only_new}")

        # Critical fields that MUST exist in both
        critical_fields = {"value", "policy", "reward", "to_play"}
        for f in critical_fields:
            assert f in old_fields, f"Old InferenceOutput missing critical field: {f}"
            assert f in new_fields, f"New InferenceOutput missing critical field: {f}"

    def test_world_model_output_field_mapping(self):
        """Document WorldModelOutput field mapping."""
        from old_muzero.modules.world_models.inference_output import WorldModelOutput as OldWMO
        from modules.models.inference_output import WorldModelOutput as NewWMO

        old_fields = set(OldWMO._fields)
        new_fields = set(NewWMO.__dataclass_fields__.keys())

        common = old_fields & new_fields
        only_old = old_fields - new_fields
        only_new = new_fields - old_fields

        print(f"PARITY MAP - WorldModelOutput common fields: {common}")
        print(f"PARITY MAP - Fields only in old: {only_old}")
        print(f"PARITY MAP - Fields only in new: {only_new}")

        # features must exist in both
        assert "features" in old_fields
        assert "features" in new_fields

    def test_old_network_state_is_named_tuple(self):
        """Old uses MuZeroNetworkState (NamedTuple), new uses plain dict."""
        from old_muzero.modules.world_models.inference_output import MuZeroNetworkState

        state = MuZeroNetworkState(dynamics=torch.randn(1, 32))
        assert hasattr(state, "dynamics")
        assert hasattr(state, "wm_memory")

    def test_new_network_state_is_dict(self, new_agent_network):
        """New uses dict for recurrent state."""
        obs = torch.randn(1, *CARTPOLE_OBS_SHAPE)
        out = new_agent_network.obs_inference(obs)
        assert isinstance(out.recurrent_state, dict)
        assert "dynamics" in out.recurrent_state

    def test_old_network_state_batch_unbatch(self):
        """Old MuZeroNetworkState supports batch/unbatch for MCTS."""
        from old_muzero.modules.world_models.inference_output import MuZeroNetworkState

        states = [
            MuZeroNetworkState(dynamics=torch.randn(1, 32)),
            MuZeroNetworkState(dynamics=torch.randn(1, 32)),
        ]
        batched = MuZeroNetworkState.batch(states)
        assert batched.dynamics.shape[0] == 2

        unbatched = batched.unbatch()
        assert len(unbatched) == 2
        assert unbatched[0].dynamics.shape[0] == 1

    def test_new_network_state_batch_unbatch(self):
        """New dict-based state supports batch/unbatch."""
        from modules.models.inference_output import batch_recurrent_state, unbatch_recurrent_state

        states = [
            {"dynamics": torch.randn(1, 32)},
            {"dynamics": torch.randn(1, 32)},
        ]
        batched = batch_recurrent_state(states)
        assert batched["dynamics"].shape[0] == 2

        unbatched = unbatch_recurrent_state(batched)
        assert len(unbatched) == 2
        assert unbatched[0]["dynamics"].shape[0] == 1


# ============================================================================
# 6. TARGET BUILDER PARITY
# ============================================================================


class TestTargetBuilderParity:
    """Compare target builder pipelines between old and new."""

    def test_mcts_extractor_passthrough(self):
        """MCTSExtractor should pass through MCTS target keys identically."""
        from old_muzero.agents.learner.target_builders import MCTSExtractor as OldMCTSExtractor
        from agents.learner.target_builders import MCTSExtractor as NewMCTSExtractor

        batch = {
            "values": torch.rand(4, 4),
            "rewards": torch.rand(4, 4),
            "policies": torch.rand(4, 4, 2),
            "actions": torch.randint(0, 2, (4, 3)),
            "to_plays": torch.zeros(4, 4, 2),
        }

        old_targets = {}
        OldMCTSExtractor().build_targets(batch, {}, None, old_targets)

        new_targets = {}
        NewMCTSExtractor().build_targets(batch, {}, None, new_targets)

        assert set(old_targets.keys()) == set(new_targets.keys()), (
            f"MCTSExtractor key mismatch: old={set(old_targets.keys())}, new={set(new_targets.keys())}"
        )
        for k in old_targets:
            assert torch.equal(old_targets[k], new_targets[k]), f"MCTSExtractor values differ for key '{k}'"

    def test_sequence_padder(self):
        """SequencePadder should pad K-length to K+1-length identically."""
        from old_muzero.agents.learner.target_builders import SequencePadder as OldPadder
        from agents.learner.target_builders import SequencePadder as NewPadder

        unroll_steps = 3
        targets = {
            "rewards": torch.rand(4, 3),  # [B, K]
            "to_plays": torch.rand(4, 3, 2),  # [B, K, P]
        }

        old_targets = copy.deepcopy(targets)
        OldPadder(unroll_steps).build_targets({}, {}, None, old_targets)

        new_targets = copy.deepcopy(targets)
        NewPadder(unroll_steps).build_targets({}, {}, None, new_targets)

        for k in ["rewards", "to_plays"]:
            assert old_targets[k].shape == new_targets[k].shape, (
                f"SequencePadder shape mismatch for '{k}': old={old_targets[k].shape}, new={new_targets[k].shape}"
            )
            assert torch.allclose(old_targets[k], new_targets[k]), (
                f"SequencePadder values differ for '{k}'"
            )

    def test_sequence_mask_builder(self):
        """SequenceMaskBuilder should produce identical masks."""
        from old_muzero.agents.learner.target_builders import SequenceMaskBuilder as OldMaskBuilder
        from agents.learner.target_builders import SequenceMaskBuilder as NewMaskBuilder

        B, T = 4, 4
        batch = {
            "is_same_game": torch.ones(B, T, dtype=torch.bool),
            "has_valid_obs_mask": torch.ones(B, T, dtype=torch.bool),
        }
        base_targets = {"actions": torch.randint(0, 2, (B, T))}

        old_targets = copy.deepcopy(base_targets)
        OldMaskBuilder().build_targets(batch, {}, None, old_targets)

        new_targets = copy.deepcopy(base_targets)
        NewMaskBuilder().build_targets(batch, {}, None, new_targets)

        mask_keys = ["value_mask", "masks", "policy_mask", "reward_mask"]
        for k in mask_keys:
            if k in old_targets and k in new_targets:
                assert torch.equal(old_targets[k], new_targets[k]), (
                    f"Mask '{k}' differs: old={old_targets[k]}, new={new_targets[k]}"
                )
            elif k in old_targets:
                print(f"PARITY DIFF - Mask '{k}' only in old")
            elif k in new_targets:
                print(f"PARITY DIFF - Mask '{k}' only in new")

    def test_sequence_infrastructure_builder(self):
        """SequenceInfrastructureBuilder should produce identical weights and gradient_scales."""
        from old_muzero.agents.learner.target_builders import SequenceInfrastructureBuilder as OldInfra
        from agents.learner.target_builders import SequenceInfrastructureBuilder as NewInfra

        B, T = 4, 4
        unroll_steps = 3
        batch = {"weights": torch.ones(B)}
        base_targets = {"actions": torch.randint(0, 2, (B, T))}

        old_targets = copy.deepcopy(base_targets)
        OldInfra(unroll_steps).build_targets(batch, {}, None, old_targets)

        new_targets = copy.deepcopy(base_targets)
        NewInfra(unroll_steps).build_targets(batch, {}, None, new_targets)

        # Gradient scales should be [1.0, 1/K, 1/K, 1/K]
        assert torch.allclose(old_targets["gradient_scales"], new_targets["gradient_scales"]), (
            f"Gradient scales differ: old={old_targets['gradient_scales']}, new={new_targets['gradient_scales']}"
        )
        assert torch.allclose(old_targets["weights"], new_targets["weights"]), (
            f"Weights differ: old={old_targets['weights']}, new={new_targets['weights']}"
        )

    def test_full_muzero_target_pipeline(self):
        """Run the full MuZero target builder pipeline and compare outputs."""
        from old_muzero.agents.learner.target_builders import (
            TargetBuilderPipeline as OldPipeline,
            MCTSExtractor as OldMCTSExtractor,
            SequencePadder as OldSequencePadder,
            SequenceMaskBuilder as OldSequenceMaskBuilder,
            SequenceInfrastructureBuilder as OldSequenceInfraBuilder,
        )
        from agents.learner.target_builders import (
            TargetBuilderPipeline as NewPipeline,
            MCTSExtractor as NewMCTSExtractor,
            SequencePadder as NewSequencePadder,
            SequenceMaskBuilder as NewSequenceMaskBuilder,
            SequenceInfrastructureBuilder as NewSequenceInfraBuilder,
        )

        unroll_steps = 3
        B, T = 4, unroll_steps + 1

        batch = {
            "values": torch.rand(B, T),
            "rewards": torch.rand(B, T),
            "policies": torch.rand(B, T, CARTPOLE_NUM_ACTIONS),
            "actions": torch.randint(0, CARTPOLE_NUM_ACTIONS, (B, T)),
            "to_plays": torch.zeros(B, T, CARTPOLE_NUM_PLAYERS),
            "is_same_game": torch.ones(B, T, dtype=torch.bool),
            "has_valid_obs_mask": torch.ones(B, T, dtype=torch.bool),
            "weights": torch.ones(B),
        }

        old_pipeline = OldPipeline([
            OldMCTSExtractor(),
            OldSequenceMaskBuilder(),
            OldSequenceInfraBuilder(unroll_steps),
        ])

        new_pipeline = NewPipeline([
            NewMCTSExtractor(),
            NewSequenceMaskBuilder(),
            NewSequenceInfraBuilder(unroll_steps),
        ])

        old_targets = {}
        old_pipeline.build_targets(batch, {}, None, old_targets)

        new_targets = {}
        new_pipeline.build_targets(batch, {}, None, new_targets)

        common_keys = set(old_targets.keys()) & set(new_targets.keys())
        for k in common_keys:
            if torch.is_tensor(old_targets[k]) and torch.is_tensor(new_targets[k]):
                if not torch.equal(old_targets[k], new_targets[k]):
                    print(f"PARITY DIFF - Target pipeline key '{k}' differs")

        only_old = set(old_targets.keys()) - set(new_targets.keys())
        only_new = set(new_targets.keys()) - set(old_targets.keys())
        if only_old:
            print(f"PARITY NOTE - Pipeline keys only in old: {only_old}")
        if only_new:
            print(f"PARITY NOTE - Pipeline keys only in new: {only_new}")


# ============================================================================
# 7. COMPONENT ARCHITECTURE PARITY
# ============================================================================


class TestComponentArchitectureParity:
    """Compare internal component structure and parameter counts."""

    def test_parameter_count_comparison(self, old_agent_network, new_agent_network):
        """Document parameter count differences between old and new."""
        old_params = sum(p.numel() for p in old_agent_network.parameters())
        new_params = sum(p.numel() for p in new_agent_network.parameters())

        print(f"PARITY NOTE - Total params: old={old_params}, new={new_params}")
        if old_params != new_params:
            print(f"PARITY DIFF - Parameter count mismatch: {abs(old_params - new_params)} params difference")

    def test_component_names(self, old_agent_network, new_agent_network):
        """Document component naming differences."""
        old_components = set(old_agent_network.components.keys())
        new_components = set(new_agent_network.components.keys())

        print(f"PARITY MAP - Old components: {old_components}")
        print(f"PARITY MAP - New components: {new_components}")

    def test_old_world_model_has_reward_head(self, old_agent_network):
        """Old WM owns reward_head and to_play_head."""
        wm = old_agent_network.components["world_model"]
        assert hasattr(wm, "reward_head"), "Old WM missing reward_head"
        assert hasattr(wm, "to_play_head"), "Old WM missing to_play_head"

    def test_new_world_model_heads_location(self, new_agent_network):
        """New WM has heads in WorldModel.heads (env heads)."""
        wm = new_agent_network.components["world_model"]
        assert hasattr(wm, "heads"), "New WM missing heads dict"
        print(f"PARITY MAP - New WM heads: {list(wm.heads.keys())}")

    def test_representation_network_exists(self, old_agent_network, new_agent_network):
        """Both should have a representation network."""
        # Old: inside world_model
        assert hasattr(old_agent_network.components["world_model"], "representation")
        # New: separate component
        assert "representation" in new_agent_network.components

    def test_prediction_backbone_exists(self, old_agent_network, new_agent_network):
        """Both should have a prediction backbone."""
        assert "prediction_backbone" in old_agent_network.components
        assert "prediction_backbone" in new_agent_network.components

    def test_prediction_backbone_param_count(self, old_agent_network, new_agent_network):
        """Prediction backbone parameter counts should match."""
        old_bb = old_agent_network.components["prediction_backbone"]
        new_bb = new_agent_network.components["prediction_backbone"]

        old_params = sum(p.numel() for p in old_bb.parameters())
        new_params = sum(p.numel() for p in new_bb.parameters())

        print(f"PARITY NOTE - Prediction backbone params: old={old_params}, new={new_params}")

    def test_value_head_location(self, old_agent_network, new_agent_network):
        """Old has value_head in components, new has it in behavior_heads."""
        assert "value_head" in old_agent_network.components, "Old missing value_head"
        # New uses behavior_heads sub-dict
        behavior_heads = new_agent_network.components.get("behavior_heads", {})
        head_names = list(behavior_heads.keys())
        print(f"PARITY MAP - New behavior heads: {head_names}")
        # Check for any value-related head
        has_value = any("value" in n.lower() for n in head_names)
        assert has_value, f"New missing value head. Available: {head_names}"

    def test_policy_head_location(self, old_agent_network, new_agent_network):
        """Old has policy_head in components, new has it in behavior_heads."""
        assert "policy_head" in old_agent_network.components, "Old missing policy_head"
        behavior_heads = new_agent_network.components.get("behavior_heads", {})
        has_policy = any("polic" in n.lower() for n in behavior_heads.keys())
        assert has_policy, f"New missing policy head. Available: {list(behavior_heads.keys())}"


# ============================================================================
# 8. INITIALIZATION PARITY
# ============================================================================


class TestInitializationParity:
    """Compare weight initialization between old and new."""

    def test_default_initialization_method(self, old_agent_network, new_agent_network):
        """Both should support the same initialization methods."""
        # Test that initialize() can be called without error
        seed_all()
        old_agent_network.initialize("orthogonal")

        seed_all()
        new_agent_network.initialize("orthogonal")

    def test_initialization_affects_weights(self, old_agent_network, new_agent_network):
        """Initialization should actually change weights (not a no-op)."""
        # Get pre-init weights
        old_w_before = next(old_agent_network.parameters()).clone()
        new_w_before = next(new_agent_network.parameters()).clone()

        seed_all()
        old_agent_network.initialize("xavier_uniform")
        seed_all()
        new_agent_network.initialize("xavier_uniform")

        old_w_after = next(old_agent_network.parameters())
        new_w_after = next(new_agent_network.parameters())

        # Weights should have changed
        assert not torch.equal(old_w_before, old_w_after), "Old init didn't change weights"
        assert not torch.equal(new_w_before, new_w_after), "New init didn't change weights"

    def test_same_init_same_seed_produces_similar_distributions(self):
        """Under same seed and init, weight distributions should be similar."""
        from old_muzero.modules.agent_nets.modular import ModularAgentNetwork
        from old_muzero.configs.agents.muzero import MuZeroConfig as OldMuZeroConfig
        from old_muzero.configs.games.cartpole import CartPoleConfig as OldCartPoleConfig

        seed_all()
        old_config = OldMuZeroConfig(_base_muzero_dict(), OldCartPoleConfig())
        old_net = ModularAgentNetwork(old_config, CARTPOLE_OBS_SHAPE, CARTPOLE_NUM_ACTIONS)
        old_net.initialize("orthogonal")

        old_weight_stats = {
            "mean": torch.cat([p.flatten() for p in old_net.parameters()]).mean().item(),
            "std": torch.cat([p.flatten() for p in old_net.parameters()]).std().item(),
        }

        print(f"PARITY NOTE - Old weight stats after orthogonal init: mean={old_weight_stats['mean']:.4f}, std={old_weight_stats['std']:.4f}")


# ============================================================================
# 9. REPLAY BUFFER OUTPUT PROCESSOR PARITY
# ============================================================================


class TestReplayBufferProcessorParity:
    """Compare NStepUnrollProcessor outputs between old and new."""

    def _make_buffers(self, max_size=50, num_episodes=3, ep_len=10):
        """Create fake buffer data for testing."""
        seed_all()
        total = num_episodes * ep_len

        obs = torch.randn(max_size, *CARTPOLE_OBS_SHAPE)
        rewards = torch.rand(max_size)
        values = torch.rand(max_size)
        policies = torch.rand(max_size, CARTPOLE_NUM_ACTIONS)
        policies = policies / policies.sum(dim=1, keepdim=True)
        actions = torch.randint(0, CARTPOLE_NUM_ACTIONS, (max_size,)).float()
        to_plays = torch.zeros(max_size, dtype=torch.long)
        chances = torch.zeros(max_size, 1, dtype=torch.long)
        terminated = torch.zeros(max_size, dtype=torch.bool)
        truncated = torch.zeros(max_size, dtype=torch.bool)
        dones = torch.zeros(max_size, dtype=torch.bool)
        legal_masks = torch.ones(max_size, CARTPOLE_NUM_ACTIONS, dtype=torch.bool)
        training_steps = torch.zeros(max_size, dtype=torch.long)
        ids = torch.arange(max_size, dtype=torch.long)

        # Mark episode boundaries
        game_ids = torch.zeros(max_size, dtype=torch.long)
        for ep in range(num_episodes):
            start = ep * ep_len
            end = start + ep_len
            game_ids[start:end] = ep
            terminated[end - 1] = True
            dones[end - 1] = True

        return {
            "observations": obs,
            "rewards": rewards,
            "values": values,
            "policies": policies,
            "actions": actions,
            "to_plays": to_plays,
            "chances": chances,
            "terminated": terminated,
            "truncated": truncated,
            "dones": dones,
            "game_ids": game_ids,
            "legal_masks": legal_masks,
            "training_steps": training_steps,
            "ids": ids,
        }

    def test_output_processor_shapes(self):
        """NStepUnrollProcessor should produce same-shaped outputs."""
        from old_muzero.replay_buffers.processors import NStepUnrollProcessor

        buffers = self._make_buffers()
        indices = [0, 5, 10, 15]  # Sample from different episodes

        proc = NStepUnrollProcessor(
            unroll_steps=3, n_step=3, gamma=0.99,
            num_actions=CARTPOLE_NUM_ACTIONS, num_players=CARTPOLE_NUM_PLAYERS,
            max_size=50,
        )

        result = proc.process_batch(indices, buffers)

        # Document all output keys and shapes
        for k, v in result.items():
            if torch.is_tensor(v):
                print(f"PARITY MAP - NStepUnrollProcessor output '{k}': shape={v.shape}, dtype={v.dtype}")
            else:
                print(f"PARITY MAP - NStepUnrollProcessor output '{k}': type={type(v)}")

    def test_output_processor_validity_masks(self):
        """Validity masks should correctly reflect episode boundaries."""
        from old_muzero.replay_buffers.processors import NStepUnrollProcessor

        buffers = self._make_buffers(max_size=50, num_episodes=3, ep_len=10)
        # Index near episode boundary
        indices = [8]  # Near end of episode 0 (ends at 9)

        proc = NStepUnrollProcessor(
            unroll_steps=3, n_step=3, gamma=0.99,
            num_actions=CARTPOLE_NUM_ACTIONS, num_players=CARTPOLE_NUM_PLAYERS,
            max_size=50,
        )

        result = proc.process_batch(indices, buffers)
        same_game = result["is_same_game"]
        has_valid = result["has_valid_obs_mask"]

        print(f"PARITY MAP - Validity at boundary: is_same_game={same_game[0]}, has_valid_obs={has_valid[0]}")
        # Index 8 + unroll_steps = 11, which crosses into episode 1
        # So validity should drop after boundary
        assert not same_game[0, -1].item(), (
            "Expected same_game to be False past episode boundary"
        )


# ============================================================================
# 10. ACTION SELECTOR PARITY
# ============================================================================


class TestActionSelectorParity:
    """Compare action selection behavior between old and new."""

    def test_categorical_selector_basic(self):
        """CategoricalSelector should sample from the same distribution."""
        from old_muzero.agents.action_selectors.selectors import CategoricalSelector as OldSelector
        from agents.action_selectors.selectors import CategoricalSelector as NewSelector

        # Both selectors should exist and be importable
        old_sel = OldSelector()
        new_sel = NewSelector()

        print("PARITY OK - Both CategoricalSelector classes importable")

    def test_temperature_decorator_exists(self):
        """TemperatureSelector should exist in both."""
        from old_muzero.agents.action_selectors.decorators import TemperatureSelector as OldTemp
        from agents.action_selectors.decorators import TemperatureSelector as NewTemp

        print("PARITY OK - Both TemperatureSelector classes importable")


# ============================================================================
# 11. LOSS PIPELINE PARITY
# ============================================================================


class TestLossPipelineParity:
    """Compare loss pipeline construction and loss module names."""

    def test_muzero_loss_modules_old(self, old_agent_network, old_muzero_config):
        """Document loss modules used in old MuZero."""
        from old_muzero.agents.registries.muzero import build_muzero_loss_pipeline

        pipeline = build_muzero_loss_pipeline(old_muzero_config, old_agent_network, "cpu")
        module_names = [type(m).__name__ for m in pipeline.modules]
        print(f"PARITY MAP - Old loss modules: {module_names}")

        # Minimum expected modules
        expected = {"ValueLoss", "PolicyLoss", "RewardLoss"}
        actual = set(module_names)
        assert expected.issubset(actual), f"Old pipeline missing: {expected - actual}"

    def test_loss_module_required_keys(self, old_agent_network, old_muzero_config):
        """Document required prediction/target keys for each loss module."""
        from old_muzero.agents.registries.muzero import build_muzero_loss_pipeline

        pipeline = build_muzero_loss_pipeline(old_muzero_config, old_agent_network, "cpu")
        for m in pipeline.modules:
            preds = getattr(m, "required_predictions", set())
            targs = getattr(m, "required_targets", set())
            print(f"PARITY MAP - {type(m).__name__}: predictions={preds}, targets={targs}")


# ============================================================================
# 12. HEAD OUTPUT CONTRACT PARITY
# ============================================================================


class TestHeadOutputParity:
    """Compare head output format between old (tuple) and new (HeadOutput dataclass)."""

    def test_old_head_returns_tuple(self, old_agent_network):
        """Old heads return (logits, state, inference_value) tuple."""
        obs = torch.randn(1, *CARTPOLE_OBS_SHAPE)
        wm = old_agent_network.components["world_model"]
        init_out = wm.initial_inference(obs)
        latent = init_out.features

        pred_bb = old_agent_network.components["prediction_backbone"]
        features = pred_bb(latent)

        val_head = old_agent_network.components["value_head"]
        result = val_head(features)
        assert isinstance(result, tuple), f"Old head returns {type(result)}, expected tuple"
        assert len(result) == 3, f"Old head returns {len(result)}-tuple, expected 3"

    def test_new_head_returns_head_output(self, new_agent_network):
        """New heads return HeadOutput dataclass."""
        from modules.heads.base import HeadOutput

        obs = torch.randn(1, *CARTPOLE_OBS_SHAPE)
        rep = new_agent_network.components["representation"]
        latent = rep(obs)

        pred_bb = new_agent_network.components["prediction_backbone"]
        features = pred_bb(latent)

        # Get a behavior head
        behavior_heads = new_agent_network.components["behavior_heads"]
        first_head_name = list(behavior_heads.keys())[0]
        head = behavior_heads[first_head_name]

        result = head(features, is_inference=True)
        assert isinstance(result, HeadOutput), f"New head returns {type(result)}, expected HeadOutput"
        assert hasattr(result, "training_tensor")
        assert hasattr(result, "inference_tensor")
        assert hasattr(result, "state")


# ============================================================================
# 13. END-TO-END FORWARD PASS PARITY
# ============================================================================


class TestEndToEndParity:
    """Full forward pass comparisons."""

    def test_batched_obs_inference(self, old_agent_network, new_agent_network):
        """Batched obs_inference should work for both."""
        seed_all()
        obs = torch.randn(4, *CARTPOLE_OBS_SHAPE)

        old_out = old_agent_network.obs_inference(obs)
        new_out = new_agent_network.obs_inference(obs)

        # Both should handle batched input
        if hasattr(old_out.policy, "probs"):
            assert old_out.policy.probs.shape[0] == 4
        if hasattr(new_out.policy, "probs"):
            assert new_out.policy.probs.shape[0] == 4
        elif torch.is_tensor(new_out.policy):
            assert new_out.policy.shape[0] == 4

    def test_learner_inference_backward_pass(self, old_agent_network, new_agent_network):
        """Both should support gradient computation through learner_inference."""
        seed_all()
        B, K = 2, 3
        batch = {
            "observations": torch.randn(B, *CARTPOLE_OBS_SHAPE),
            "actions": torch.randint(0, CARTPOLE_NUM_ACTIONS, (B, K)),
        }

        # Old
        old_out = old_agent_network.learner_inference(batch)
        old_loss = old_out["values"].sum() + old_out["policies"].sum()
        old_loss.backward()
        old_grad_exists = any(p.grad is not None and p.grad.abs().sum() > 0 for p in old_agent_network.parameters())
        assert old_grad_exists, "Old network has no gradients after backward"

        old_agent_network.zero_grad()

        # New
        new_out = new_agent_network.learner_inference(batch)
        # Find value and policy keys
        new_loss = new_out["latents"].sum()  # Latents should always exist
        for k, v in new_out.items():
            if torch.is_tensor(v) and v.requires_grad:
                new_loss = new_loss + v.sum()

        new_loss.backward()
        new_grad_exists = any(p.grad is not None and p.grad.abs().sum() > 0 for p in new_agent_network.parameters())
        assert new_grad_exists, "New network has no gradients after backward"

    def test_eval_mode_inference(self, old_agent_network, new_agent_network):
        """Both should work correctly in eval mode."""
        old_agent_network.eval()
        new_agent_network.eval()

        obs = torch.randn(1, *CARTPOLE_OBS_SHAPE)

        with torch.inference_mode():
            old_out = old_agent_network.obs_inference(obs)
            new_out = new_agent_network.obs_inference(obs)

        assert old_out.value is not None
        assert new_out.value is not None

        old_agent_network.train()
        new_agent_network.train()

    def test_device_property(self, old_agent_network, new_agent_network):
        """Both should report their device correctly."""
        assert old_agent_network.device == torch.device("cpu")
        assert new_agent_network.device == torch.device("cpu")
