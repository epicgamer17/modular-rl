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
5. Inference Output Data Structure Parity
6. Target Builder Parity
7. Component Architecture Parity
8. Initialization Parity
9. Replay Buffer Processor Parity
10. Action Selector Parity
11. Loss Pipeline Parity
12. Head Output Contract Parity
13. End-to-End Forward Pass Parity
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


def _base_muzero_dict(flavor="old"):
    """
    Create a base MuZero config dict.
    flavor="old" uses "dense" backbone type (old system).
    flavor="new" uses "mlp" backbone type (new system).
    """
    bb_type = "dense" if flavor == "old" else "mlp"
    bb_key = "widths"
    return {
        "minibatch_size": 4,
        "min_replay_buffer_size": 0,
        "num_simulations": 3,
        "discount_factor": 0.99,
        "unroll_steps": 3,
        "n_step": 3,
        "learning_rate": 0.01,
        "architecture": {"backbone": {"type": bb_type, bb_key: [32]}},
        "representation_backbone": {"type": bb_type, bb_key: [32]},
        "dynamics_backbone": {"type": bb_type, bb_key: [32]},
        "prediction_backbone": {"type": bb_type, bb_key: [32]},
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


def _base_muzero_dict_distributional(flavor="old"):
    d = _base_muzero_dict(flavor)
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
    return MuZeroConfig(_base_muzero_dict("old"), old_cartpole_game_config)


@pytest.fixture
def old_muzero_config_distributional(old_cartpole_game_config):
    from old_muzero.configs.agents.muzero import MuZeroConfig
    return MuZeroConfig(_base_muzero_dict_distributional("old"), old_cartpole_game_config)


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
    return MuZeroConfig(_base_muzero_dict("new"), new_cartpole_game_config)


@pytest.fixture
def new_muzero_config_distributional(new_cartpole_game_config):
    from configs.agents.muzero import MuZeroConfig
    return MuZeroConfig(_base_muzero_dict_distributional("new"), new_cartpole_game_config)


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
        assert old_muzero_config.atom_size == new_muzero_config.atom_size

    def test_distributional_config_explicit(
        self, old_muzero_config_distributional, new_muzero_config_distributional
    ):
        old = old_muzero_config_distributional
        new = new_muzero_config_distributional
        assert old.atom_size == new.atom_size
        assert old.support_range == new.support_range

    def test_backbone_configs_present(self, old_muzero_config, new_muzero_config):
        assert old_muzero_config.representation_backbone is not None
        assert old_muzero_config.dynamics_backbone is not None
        assert old_muzero_config.prediction_backbone is not None
        assert new_muzero_config.representation_backbone is not None
        assert new_muzero_config.dynamics_backbone is not None
        assert new_muzero_config.prediction_backbone is not None


# ============================================================================
# 2. WORLD MODEL PARITY
# ============================================================================


class TestWorldModelParity:
    """Compare world model outputs between old and new."""

    def test_initial_inference_output_shapes(self, old_agent_network, new_agent_network):
        seed_all()
        obs = torch.randn(1, *CARTPOLE_OBS_SHAPE)

        old_wm = old_agent_network.components["world_model"]
        old_out = old_wm.initial_inference(obs)

        new_rep = new_agent_network.components["representation"]
        new_latent = new_rep(obs)

        assert old_out.features.shape == new_latent.shape, (
            f"Old latent shape: {old_out.features.shape}, New latent shape: {new_latent.shape}"
        )

    def test_recurrent_inference_output_keys(self, old_agent_network, new_agent_network):
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

        assert old_rec.features is not None
        assert new_rec.features is not None
        assert old_rec.features.shape == new_rec.features.shape

    def test_recurrent_inference_reward_shape(self, old_agent_network, new_agent_network):
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

        assert old_rec.reward is not None
        assert new_rec.reward is not None
        assert old_rec.reward.shape == new_rec.reward.shape, (
            f"Old reward shape: {old_rec.reward.shape}, New reward shape: {new_rec.reward.shape}"
        )

    def test_unroll_physics_latents_shape(self, old_agent_network, new_agent_network):
        seed_all()
        B, K = 4, 3
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

        old_latents = old_physics.latents
        new_latents = new_physics["latents"]

        assert old_latents.shape == new_latents.shape, (
            f"Old latents: {old_latents.shape}, New latents: {new_latents.shape}"
        )

    def test_unroll_physics_temporal_dim_k_plus_1(self, old_agent_network, new_agent_network):
        seed_all()
        B, K = 4, 3
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
        """Old produces [B, K] rewards. New may produce [B, K+1] (head at each state)."""
        seed_all()
        B, K = 4, 3
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
        new_rew_key = None
        for k in new_physics.keys():
            if "reward" in k.lower():
                new_rew_key = k
                break

        if new_rew_key:
            new_rew_t = new_physics[new_rew_key].shape[1]
            print(f"PARITY NOTE - Old reward T={old_rew_t}, New reward T={new_rew_t} (key={new_rew_key})")
        else:
            pytest.skip("New world model does not produce reward key in unroll_physics")


# ============================================================================
# 3. AGENT NETWORK PARITY
# ============================================================================


class TestAgentNetworkParity:
    """Compare full agent network inference outputs."""

    def test_obs_inference_returns_value(self, old_agent_network, new_agent_network):
        seed_all()
        obs = torch.randn(1, *CARTPOLE_OBS_SHAPE)
        old_out = old_agent_network.obs_inference(obs)
        new_out = new_agent_network.obs_inference(obs)
        assert old_out.value is not None
        assert new_out.value is not None

    def test_obs_inference_returns_policy(self, old_agent_network, new_agent_network):
        seed_all()
        obs = torch.randn(1, *CARTPOLE_OBS_SHAPE)
        old_out = old_agent_network.obs_inference(obs)
        new_out = new_agent_network.obs_inference(obs)
        assert old_out.policy is not None
        assert new_out.policy is not None

    def test_obs_inference_network_state_has_dynamics(self, old_agent_network, new_agent_network):
        seed_all()
        obs = torch.randn(1, *CARTPOLE_OBS_SHAPE)
        old_out = old_agent_network.obs_inference(obs)
        new_out = new_agent_network.obs_inference(obs)
        assert hasattr(old_out.network_state, "dynamics")
        assert "dynamics" in new_out.recurrent_state

    def test_obs_inference_policy_num_actions(self, old_agent_network, new_agent_network):
        seed_all()
        obs = torch.randn(1, *CARTPOLE_OBS_SHAPE)
        old_out = old_agent_network.obs_inference(obs)
        new_out = new_agent_network.obs_inference(obs)

        old_pol = old_out.policy
        new_pol = new_out.policy

        if hasattr(old_pol, "probs"):
            assert old_pol.probs.shape[-1] == CARTPOLE_NUM_ACTIONS
        if hasattr(new_pol, "probs"):
            assert new_pol.probs.shape[-1] == CARTPOLE_NUM_ACTIONS
        elif torch.is_tensor(new_pol):
            assert new_pol.shape[-1] == CARTPOLE_NUM_ACTIONS

    def test_hidden_state_inference_output_fields(self, old_agent_network, new_agent_network):
        seed_all()
        obs = torch.randn(1, *CARTPOLE_OBS_SHAPE)
        action = torch.tensor([0])

        old_root = old_agent_network.obs_inference(obs)
        old_step = old_agent_network.hidden_state_inference(old_root.network_state, action)

        new_root = new_agent_network.obs_inference(obs)
        new_step = new_agent_network.hidden_state_inference(new_root.recurrent_state, action)

        assert old_step.value is not None
        assert new_step.value is not None
        assert old_step.policy is not None
        assert new_step.policy is not None

    def test_hidden_state_chaining(self, old_agent_network, new_agent_network):
        seed_all()
        obs = torch.randn(1, *CARTPOLE_OBS_SHAPE)

        old_state = old_agent_network.obs_inference(obs).network_state
        for i in range(3):
            action = torch.tensor([i % CARTPOLE_NUM_ACTIONS])
            old_out = old_agent_network.hidden_state_inference(old_state, action)
            old_state = old_out.network_state

        new_state = new_agent_network.obs_inference(obs).recurrent_state
        for i in range(3):
            action = torch.tensor([i % CARTPOLE_NUM_ACTIONS])
            new_out = new_agent_network.hidden_state_inference(new_state, action)
            new_state = new_out.recurrent_state

        assert old_out.value is not None
        assert new_out.value is not None

    def test_learner_inference_output_keys(self, old_agent_network, new_agent_network):
        seed_all()
        B, K = 4, 3
        batch = {
            "observations": torch.randn(B, *CARTPOLE_OBS_SHAPE),
            "actions": torch.randint(0, CARTPOLE_NUM_ACTIONS, (B, K)),
        }
        old_out = old_agent_network.learner_inference(batch)
        new_out = new_agent_network.learner_inference(batch)

        assert "values" in old_out
        assert "policies" in old_out
        assert "rewards" in old_out
        assert "latents" in old_out
        assert "latents" in new_out

        old_keys = set(old_out.keys())
        new_keys = set(new_out.keys())
        print(f"PARITY MAP - Old learner_inference keys: {sorted(old_keys)}")
        print(f"PARITY MAP - New learner_inference keys: {sorted(new_keys)}")

    def test_learner_inference_values_shape(self, old_agent_network, new_agent_network):
        seed_all()
        B, K = 4, 3
        batch = {
            "observations": torch.randn(B, *CARTPOLE_OBS_SHAPE),
            "actions": torch.randint(0, CARTPOLE_NUM_ACTIONS, (B, K)),
        }
        old_out = old_agent_network.learner_inference(batch)
        new_out = new_agent_network.learner_inference(batch)

        old_vals = old_out["values"]
        assert old_vals.shape[0] == B
        assert old_vals.shape[1] == K + 1

        # Find value key in new output
        for k in new_out:
            if "value" in k.lower():
                new_vals = new_out[k]
                print(f"PARITY NOTE - Old values shape: {old_vals.shape}, New '{k}' shape: {new_vals.shape}")
                assert new_vals.shape[0] == B
                assert new_vals.shape[1] == K + 1
                break

    def test_learner_inference_policies_shape(self, old_agent_network, new_agent_network):
        seed_all()
        B, K = 4, 3
        batch = {
            "observations": torch.randn(B, *CARTPOLE_OBS_SHAPE),
            "actions": torch.randint(0, CARTPOLE_NUM_ACTIONS, (B, K)),
        }
        old_out = old_agent_network.learner_inference(batch)
        new_out = new_agent_network.learner_inference(batch)

        old_pols = old_out["policies"]
        assert old_pols.shape == (B, K + 1, CARTPOLE_NUM_ACTIONS)

        for k in new_out:
            if "polic" in k.lower():
                new_pols = new_out[k]
                assert new_pols.shape[0] == B
                assert new_pols.shape[1] == K + 1
                assert new_pols.shape[2] == CARTPOLE_NUM_ACTIONS
                break


# ============================================================================
# 4. N-STEP TARGET COMPUTATION PARITY
# ============================================================================


class TestNStepTargetParity:
    """Compare the vectorized N-step target computation between old and new."""

    def test_n_step_single_player_no_termination(self):
        seed_all()
        B, L = 4, 15
        raw_rewards = torch.rand(B, L)
        raw_values = torch.rand(B, L)
        raw_to_plays = torch.zeros(B, L, dtype=torch.long)
        raw_terminated = torch.zeros(B, L, dtype=torch.bool)
        valid_mask = torch.ones(B, L, dtype=torch.bool)

        gamma, n_step, unroll_steps = 0.99, 3, 3

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

        from agents.learner.functional.returns import compute_unrolled_n_step_targets
        new_vals, new_rews = compute_unrolled_n_step_targets(
            raw_rewards=raw_rewards, raw_values=raw_values,
            raw_to_plays=raw_to_plays, raw_terminated=raw_terminated,
            valid_mask=valid_mask, gamma=gamma, n_step=n_step, unroll_steps=unroll_steps,
        )

        assert old_vals.shape == new_vals.shape
        assert old_rews.shape == new_rews.shape

        if not torch.allclose(old_vals, new_vals, atol=1e-5):
            diff = (old_vals - new_vals).abs()
            print(f"PARITY DIFF - N-step values max diff: {diff.max().item():.8f}")
        else:
            print("PARITY OK - N-step values match")

        if not torch.allclose(old_rews, new_rews, atol=1e-5):
            diff = (old_rews - new_rews).abs()
            print(f"PARITY DIFF - N-step rewards max diff: {diff.max().item():.8f}")
        else:
            print("PARITY OK - N-step rewards match")

    def test_n_step_with_termination(self):
        seed_all()
        B, L = 4, 10
        raw_rewards = torch.rand(B, L)
        raw_values = torch.rand(B, L)
        raw_to_plays = torch.zeros(B, L, dtype=torch.long)
        raw_terminated = torch.zeros(B, L, dtype=torch.bool)
        raw_terminated[:, 7] = True
        valid_mask = torch.ones(B, L, dtype=torch.bool)
        valid_mask[:, 8:] = False

        gamma, n_step, unroll_steps = 0.99, 3, 3

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

        from agents.learner.functional.returns import compute_unrolled_n_step_targets
        new_vals, new_rews = compute_unrolled_n_step_targets(
            raw_rewards=raw_rewards, raw_values=raw_values,
            raw_to_plays=raw_to_plays, raw_terminated=raw_terminated,
            valid_mask=valid_mask, gamma=gamma, n_step=n_step, unroll_steps=unroll_steps,
        )

        assert old_vals.shape == new_vals.shape
        if not torch.allclose(old_vals, new_vals, atol=1e-5):
            diff = (old_vals - new_vals).abs()
            print(f"PARITY DIFF - Terminated n-step values max diff: {diff.max().item():.8f}")
            for b in range(min(2, B)):
                print(f"  Batch {b} old: {old_vals[b]}")
                print(f"  Batch {b} new: {new_vals[b]}")

    def test_n_step_multiplayer_signs(self):
        seed_all()
        B, L = 4, 15
        raw_rewards = torch.rand(B, L)
        raw_values = torch.rand(B, L)
        raw_to_plays = torch.zeros(B, L, dtype=torch.long)
        for i in range(L):
            raw_to_plays[:, i] = i % 2
        raw_terminated = torch.zeros(B, L, dtype=torch.bool)
        valid_mask = torch.ones(B, L, dtype=torch.bool)

        gamma, n_step, unroll_steps = 0.99, 3, 3

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
            raw_rewards=raw_rewards, raw_values=raw_values,
            raw_to_plays=raw_to_plays, raw_terminated=raw_terminated,
            valid_mask=valid_mask, gamma=gamma, n_step=n_step, unroll_steps=unroll_steps,
        )

        if not torch.allclose(old_vals, new_vals, atol=1e-5):
            diff = (old_vals - new_vals).abs()
            print(f"PARITY DIFF - Multiplayer n-step values max diff: {diff.max().item():.8f}")
        else:
            print("PARITY OK - Multiplayer n-step values match")

    def test_n_step_value_prefix_mode(self):
        seed_all()
        B, L = 4, 15
        raw_rewards = torch.rand(B, L)
        raw_values = torch.rand(B, L)
        raw_to_plays = torch.zeros(B, L, dtype=torch.long)
        raw_terminated = torch.zeros(B, L, dtype=torch.bool)
        valid_mask = torch.ones(B, L, dtype=torch.bool)

        gamma, n_step, unroll_steps, lstm_horizon_len = 0.99, 3, 3, 5

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
            raw_rewards=raw_rewards, raw_values=raw_values,
            raw_to_plays=raw_to_plays, raw_terminated=raw_terminated,
            valid_mask=valid_mask, gamma=gamma, n_step=n_step, unroll_steps=unroll_steps,
            use_value_prefix=True, lstm_horizon_len=lstm_horizon_len,
        )

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

    def test_inference_output_field_mapping(self):
        from old_muzero.modules.world_models.inference_output import InferenceOutput as OldIO
        from modules.models.inference_output import InferenceOutput as NewIO

        old_fields = set(OldIO._fields)
        new_fields = set(NewIO.__dataclass_fields__.keys())

        print(f"PARITY MAP - InferenceOutput common: {old_fields & new_fields}")
        print(f"PARITY MAP - Only in old: {old_fields - new_fields}")
        print(f"PARITY MAP - Only in new: {new_fields - old_fields}")

        for f in {"value", "policy", "reward", "to_play"}:
            assert f in old_fields, f"Old missing {f}"
            assert f in new_fields, f"New missing {f}"

    def test_world_model_output_field_mapping(self):
        from old_muzero.modules.world_models.inference_output import WorldModelOutput as OldWMO
        from modules.models.inference_output import WorldModelOutput as NewWMO

        old_fields = set(OldWMO._fields)
        new_fields = set(NewWMO.__dataclass_fields__.keys())

        print(f"PARITY MAP - WorldModelOutput common: {old_fields & new_fields}")
        print(f"PARITY MAP - Only in old: {old_fields - new_fields}")
        print(f"PARITY MAP - Only in new: {new_fields - old_fields}")

    def test_old_network_state_batch_unbatch(self):
        from old_muzero.modules.world_models.inference_output import MuZeroNetworkState
        states = [
            MuZeroNetworkState(dynamics=torch.randn(1, 32)),
            MuZeroNetworkState(dynamics=torch.randn(1, 32)),
        ]
        batched = MuZeroNetworkState.batch(states)
        assert batched.dynamics.shape[0] == 2
        unbatched = batched.unbatch()
        assert len(unbatched) == 2

    def test_new_network_state_batch_unbatch(self):
        from modules.models.inference_output import batch_recurrent_state, unbatch_recurrent_state
        states = [
            {"dynamics": torch.randn(1, 32)},
            {"dynamics": torch.randn(1, 32)},
        ]
        batched = batch_recurrent_state(states)
        assert batched["dynamics"].shape[0] == 2
        unbatched = unbatch_recurrent_state(batched)
        assert len(unbatched) == 2


# ============================================================================
# 6. TARGET BUILDER PARITY
# ============================================================================


class TestTargetBuilderParity:

    def test_mcts_extractor_removed_in_new(self):
        """PARITY DIFF: MCTSExtractor exists in old but was removed in new system.
        The new system handles MCTS target extraction differently."""
        from old_muzero.agents.learner.target_builders import MCTSExtractor as OldMCTS
        assert OldMCTS is not None, "Old MCTSExtractor should exist"

        # Verify it's NOT in the new system
        import agents.learner.target_builders as new_tb
        assert not hasattr(new_tb, "MCTSExtractor"), (
            "PARITY DIFF: MCTSExtractor was removed from new target_builders"
        )
        print("PARITY DIFF: MCTSExtractor exists in old, removed in new system")

    def test_sequence_padder(self):
        from old_muzero.agents.learner.target_builders import SequencePadder as OldP
        from agents.learner.target_builders import SequencePadder as NewP

        targets = {"rewards": torch.rand(4, 3), "to_plays": torch.rand(4, 3, 2)}
        old_t = copy.deepcopy(targets)
        new_t = copy.deepcopy(targets)
        OldP(3).build_targets({}, {}, None, old_t)
        NewP(3).build_targets({}, {}, None, new_t)

        for k in ["rewards", "to_plays"]:
            assert old_t[k].shape == new_t[k].shape
            assert torch.allclose(old_t[k], new_t[k])

    def test_sequence_mask_builder(self):
        from old_muzero.agents.learner.target_builders import SequenceMaskBuilder as OldM
        from agents.learner.target_builders import SequenceMaskBuilder as NewM

        B, T = 4, 4
        batch = {
            "is_same_game": torch.ones(B, T, dtype=torch.bool),
            "has_valid_obs_mask": torch.ones(B, T, dtype=torch.bool),
        }
        base = {"actions": torch.randint(0, 2, (B, T))}

        old_t = copy.deepcopy(base)
        new_t = copy.deepcopy(base)
        OldM().build_targets(batch, {}, None, old_t)
        NewM().build_targets(batch, {}, None, new_t)

        for k in ["value_mask", "masks", "policy_mask", "reward_mask"]:
            if k in old_t and k in new_t:
                assert torch.equal(old_t[k], new_t[k]), f"Mask '{k}' differs"

    def test_sequence_infrastructure_builder(self):
        from old_muzero.agents.learner.target_builders import SequenceInfrastructureBuilder as OldI
        from agents.learner.target_builders import SequenceInfrastructureBuilder as NewI

        B, T = 4, 4
        batch = {"weights": torch.ones(B)}
        base = {"actions": torch.randint(0, 2, (B, T))}

        old_t = copy.deepcopy(base)
        new_t = copy.deepcopy(base)
        OldI(3).build_targets(batch, {}, None, old_t)
        NewI(3).build_targets(batch, {}, None, new_t)

        assert torch.allclose(old_t["gradient_scales"], new_t["gradient_scales"])
        assert torch.allclose(old_t["weights"], new_t["weights"])


# ============================================================================
# 7. COMPONENT ARCHITECTURE PARITY
# ============================================================================


class TestComponentArchitectureParity:

    def test_parameter_count_comparison(self, old_agent_network, new_agent_network):
        old_p = sum(p.numel() for p in old_agent_network.parameters())
        new_p = sum(p.numel() for p in new_agent_network.parameters())
        print(f"PARITY NOTE - Total params: old={old_p}, new={new_p}")

    def test_component_names(self, old_agent_network, new_agent_network):
        print(f"PARITY MAP - Old components: {set(old_agent_network.components.keys())}")
        print(f"PARITY MAP - New components: {set(new_agent_network.components.keys())}")

    def test_old_wm_owns_reward_and_toplay_heads(self, old_agent_network):
        wm = old_agent_network.components["world_model"]
        assert hasattr(wm, "reward_head")
        assert hasattr(wm, "to_play_head")

    def test_new_wm_heads_location(self, new_agent_network):
        wm = new_agent_network.components["world_model"]
        assert hasattr(wm, "heads")
        print(f"PARITY MAP - New WM heads: {list(wm.heads.keys())}")

    def test_representation_exists(self, old_agent_network, new_agent_network):
        assert hasattr(old_agent_network.components["world_model"], "representation")
        assert "representation" in new_agent_network.components

    def test_prediction_backbone_exists(self, old_agent_network, new_agent_network):
        assert "prediction_backbone" in old_agent_network.components
        assert "prediction_backbone" in new_agent_network.components

    def test_value_head_location(self, old_agent_network, new_agent_network):
        assert "value_head" in old_agent_network.components
        assert "behavior_heads" in new_agent_network.components
        bh = new_agent_network.components["behavior_heads"]
        assert any("value" in n.lower() for n in bh.keys()), f"No value head in {list(bh.keys())}"

    def test_policy_head_location(self, old_agent_network, new_agent_network):
        assert "policy_head" in old_agent_network.components
        assert "behavior_heads" in new_agent_network.components
        bh = new_agent_network.components["behavior_heads"]
        assert any("polic" in n.lower() for n in bh.keys()), f"No policy head in {list(bh.keys())}"


# ============================================================================
# 8. INITIALIZATION PARITY
# ============================================================================


class TestInitializationParity:

    def test_orthogonal_init(self, old_agent_network, new_agent_network):
        seed_all()
        old_agent_network.initialize("orthogonal")
        seed_all()
        new_agent_network.initialize("orthogonal")

    def test_xavier_uniform_init_naming_diff(self, old_agent_network, new_agent_network):
        """PARITY DIFF: Old uses 'glorot_uniform', new uses 'xavier_uniform' for same init."""
        seed_all()
        old_agent_network.initialize("glorot_uniform")
        seed_all()
        new_agent_network.initialize("xavier_uniform")
        print("PARITY DIFF: Old='glorot_uniform', New='xavier_uniform' (same underlying init)")

    def test_init_changes_weights(self, old_agent_network, new_agent_network):
        old_w = next(old_agent_network.parameters()).clone()
        new_w = next(new_agent_network.parameters()).clone()
        seed_all()
        old_agent_network.initialize("glorot_uniform")
        seed_all()
        new_agent_network.initialize("xavier_uniform")
        assert not torch.equal(old_w, next(old_agent_network.parameters()))
        assert not torch.equal(new_w, next(new_agent_network.parameters()))


# ============================================================================
# 9. REPLAY BUFFER PROCESSOR PARITY
# ============================================================================


class TestReplayBufferProcessorParity:

    def _make_buffers(self, max_size=50, num_episodes=3, ep_len=10):
        seed_all()
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
        game_ids = torch.zeros(max_size, dtype=torch.long)

        for ep in range(num_episodes):
            start = ep * ep_len
            end = start + ep_len
            game_ids[start:end] = ep
            terminated[end - 1] = True
            dones[end - 1] = True

        return {
            "observations": obs, "rewards": rewards, "values": values,
            "policies": policies, "actions": actions, "to_plays": to_plays,
            "chances": chances, "terminated": terminated, "truncated": truncated,
            "dones": dones, "game_ids": game_ids, "legal_masks": legal_masks,
            "training_steps": training_steps, "ids": ids,
        }

    def test_output_processor_shapes(self):
        from old_muzero.replay_buffers.processors import NStepUnrollProcessor
        buffers = self._make_buffers()
        proc = NStepUnrollProcessor(
            unroll_steps=3, n_step=3, gamma=0.99,
            num_actions=CARTPOLE_NUM_ACTIONS, num_players=CARTPOLE_NUM_PLAYERS,
            max_size=50,
        )
        result = proc.process_batch([0, 5, 10, 15], buffers)
        for k, v in result.items():
            if torch.is_tensor(v):
                print(f"PARITY MAP - NStepUnrollProcessor '{k}': {v.shape} {v.dtype}")

    def test_validity_masks_at_episode_boundary(self):
        from old_muzero.replay_buffers.processors import NStepUnrollProcessor
        buffers = self._make_buffers(max_size=50, num_episodes=3, ep_len=10)
        proc = NStepUnrollProcessor(
            unroll_steps=3, n_step=3, gamma=0.99,
            num_actions=CARTPOLE_NUM_ACTIONS, num_players=CARTPOLE_NUM_PLAYERS,
            max_size=50,
        )
        result = proc.process_batch([8], buffers)
        same_game = result["is_same_game"]
        assert not same_game[0, -1].item(), "Expected same_game=False past episode boundary"


# ============================================================================
# 10. ACTION SELECTOR PARITY
# ============================================================================


class TestActionSelectorParity:

    def test_categorical_selector_importable(self):
        from old_muzero.agents.action_selectors.selectors import CategoricalSelector as Old
        from agents.action_selectors.selectors import CategoricalSelector as New
        Old()
        New()

    def test_temperature_decorator_importable(self):
        from old_muzero.agents.action_selectors.decorators import TemperatureSelector as Old
        from agents.action_selectors.decorators import TemperatureSelector as New


# ============================================================================
# 11. LOSS PIPELINE PARITY
# ============================================================================


class TestLossPipelineParity:

    def test_muzero_loss_modules_old(self, old_agent_network, old_muzero_config):
        from old_muzero.agents.registries.muzero import build_muzero_loss_pipeline
        pipeline = build_muzero_loss_pipeline(old_muzero_config, old_agent_network, "cpu")
        names = [type(m).__name__ for m in pipeline.modules]
        print(f"PARITY MAP - Old loss modules: {names}")
        assert {"ValueLoss", "PolicyLoss", "RewardLoss"}.issubset(set(names))

    def test_loss_module_required_keys(self, old_agent_network, old_muzero_config):
        from old_muzero.agents.registries.muzero import build_muzero_loss_pipeline
        pipeline = build_muzero_loss_pipeline(old_muzero_config, old_agent_network, "cpu")
        for m in pipeline.modules:
            preds = getattr(m, "required_predictions", set())
            targs = getattr(m, "required_targets", set())
            print(f"PARITY MAP - {type(m).__name__}: preds={preds}, targs={targs}")


# ============================================================================
# 12. HEAD OUTPUT CONTRACT PARITY
# ============================================================================


class TestHeadOutputParity:

    def test_old_head_returns_tuple(self, old_agent_network):
        obs = torch.randn(1, *CARTPOLE_OBS_SHAPE)
        wm = old_agent_network.components["world_model"]
        latent = wm.initial_inference(obs).features
        features = old_agent_network.components["prediction_backbone"](latent)
        result = old_agent_network.components["value_head"](features)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_new_head_returns_head_output(self, new_agent_network):
        from modules.heads.base import HeadOutput
        obs = torch.randn(1, *CARTPOLE_OBS_SHAPE)
        latent = new_agent_network.components["representation"](obs)
        features = new_agent_network.components["prediction_backbone"](latent)

        bh = new_agent_network.components["behavior_heads"]
        head = bh[list(bh.keys())[0]]
        result = head(features, is_inference=True)
        assert isinstance(result, HeadOutput)
        assert hasattr(result, "training_tensor")
        assert hasattr(result, "inference_tensor")


# ============================================================================
# 13. END-TO-END FORWARD PASS PARITY
# ============================================================================


class TestEndToEndParity:

    def test_batched_obs_inference(self, old_agent_network, new_agent_network):
        seed_all()
        obs = torch.randn(4, *CARTPOLE_OBS_SHAPE)
        old_out = old_agent_network.obs_inference(obs)
        new_out = new_agent_network.obs_inference(obs)
        assert old_out.value is not None
        assert new_out.value is not None

    def test_learner_inference_backward(self, old_agent_network, new_agent_network):
        seed_all()
        B, K = 4, 3
        batch = {
            "observations": torch.randn(B, *CARTPOLE_OBS_SHAPE),
            "actions": torch.randint(0, CARTPOLE_NUM_ACTIONS, (B, K)),
        }

        old_out = old_agent_network.learner_inference(batch)
        (old_out["values"].sum() + old_out["policies"].sum()).backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in old_agent_network.parameters())
        old_agent_network.zero_grad()

        new_out = new_agent_network.learner_inference(batch)
        loss = sum(v.sum() for v in new_out.values() if torch.is_tensor(v) and v.requires_grad)
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in new_agent_network.parameters())

    def test_eval_mode_inference(self, old_agent_network, new_agent_network):
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
        assert old_agent_network.device == torch.device("cpu")
        assert new_agent_network.device == torch.device("cpu")
