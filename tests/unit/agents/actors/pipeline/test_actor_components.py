"""Unit tests for actor pipeline components.

Each test verifies a component's blackboard contract in isolation
using mock environments, networks, and selectors.
"""

import torch
import numpy as np
import pytest
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

from actors.pipeline.components import (
    EnvironmentState,
    EnvObservationComponent,
    ActorInferenceComponent,
    ActionSelectionComponent,
    EnvStepComponent,
    BufferStoreComponent,
    infinite_ticks,
    _sanitize_info,
)
from learner.core import Blackboard

pytestmark = pytest.mark.unit

DEVICE = torch.device("cpu")
OBS_SHAPE = (4,)
NUM_ACTIONS = 2


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class FakeEnv:
    """Minimal Gymnasium-like environment for testing."""

    def __init__(self, obs_shape=(4,), max_steps=3):
        self.obs_shape = obs_shape
        self.max_steps = max_steps
        self._step_count = 0

    def reset(self):
        self._step_count = 0
        obs = np.ones(self.obs_shape, dtype=np.float32)
        info = {"legal_moves": [0, 1]}
        return obs, info

    def step(self, action):
        self._step_count += 1
        obs = np.ones(self.obs_shape, dtype=np.float32) * self._step_count
        reward = 1.0
        terminated = self._step_count >= self.max_steps
        truncated = False
        info = {"legal_moves": [0, 1]}
        return obs, reward, terminated, truncated, info


@pytest.fixture
def fake_env():
    return FakeEnv(obs_shape=OBS_SHAPE)


@pytest.fixture
def env_state(fake_env):
    return EnvironmentState(
        env=fake_env,
        device=DEVICE,
        num_actions=NUM_ACTIONS,
        input_shape=OBS_SHAPE,
    )


# ---------------------------------------------------------------------------
# _sanitize_info
# ---------------------------------------------------------------------------

def test_sanitize_info_creates_mask_from_list():
    """Legal moves list becomes a boolean mask tensor."""
    info = {"legal_moves": [0, 2]}
    result = _sanitize_info(info, num_actions=4, device=DEVICE)

    mask = result["legal_moves_mask"]
    assert mask.dtype == torch.bool
    assert mask.shape == (4,)
    assert mask[0].item() is True
    assert mask[1].item() is False
    assert mask[2].item() is True


def test_sanitize_info_all_legal_when_empty():
    """When no legal_moves provided, all actions are legal."""
    info = {}
    result = _sanitize_info(info, num_actions=3, device=DEVICE)

    mask = result["legal_moves_mask"]
    assert mask.all()


# ---------------------------------------------------------------------------
# EnvObservationComponent
# ---------------------------------------------------------------------------

def test_env_observation_resets_on_first_tick(env_state):
    """First execute resets env and writes observation to blackboard."""
    comp = EnvObservationComponent(env_state)
    bb = Blackboard()

    comp.execute(bb)

    assert "observations" in bb.data
    obs = bb.data["observations"]
    assert obs.shape == (1, *OBS_SHAPE)  # unsqueezed batch dim
    assert obs.dtype == torch.float32
    assert "info" in bb.meta
    assert env_state.done is False


def test_env_observation_reuses_obs_when_not_done(env_state):
    """When env is not done, uses existing obs without reset."""
    env_state.obs = np.zeros(OBS_SHAPE, dtype=np.float32)
    env_state.info = {"legal_moves": [0]}
    env_state.done = False

    comp = EnvObservationComponent(env_state)
    bb = Blackboard()

    comp.execute(bb)

    obs = bb.data["observations"]
    assert torch.allclose(obs, torch.zeros(1, *OBS_SHAPE))


# ---------------------------------------------------------------------------
# ActorInferenceComponent
# ---------------------------------------------------------------------------

def test_actor_inference_writes_result():
    """ActorInferenceComponent writes InferenceResult to predictions."""
    mock_source = MagicMock()
    mock_result = MagicMock()
    mock_source.get_inference.return_value = mock_result

    comp = ActorInferenceComponent(policy_source=mock_source)
    bb = Blackboard()
    bb.data["observations"] = torch.zeros(1, 4)
    bb.meta["info"] = {}

    comp.execute(bb)

    mock_source.get_inference.assert_called_once()
    assert bb.predictions["inference_result"] is mock_result


# ---------------------------------------------------------------------------
# ActionSelectionComponent
# ---------------------------------------------------------------------------

def test_action_selection_writes_action():
    """ActionSelectionComponent writes action and metadata to meta."""
    mock_selector = MagicMock()
    mock_selector.select_action.return_value = (torch.tensor(1), {"log_prob": torch.tensor(-0.5)})

    mock_result = MagicMock()
    mock_result.extra_metadata = {}
    mock_result.probs = None
    mock_result.value = torch.tensor(0.5)

    comp = ActionSelectionComponent(action_selector=mock_selector, exploration=True)
    bb = Blackboard()
    bb.predictions["inference_result"] = mock_result
    bb.meta["info"] = {}

    comp.execute(bb)

    assert bb.meta["action"] == 1
    assert torch.is_tensor(bb.meta["action_tensor"])
    metadata = bb.meta["action_metadata"]
    assert "log_prob" in metadata
    # Value fallback from result
    assert metadata["value"] is not None


def test_action_selection_merges_extra_metadata():
    """Extra metadata from PolicySource is merged into action_metadata."""
    mock_selector = MagicMock()
    mock_selector.select_action.return_value = (torch.tensor(0), {})

    mock_result = MagicMock()
    mock_result.extra_metadata = {"search_duration": 0.01}
    mock_result.probs = None
    mock_result.value = None

    comp = ActionSelectionComponent(action_selector=mock_selector)
    bb = Blackboard()
    bb.predictions["inference_result"] = mock_result
    bb.meta["info"] = {}

    comp.execute(bb)

    assert bb.meta["action_metadata"]["search_duration"] == 0.01


# ---------------------------------------------------------------------------
# EnvStepComponent
# ---------------------------------------------------------------------------

def test_env_step_writes_transition(env_state):
    """EnvStepComponent steps env and writes transition to blackboard."""
    # Manually initialise env state (simulate post-reset)
    env_state.obs, env_state.info = env_state.env.reset()
    env_state.done = False

    comp = EnvStepComponent(env_state)
    bb = Blackboard()
    bb.meta["action"] = 0

    comp.execute(bb)

    assert "rewards" in bb.data
    assert "dones" in bb.data
    assert "terminated" in bb.data
    assert "truncated" in bb.data
    assert "next_observations" in bb.data
    assert bb.data["rewards"].item() == 1.0
    assert env_state.episode_reward == 1.0
    assert env_state.episode_length == 1


def test_env_step_signals_episode_end(env_state):
    """EnvStepComponent writes episode stats when done."""
    env_state.obs, env_state.info = env_state.env.reset()
    env_state.done = False
    comp = EnvStepComponent(env_state)

    # Step to terminal (max_steps=3)
    for _ in range(3):
        bb = Blackboard()
        bb.meta["action"] = 0
        comp.execute(bb)

    assert env_state.done is True
    assert bb.meta["episode_score"] == 3.0
    assert bb.meta["episode_length"] == 3


# ---------------------------------------------------------------------------
# BufferStoreComponent
# ---------------------------------------------------------------------------

def test_buffer_store_calls_store():
    """BufferStoreComponent calls replay_buffer.store with transition data."""
    mock_buffer = MagicMock()
    comp = BufferStoreComponent(replay_buffer=mock_buffer)

    bb = Blackboard()
    bb.data["observations"] = torch.ones(1, 4)
    bb.data["rewards"] = torch.tensor(1.0)
    bb.data["dones"] = torch.tensor(False)
    bb.data["next_observations"] = torch.ones(1, 4) * 2
    bb.meta["action"] = 1
    bb.meta["info"] = {}
    bb.meta["action_metadata"] = {"value": torch.tensor(0.5)}

    comp.execute(bb)

    mock_buffer.store.assert_called_once()
    call_kwargs = mock_buffer.store.call_args.kwargs
    assert "observations" in call_kwargs
    assert "actions" in call_kwargs
    assert call_kwargs["actions"] == 1
    assert "value" in call_kwargs


# ---------------------------------------------------------------------------
# infinite_ticks
# ---------------------------------------------------------------------------

def test_infinite_ticks_yields_empty_dicts():
    """infinite_ticks yields empty dicts indefinitely."""
    gen = infinite_ticks()
    for _ in range(5):
        tick = next(gen)
        assert tick == {}


# ---------------------------------------------------------------------------
# Integration: full actor pipeline (one tick)
# ---------------------------------------------------------------------------

def test_full_actor_pipeline_one_tick(env_state):
    """Smoke test: all actor components compose through BlackboardEngine."""
    from learner.core import BlackboardEngine

    mock_source = MagicMock()
    mock_result = MagicMock()
    mock_result.extra_metadata = {}
    mock_result.probs = None
    mock_result.value = torch.tensor(0.5)
    mock_source.get_inference.return_value = mock_result

    mock_selector = MagicMock()
    mock_selector.select_action.return_value = (torch.tensor(0), {})

    mock_buffer = MagicMock()

    pipeline = [
        EnvObservationComponent(env_state),
        ActorInferenceComponent(mock_source),
        ActionSelectionComponent(mock_selector),
        EnvStepComponent(env_state),
        BufferStoreComponent(mock_buffer),
    ]

    engine = BlackboardEngine(components=pipeline, device=DEVICE)
    gen = engine.step(infinite_ticks())

    # Run one tick
    output = next(gen)

    assert "meta" in output
    mock_source.get_inference.assert_called_once()
    mock_selector.select_action.assert_called_once()
    mock_buffer.store.assert_called_once()
    assert env_state.episode_length == 1
