"""
Regression test for PettingZooObservationComponent multi-episode reset.

Before the fix, the component failed to reset the environment between episodes
because PettingZoo AEC envs still have agents after a terminal move (they need
step(None) calls to drain). The reset condition only checked `not self.env.agents`,
which was False, so the env was never reset. Every episode after the first was a
single no-op tick that immediately hit stop_execution, and the replay buffer
stayed frozen at the size of episode 1.

The fix adds `self.done` to the reset condition so that a completed episode
triggers a fresh env.reset() on the next tick.
"""
import pytest
import numpy as np

from core import Blackboard
from components.environments.pettingzoo import (
    PettingZooObservationComponent,
    PettingZooStepComponent,
)
from envs.factories.tictactoe import tictactoe_factory

pytestmark = pytest.mark.integration


def _play_one_episode(obs_comp, step_comp):
    """
    Plays one full episode using random legal moves.

    Mirrors the real pipeline: obs_comp reads state, if done we stop (like
    MCTSSearchComponent would set stop_execution), otherwise we act and step.

    Returns the number of actions taken.
    """
    actions_taken = 0
    for _ in range(50):  # safety limit
        bb = Blackboard()
        obs_comp.execute(bb)

        # MCTSSearchComponent would check done here and set stop_execution
        if bb.data.get("done") or bb.data.get("terminated"):
            break

        info = bb.data.get("info", {})
        action_mask = info.get("action_mask", None)
        if action_mask is not None:
            legal = [i for i, v in enumerate(action_mask) if v]
        else:
            legal = list(range(9))
        action = np.random.choice(legal) if legal else 0

        bb.meta["action"] = action
        step_comp.execute(bb)
        actions_taken += 1

        # The step component sets obs_comp.done when the game ends.
        # In the real pipeline, the NEXT tick's obs_comp.execute() sees done
        # and the MCTS component stops execution. We simulate this by checking
        # obs_comp.done after stepping.
        if obs_comp.done:
            break

    return actions_taken


def test_multi_episode_reset():
    """Component must reset the env between episodes via the done flag."""
    np.random.seed(42)
    env = tictactoe_factory()
    obs_comp = PettingZooObservationComponent(env)
    step_comp = PettingZooStepComponent(env, obs_comp)

    episode_lengths = []
    for _ in range(5):
        length = _play_one_episode(obs_comp, step_comp)
        episode_lengths.append(length)

    # TicTacToe needs at least 3 moves (5 half-moves = 3 by one player) to end
    for i, length in enumerate(episode_lengths):
        assert length >= 3, (
            f"Episode {i} played only {length} action(s). "
            f"All episode lengths: {episode_lengths}. "
            "The env was not reset between episodes."
        )


def test_done_flag_lifecycle():
    """done flag must be True after episode, False after reset on next tick."""
    np.random.seed(42)
    env = tictactoe_factory()
    obs_comp = PettingZooObservationComponent(env)
    step_comp = PettingZooStepComponent(env, obs_comp)

    _play_one_episode(obs_comp, step_comp)
    assert obs_comp.done is True, "done flag should be True after episode ends"

    # First tick of the next episode — obs_comp should reset and clear done
    bb = Blackboard()
    obs_comp.execute(bb)
    assert obs_comp.done is False, "done flag should be False after env reset"
    assert bb.data["done"] is False, "new episode should not start as done"


def test_consecutive_episodes_produce_valid_observations():
    """Each episode must produce valid (non-None) observations after reset."""
    np.random.seed(42)
    env = tictactoe_factory()
    obs_comp = PettingZooObservationComponent(env)
    step_comp = PettingZooStepComponent(env, obs_comp)

    for episode in range(3):
        _play_one_episode(obs_comp, step_comp)

        # Start the next episode and check the first observation is valid
        bb = Blackboard()
        obs_comp.execute(bb)
        obs = bb.data.get("obs")
        assert obs is not None, f"Episode {episode+1} started with None observation"
        assert obs.shape[-2:] == (3, 3), (
            f"Episode {episode+1} observation has wrong shape: {obs.shape}"
        )
