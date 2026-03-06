import pytest
import numpy as np
from gymnasium.spaces import Box, Discrete

from utils.wrappers import RecordVideo


class MockAECEnv:
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self):
        self.possible_agents = ["player_0"]
        self._reset_state()

    @property
    def unwrapped(self):
        return self

    def _reset_state(self):
        self.agents = ["player_0"]
        self.agent_selection = "player_0"
        self.rewards = {"player_0": 0.0}
        self.terminations = {"player_0": False}
        self.truncations = {"player_0": False}
        self.infos = {"player_0": {"legal_moves": [0]}}
        self._cumulative_rewards = {"player_0": 0.0}

    def observation_space(self, agent):
        return Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

    def action_space(self, agent):
        return Discrete(1)

    def observe(self, agent):
        return np.array([0.0], dtype=np.float32)

    def last(self, observe=False):
        return None, 0.0, False, False, self.infos[self.agent_selection]

    def reset(self, seed=None, options=None):
        self._reset_state()

    def step(self, action):
        pass

    def render(self):
        return np.zeros((4, 4, 3), dtype=np.uint8)

    def close(self):
        pass


@pytest.mark.regression
def test_regression_attribute_error(tmp_path):
    """
    RecordVideo.last() used to raise AttributeError through wrapper chains.
    This regression ensures last() returns the expected 5-tuple.
    """
    env = MockAECEnv()
    wrapped = RecordVideo(
        env,
        video_folder=str(tmp_path / "regression_videos"),
        episode_trigger=lambda _: False,
    )

    result = None
    try:
        wrapped.reset()
        result = wrapped.last()
    except AttributeError as err:
        pytest.fail(f"RecordVideo.last raised AttributeError: {err}")
    finally:
        wrapped.close()

    assert isinstance(result, tuple)
    assert len(result) == 5
