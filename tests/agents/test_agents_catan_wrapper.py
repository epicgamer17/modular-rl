import pytest
import torch
from agents.catan_player_wrapper import (
    CatanPlayerWrapper,
    normalize_action,
    to_action_space,
)
from catanatron.models.enums import Action, ActionType

pytestmark = pytest.mark.unit


# --- Pure Python Dummies (No MagicMock) ---
class DummyState:
    def current_color(self):
        return "BLUE"

    @property
    def playable_actions(self):
        return [Action("BLUE", ActionType.ROLL, None)]


class DummyGame:
    def __init__(self):
        self.state = DummyState()

    @property
    def playable_actions(self):
        return self.state.playable_actions


class DummyUnwrappedEnv:
    def __init__(self):
        self.game = DummyGame()


class DummyEnv:
    def __init__(self):
        self.unwrapped = DummyUnwrappedEnv()


class DummyPlayer:
    def __init__(self, color):
        self.color = color

    def decide(self, game, playable_actions):
        # Always return deterministic action
        return Action(self.color, ActionType.ROLL, None)


def test_catan_wrapper_predict_passthrough():
    """Verifies predict returns the inputs directly including the env."""
    wrapper = CatanPlayerWrapper(DummyPlayer, "RED")
    env = DummyEnv()
    obs, info, ret_env = wrapper.predict("obs", "info", env=env)

    assert obs == "obs"
    assert info == "info"
    assert ret_env is env


def test_catan_wrapper_select_actions():
    """Verifies correct color alignment, state injection, and action-int mapping."""
    wrapper = CatanPlayerWrapper(DummyPlayer, "RED")
    env = DummyEnv()

    # Wrapper extracts env from prediction[2]
    prediction = ("obs", "info", env)

    action_tensor = wrapper.select_actions(prediction, "info")

    # Verify the wrapper forcibly updated the internal player's color
    assert wrapper.player.color == "BLUE"

    # Verify int tensor mapping
    expected_int = to_action_space(Action("BLUE", ActionType.ROLL, None))
    assert isinstance(action_tensor, torch.Tensor)
    assert action_tensor.item() == expected_int


def test_catan_wrapper_normalize_action():
    """Verifies Catanatron action stripping logic."""
    # ROLL should safely strip extra nested values
    action1 = Action("RED", ActionType.ROLL, "some_value")
    norm1 = normalize_action(action1)
    assert norm1.value is None

    # BUILD_ROAD should sort the edge tuples
    action2 = Action("RED", ActionType.BUILD_ROAD, (3, 1))
    norm2 = normalize_action(action2)
    assert norm2.value == (1, 3)
