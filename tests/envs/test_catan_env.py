import pytest
try:
    import custom_gym_envs
    from custom_gym_envs.envs.catan import CatanAECEnv
except (ImportError, ModuleNotFoundError):
    pytest.skip("custom_gym_envs or catan env not found", allow_module_level=True)
import numpy as np
import torch
import gymnasium as gym
import unittest.mock as mock
import math
from custom_gym_envs.envs.catan import (
    CatanAECEnv,
    ACTION_SPACE_SIZE,
    normalize_action,
    compute_port_coordinates,
    from_action_space,
    to_action_space,
    env as aec_env_factory,
    parallel_env as parallel_env_factory,
)
from catanatron.models.enums import ActionType, SETTLEMENT, CITY, ROAD
from catanatron.models.player import Color
from catanatron.models.actions import Action
from pettingzoo.utils.env import AECEnv
from pettingzoo.utils.conversions import ParallelEnv

pytestmark = pytest.mark.unit


def test_catan_env_initialization():
    env = CatanAECEnv()
    assert len(env.possible_agents) == 2
    assert env.action_space("player_0").n == ACTION_SPACE_SIZE
    env.close()


def test_catan_env_reset_and_basic_step():
    env = CatanAECEnv()
    env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.last()
    agent = env.agent_selection
    # Take a legal step
    legal_moves = np.where(obs["action_mask"] == 1)[0]
    env.step(legal_moves[0])
    env.close()


def test_catan_env_normalization_coverage():
    # DISCARD
    a = Action(Color.RED, ActionType.DISCARD, {"WOOD": 1})
    n = normalize_action(a)
    assert n.value is None

    # MOVE_ROBBER
    a = Action(Color.RED, ActionType.MOVE_ROBBER, (1, (0, 0, 0)))
    n = normalize_action(a)
    assert n.value == 1

    # BUY_DEV
    a = Action(Color.RED, ActionType.BUY_DEVELOPMENT_CARD, "KNIGHT")
    n = normalize_action(a)
    assert n.value is None

    # ROLL
    a = Action(Color.RED, ActionType.ROLL, (3, 4))
    n = normalize_action(a)
    assert n.value is None

    # BUILD_ROAD
    a = Action(Color.RED, ActionType.BUILD_ROAD, (2, 1))
    n = normalize_action(a)
    assert n.value == (1, 2)


def test_catan_env_to_from_action_space():
    env = CatanAECEnv()
    env.reset()
    action = list(env.game.playable_actions)[0]
    idx = to_action_space(action)
    action_back = from_action_space(idx, env.game.playable_actions)
    assert action_back.action_type == action.action_type

    with pytest.raises(ValueError):
        from_action_space(0, [])
    env.close()


def test_catan_env_compute_port_coords_coverage():
    game_map = mock.Mock()
    port = mock.Mock()
    port.direction = "N"
    port.nodes = {"a_ref": 1, "b_ref": 2}
    game_map.ports_by_id = {"p1": port}

    with mock.patch(
        "custom_gym_envs.envs.catan.PORT_DIRECTION_TO_NODEREFS",
        {"N": ("a_ref", "b_ref")},
    ):
        # Missing node
        assert compute_port_coordinates(game_map, {1: (0, 0)}) == {}

        # Dist 0 case
        with mock.patch("custom_gym_envs.envs.catan.ORIGIN", (0, 0)):
            assert compute_port_coordinates(game_map, {1: (0, 0), 2: (0, 0)}) == {
                "p1": (0, 0)
            }


def test_catan_env_auto_advance_complex():
    env = CatanAECEnv(auto_play_single_action=True)
    env.reset()

    # Path: winning_color is not None (line 544, 567)
    with (
        mock.patch.object(env.game, "execute"),
        mock.patch.object(env.game, "winning_color", side_effect=[None, Color.RED]),
    ):
        action = mock.Mock()
        action.action_type = ActionType.ROLL
        with mock.patch.object(env.game, "playable_actions", [action]):
            env._auto_advance()
            assert any(env.terminations.values())

    # Path: is_truncated (line 574)
    env.reset()
    with (
        mock.patch.object(env.game, "execute"),
        mock.patch.object(env.game, "winning_color", return_value=None),
        mock.patch.object(env.game.state, "num_turns", 10000),
    ):
        action = mock.Mock()
        action.action_type = ActionType.ROLL
        with mock.patch.object(env.game, "playable_actions", [action]):
            env._auto_advance()
            assert any(env.truncations.values())

    # Path: game is None (line 539)
    env.game = None
    env._auto_advance()

    env.close()


def test_catan_env_bandit_mode_and_reprs():
    for rep in ["image", "vector", "mixed"]:
        env = CatanAECEnv(representation=rep, bandit_mode=True)
        env.reset()
        obs = env.observe("player_0")
        assert "observation" in obs

        # Trigger bandit win check (line 717)
        with mock.patch(
            "custom_gym_envs.envs.catan.player_num_resource_cards", return_value=15
        ):
            env.step(obs["action_mask"].nonzero()[0][0])
            assert any(env.terminations.values())
        env.close()


def test_catan_env_bandit_mode_warnings():
    import unittest.mock as mock

    env = CatanAECEnv(bandit_mode=True)
    env.reset()

    # Trigger Discard warning (line 724)
    with mock.patch("custom_gym_envs.envs.catan.from_action_space") as mock_from:
        action = mock.Mock()
        action.action_type = ActionType.DISCARD
        mock_from.return_value = action
        env.step(0)

    # Trigger Robber warning (line 728)
    with mock.patch("custom_gym_envs.envs.catan.from_action_space") as mock_from:
        action = mock.Mock()
        action.action_type = ActionType.MOVE_ROBBER
        mock_from.return_value = action
        env.step(0)
    env.close()


def test_catan_env_rendering_full_coverage():
    import pygame

    with (
        mock.patch("pygame.display.set_mode"),
        mock.patch("pygame.time.Clock"),
        mock.patch("pygame.font.SysFont"),
        mock.patch("pygame.draw.polygon"),
        mock.patch("pygame.draw.line"),
        mock.patch("pygame.draw.circle"),
        mock.patch("pygame.draw.rect"),
        mock.patch("pygame.display.flip"),
        mock.patch("pygame.event.get", return_value=[]),
    ):

        env = CatanAECEnv(render_mode="human")
        env.reset()

        # Manually add roads and buildings to state for rendering coverage
        color = Color.RED
        env.game.state.board.roads[(1, 2)] = color
        env.game.state.board.buildings[1] = (SETTLEMENT, color)
        env.game.state.board.buildings[2] = (CITY, color)

        # Mocking coordinates as REAL INTS to avoid mock failures in int() calls
        env.tile_coords = {env.game.state.board.robber_coordinate: (100, 100)}
        env.node_coords = {1: (100, 100), 2: (200, 200)}
        env.edge_coords = {(1, 2): ((100, 100), (200, 200))}
        env.port_coords = {"p1": (150, 150)}

        # Hit length == 0 in road drawing (line 920)
        env.edge_coords[(3, 4)] = ((100, 100), (100, 100))
        env.game.state.board.roads[(3, 4)] = color

        # Hit None tile in loop (line 864)
        orig_tiles = env.game.state.board.map.land_tiles
        # We need a tile that has an ID that is in TILES_COORDINATES
        from custom_gym_envs.envs.catan import TILES_COORDINATES

        tile_id = next(iter(TILES_COORDINATES.keys()))

        # We mock board.map.land_tiles instead of using dictionary patching which failed before
        mock_map = mock.Mock()
        mock_map.land_tiles = {
            (0, 0): None
        }  # This will trigger line 840 map_tile is None check
        mock_map.tiles_by_id = {}  # Prevent iteration of unconfigured Mock
        with mock.patch.object(env.game.state.board, "map", mock_map):
            env.render()

        # Hit rainbow square (line 1181)
        env._draw_rainbow_square(0, 0, 10)

        # Hit _draw_hexagon (line 1167)
        env._draw_hexagon(0, 0, 10, (1, 2, 3))

        env.close()


def test_catan_env_step_extra():
    env = CatanAECEnv(render_mode="human")
    import pygame

    with (
        mock.patch("pygame.display.set_mode"),
        mock.patch("pygame.font.SysFont"),
        mock.patch("pygame.time.Clock"),
        mock.patch("pygame.draw.polygon"),
        mock.patch("pygame.draw.circle"),
        mock.patch("pygame.draw.line"),
        mock.patch("pygame.draw.rect"),
        mock.patch("pygame.display.flip"),
        mock.patch("pygame.event.get", return_value=[]),
    ):
        env.reset()
        # line 648: No agents
        env.agents = []
        env.step(0)

        # line 649: Action is None (skip)
        env.agents = ["player_0", "player_1"]
        env.step(None)

        # line 700: Winner check inside step
        with mock.patch.object(env.game, "winning_color", return_value=Color.RED):
            env.step(env.observe(env.agent_selection)["action_mask"].nonzero()[0][0])
            assert any(env.terminations.values())

        # hit line 693: is_truncated in step
        env.reset()
        with mock.patch.object(env.game.state, "num_turns", 10000):
            env.step(env.observe(env.agent_selection)["action_mask"].nonzero()[0][0])
            assert any(env.truncations.values())

    env.close()


def test_catan_env_factories():
    ae_env = aec_env_factory()
    assert isinstance(ae_env, AECEnv)

    p_env = parallel_env_factory()
    assert isinstance(p_env, ParallelEnv)


def test_catan_env_observe_extra():
    env = CatanAECEnv()
    env.reset()
    # Hit line 513: _HAS_ROLLED
    # Use a real string key if possible, check how it's formed in observe
    # line 512: key = f"{agent_color.name}"
    env.game.state.player_state["RED_HAS_ROLLED"] = True
    obs = env.observe("player_0")
    assert obs["observation"] is not None
    env.close()


def test_catan_env_rollout():
    env = CatanAECEnv(map_type="MINI", vps_to_win=3)
    env.reset()
    for _ in range(50):
        if any(env.terminations.values()):
            break
        obs, r, term, trunc, info = env.last()
        if term or trunc:
            env.step(None)
        else:
            env.step(np.random.choice(info["legal_moves"]))
    env.close()
