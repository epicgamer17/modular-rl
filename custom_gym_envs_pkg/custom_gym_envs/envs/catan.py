import functools
from typing import Dict, Tuple, TypedDict, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector

import os
import pygame
import math

# --- Catanatron Game Logic and Constants (from original code) ---
from catanatron.game import Game, TURNS_LIMIT
from catanatron.models.player import Color, Player
from catanatron.models.map import (
    BASE_MAP_TEMPLATE,
    NUM_NODES,
    LandTile,
    build_map,
    Port,
    PORT_DIRECTION_TO_NODEREFS,
)
from catanatron.models.enums import (
    RESOURCES,
    Action,
    ActionType,
    CITY,
    SETTLEMENT,
    WOOD,
    BRICK,
    SHEEP,
    WHEAT,
    ORE,
)
from catanatron.models.board import get_edges
from catanatron.features import (
    create_sample,
    get_feature_ordering,
)
from catanatron.gym.board_tensor_features import (
    create_board_tensor,
    get_channels,
    is_graph_feature,
    get_spatial_dims,
    SpatialEncoding,
)

from catanatron.state_functions import (
    build_city,
    build_road,
    build_settlement,
    buy_dev_card,
    maintain_longest_road,
    play_dev_card,
    player_can_afford_dev_card,
    player_can_play_dev,
    player_clean_turn,
    player_freqdeck_add,
    player_deck_draw,
    player_deck_random_draw,
    player_deck_replenish,
    player_freqdeck_subtract,
    player_deck_to_array,
    player_key,
    player_num_resource_cards,
    player_resource_freqdeck_contains,
    get_visible_victory_points,
    get_actual_victory_points,
    get_played_dev_cards,
    get_player_freqdeck,
    get_largest_army,
    get_longest_road_color,
    get_longest_road_length,
    get_dev_cards_in_hand,
)

from catanatron.models.decks import (
    CITY_COST_FREQDECK,
    DEVELOPMENT_CARD_COST_FREQDECK,
    SETTLEMENT_COST_FREQDECK,
    draw_from_listdeck,
    freqdeck_add,
    freqdeck_can_draw,
    freqdeck_contains,
    freqdeck_draw,
    freqdeck_from_listdeck,
    freqdeck_replenish,
    freqdeck_subtract,
    starting_devcard_bank,
    starting_resource_bank,
    freqdeck_count,
)


# --- Coordinates for rendering (from catanatron package) ---
import math
from collections import OrderedDict

# --- parameters ---
HEX_SIZE = 50
RECTANGLE_WIDTH = 5
ORIGIN = (320, 360)
BOARD_RADIUS = 2  # 19-tile board

sqrt3 = math.sqrt(3)


# axial -> pixel for pointy-top hexes (same as your old formula)
def axial_to_pixel(q, r, size=HEX_SIZE, origin=ORIGIN):
    x = size * sqrt3 * (q + r / 2.0)
    y = size * 3.0 / 2.0 * r
    ox, oy = origin
    return (ox + x, oy + y)


def hex_corners(cx, cy, size=HEX_SIZE):
    """Return 6 corners of a pointy-top hex centered at (cx, cy).
    We will reorder them later to be clockwise starting from the top."""
    pts = []
    for i in range(6):
        angle_deg = -90 + 60 * i  # same as your original angles
        rad = math.radians(angle_deg)
        pts.append((cx + size * math.cos(rad), cy + size * math.sin(rad)))
    return pts


def _clockwise_corners_from_top(cx, cy, corners):
    """
    Reorder corners to be clockwise, starting from the top corner.
    """
    # Compute angle for each corner relative to center
    # atan2 gives CCW from +x, we want clockwise, so invert
    corner_angles = []
    for p in corners:
        dx = p[0] - cx
        dy = p[1] - cy
        angle_deg = math.degrees(math.atan2(dy, dx))  # CCW from +x
        angle_cw = (angle_deg + 90) % 360  # CW from +x
        corner_angles.append((p, angle_cw))

    # Sort clockwise by angle
    corner_angles.sort(key=lambda t: t[1])

    # Find the corner closest to top (y < center)
    best_idx = min(
        range(6),
        key=lambda i: min(
            abs(corner_angles[i][1] - 0.0), 360.0 - abs(corner_angles[i][1] - 0.0)
        ),
    )
    rotated = corner_angles[best_idx:] + corner_angles[:best_idx]
    return [p for p, _ in rotated]


# --- generate tile axial coords (inside board radius) ---
tiles_axial = []
for q in range(-BOARD_RADIUS, BOARD_RADIUS + 1):
    for r in range(-BOARD_RADIUS, BOARD_RADIUS + 1):
        s = -q - r
        if -BOARD_RADIUS <= s <= BOARD_RADIUS:
            tiles_axial.append((q, r))

# compute pixel centers and metadata for sorting: (distance from center, clockwise angle)
tiles_with_meta = []
for q, r in tiles_axial:
    px, py = axial_to_pixel(q, r)
    s = -q - r
    distance = max(abs(q), abs(r), abs(s))

    # Calculate angle in degrees using atan2
    angle_deg = math.degrees(math.atan2(py - ORIGIN[1], px - ORIGIN[0]))

    # FIX: angle_cw is the Clockwise angle from +x (Right), ranging from [0, 360).
    # Since y is down in screen space, atan2 already gives angles CW from +x,
    # but in the range (-180, 180]. We convert it to a clean [0, 360) range.
    angle_cw = (angle_deg + 360) % 360.0

    tiles_with_meta.append(((q, r), (px, py), distance, angle_cw))

# sort: center first (distance=0), then by clockwise angle for each ring
# FIX: Since angle_cw now correctly ranges from 0 (Right) to 360 (just before Right)
# by increasing CLOCKWISE, sorting by t[3] in ASCENDING order produces the
# desired Clockwise spiral starting at 0 degrees (Right).
tiles_with_meta.sort(key=lambda t: (t[2], t[3]))
tiles_centers = [((q, r), (px, py)) for (q, r), (px, py), _, _ in tiles_with_meta]

# Build TILES_COORDINATES so tile 0 is center, tile1 is right, tile2 next clockwise, etc.
TILES_COORDINATES = OrderedDict()
for i, ((q, r), (x, y)) in enumerate(tiles_centers):
    TILES_COORDINATES[i] = (int(round(x)), int(round(y)))

# --- generate nodes and edges ---
node_map = {}  # maps (x_int, y_int) -> node id
nodes = []  # list of (nid, (x_int, y_int)) in creation order
edges = set()
nid = 0

# iterate tiles in TILES_COORDINATES order (ensures tile indexing is the spiral clockwise you asked for)
for tile_index in TILES_COORDINATES.keys():
    (q, r), (cx_f, cy_f) = tiles_centers[tile_index]
    raw_corners = hex_corners(cx_f, cy_f)
    corners = _clockwise_corners_from_top(cx_f, cy_f, raw_corners)

    corner_ids = []
    for x_f, y_f in corners:
        key = (int(round(x_f)), int(round(y_f)))
        if key not in node_map:
            node_map[key] = nid
            nodes.append((nid, key))
            nid += 1
        corner_ids.append(node_map[key])

    # add edges between consecutive corners (close the hex by mod 6)
    for i in range(6):
        a = corner_ids[i]
        b = corner_ids[(i + 1) % 6]
        edges.add(tuple(sorted((a, b))))

# deterministic ordered dicts
NODES_COORDINATES = OrderedDict((nid, coord) for nid, coord in nodes)
sorted_edges = sorted(edges)
EDGES_COORDINATES = OrderedDict(
    ((a, b), [NODES_COORDINATES[a], NODES_COORDINATES[b]]) for a, b in sorted_edges
)

# numbers slightly below tile center
NUMBERS_COORDINATES = {
    i: (int(round(x)), int(round(y + HEX_SIZE * 0.15)))
    for i, (x, y) in TILES_COORDINATES.items()
}


def compute_port_coordinates(catan_map, NODES_COORDINATES, offset_dist=25):
    """Compute port coordinates for rendering."""
    port_coords = {}

    for port_id, port in catan_map.ports_by_id.items():
        # Get the two noderefs that define this port's edge
        a_ref, b_ref = PORT_DIRECTION_TO_NODEREFS[port.direction]
        a = port.nodes[a_ref]
        b = port.nodes[b_ref]

        # Get the (x, y) coordinates of those two nodes
        if a not in NODES_COORDINATES or b not in NODES_COORDINATES:
            continue  # skip ports missing node coordinates (shouldn't happen)
        x1, y1 = NODES_COORDINATES[a]
        x2, y2 = NODES_COORDINATES[b]

        # Midpoint of the edge (port position base)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2

        # Compute outward offset — ports are drawn slightly outside the hex border
        # Direction vector from map origin to midpoint
        dx, dy = mx - ORIGIN[0], my - ORIGIN[1]
        dist = math.sqrt(dx**2 + dy**2)
        if dist == 0:
            dist = 1
        ux, uy = dx / dist, dy / dist

        # Apply offset outward
        px = mx + ux * offset_dist
        py = my + uy * offset_dist

        port_coords[port_id] = (px, py)

    return port_coords


# --- summary dicts ---
TILES_COORDINATES = dict(TILES_COORDINATES)
NODES_COORDINATES = dict(NODES_COORDINATES)
EDGES_COORDINATES = dict(EDGES_COORDINATES)
NUMBERS_COORDINATES = dict(NUMBERS_COORDINATES)

BASE_TOPOLOGY = BASE_MAP_TEMPLATE.topology
TILE_COORDINATES = [x for x, y in BASE_TOPOLOGY.items() if y == LandTile]
ACTIONS_ARRAY = [
    (ActionType.ROLL, None),
    *[(ActionType.MOVE_ROBBER, tile) for tile in TILE_COORDINATES],
    (ActionType.DISCARD, None),
    *[(ActionType.BUILD_ROAD, tuple(sorted(edge))) for edge in get_edges()],
    *[(ActionType.BUILD_SETTLEMENT, node_id) for node_id in range(NUM_NODES)],
    *[(ActionType.BUILD_CITY, node_id) for node_id in range(NUM_NODES)],
    (ActionType.BUY_DEVELOPMENT_CARD, None),
    (ActionType.PLAY_KNIGHT_CARD, None),
    *[
        (ActionType.PLAY_YEAR_OF_PLENTY, (first_card, RESOURCES[j]))
        for i, first_card in enumerate(RESOURCES)
        for j in range(i, len(RESOURCES))
    ],
    *[(ActionType.PLAY_YEAR_OF_PLENTY, (first_card,)) for first_card in RESOURCES],
    (ActionType.PLAY_ROAD_BUILDING, None),
    *[(ActionType.PLAY_MONOPOLY, r) for r in RESOURCES],
    *[
        (ActionType.MARITIME_TRADE, tuple(4 * [i] + [j]))
        for i in RESOURCES
        for j in RESOURCES
        if i != j
    ],
    *[
        (ActionType.MARITIME_TRADE, tuple(3 * [i] + [None, j]))
        for i in RESOURCES
        for j in RESOURCES
        if i != j
    ],
    *[
        (ActionType.MARITIME_TRADE, tuple(2 * [i] + [None, None, j]))
        for i in RESOURCES
        for j in RESOURCES
        if i != j
    ],
    (ActionType.END_TURN, None),
]
ACTION_SPACE_SIZE = len(ACTIONS_ARRAY)


def normalize_action(action):
    normalized = action
    if normalized.action_type == ActionType.ROLL:
        return Action(action.color, action.action_type, None)
    elif normalized.action_type == ActionType.MOVE_ROBBER:
        return Action(action.color, action.action_type, action.value[0])
    elif normalized.action_type == ActionType.BUILD_ROAD:
        return Action(action.color, action.action_type, tuple(sorted(action.value)))
    elif normalized.action_type == ActionType.BUY_DEVELOPMENT_CARD:
        return Action(action.color, action.action_type, None)
    elif normalized.action_type == ActionType.DISCARD:
        return Action(action.color, action.action_type, None)
    return normalized


def to_action_space(action):
    # print(action)
    normalized = normalize_action(action)
    # print(normalized)
    return ACTIONS_ARRAY.index((normalized.action_type, normalized.value))


def from_action_space(action_int, playable_actions):
    (action_type, value) = ACTIONS_ARRAY[action_int]
    for action in playable_actions:
        normalized = normalize_action(action)
        if normalized.action_type == action_type and normalized.value == value:
            return action
    raise ValueError(f"Action {action_int} not found in playable_actions")


HIGH = 19 * 5


class CatanAECEnv(AECEnv):
    """
    A PettingZoo AEC environment for the game of Catan.
    This environment supports multiple agents, turn-based gameplay,
    and provides action masks for legal moves.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "catanatron_v1",
        "is_parallelizable": False,
        "render_fps": 30,
    }

    def __init__(
        self,
        render_mode=None,
        num_players=2,
        map_type="BASE",
        vps_to_win=10,
        representation="image",
        invalid_action_reward=-1,
        auto_play_single_action: bool = False,
        bandit_mode: bool = False,
        spatial_encoding: SpatialEncoding = "axial",
        include_validity_mask: bool = True,
        include_last_roll: bool = True,
        include_game_phase: bool = True,
        include_bank_state: bool = True,
        include_road_distance: bool = True,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.screen_width = 900
        self.screen_height = 700

        self._pygame_initialized = False
        self._pygame_clock = None
        self.screen = None
        self.font = None
        self.font_bold = None
        self.pygame_colors = {}
        self.tile_colors = {}
        self.node_coords = NODES_COORDINATES
        self.edge_coords = EDGES_COORDINATES
        self.tile_coords = TILES_COORDINATES
        self.number_coords = NUMBERS_COORDINATES
        self.board_pos = (0, 0)
        self.info_panel_x_start = 630

        # Game configuration
        self.map_type = map_type
        self.vps_to_win = vps_to_win
        self.representation = representation
        self.invalid_action_reward = invalid_action_reward
        self.spatial_encoding = spatial_encoding
        self.include_validity_mask = include_validity_mask
        self.include_last_roll = include_last_roll
        self.include_game_phase = include_game_phase
        self.include_bank_state = include_bank_state
        self.include_road_distance = include_road_distance
        assert self.representation in ["mixed", "image", "vector"]
        assert 2 <= num_players <= 4, "Catan must be played with 2 to 4 players"

        self.auto_play_single_action = bool(auto_play_single_action)

        # Agent setup
        self.possible_agents = [f"player_{i}" for i in range(num_players)]

        # --- FIX: Initialize PettingZoo Attributes in __init__ ---
        self.agents = self.possible_agents[:]
        self.agent_selection = self.possible_agents[0]

        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        # ---------------------------------------------------------

        self.color_map = {
            agent: list(Color)[i] for i, agent in enumerate(self.possible_agents)
        }
        self.agent_map = {color: agent for agent, color in self.color_map.items()}
        self.catan_players = [Player(color) for agent, color in self.color_map.items()]

        # Define action and observation spaces
        self._action_spaces = {
            agent: spaces.Discrete(ACTION_SPACE_SIZE) for agent in self.possible_agents
        }

        self.features = get_feature_ordering(len(self.possible_agents), self.map_type)
        self.numeric_features = [f for f in self.features if not is_graph_feature(f)]

        spatial_width, spatial_height = get_spatial_dims(self.spatial_encoding)

        if self.representation == "mixed":
            channels = get_channels(
                len(self.possible_agents),
                include_validity_mask=self.include_validity_mask,
                include_last_roll=self.include_last_roll,
                include_game_phase=self.include_game_phase,
                include_bank_state=self.include_bank_state,
                include_road_distance=self.include_road_distance,
            )
            board_tensor_space = spaces.Box(
                low=0,
                high=1,
                shape=(channels, spatial_width, spatial_height),
                dtype=np.float32,
            )
            numeric_space = spaces.Box(
                low=0, high=HIGH, shape=(len(self.numeric_features),), dtype=np.float32
            )
            core_obs_space = spaces.Dict(
                {"board": board_tensor_space, "numeric": numeric_space}
            )
        elif self.representation == "image":
            board_channels = get_channels(
                len(self.possible_agents),
                include_validity_mask=self.include_validity_mask,
                include_last_roll=self.include_last_roll,
                include_game_phase=self.include_game_phase,
                include_bank_state=self.include_bank_state,
                include_road_distance=self.include_road_distance,
            )
            total_channels = board_channels + len(self.numeric_features)
            core_obs_space = spaces.Box(
                low=0,
                high=HIGH,
                shape=(total_channels, spatial_width, spatial_height),
                dtype=np.float32,
            )
        else:
            core_obs_space = spaces.Box(
                low=0, high=HIGH, shape=(len(self.features),), dtype=np.float32
            )

        self._observation_spaces = {
            agent: spaces.Dict(
                {
                    "observation": core_obs_space,
                    "action_mask": spaces.Box(
                        low=0, high=1, shape=(ACTION_SPACE_SIZE,), dtype=np.int8
                    ),
                }
            )
            for agent in self.possible_agents
        }

        # Game state holders
        self.game: Game = None
        self.invalid_actions_count = {}
        self.max_invalid_actions = 10
        self.bandit_mode = bandit_mode

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_spaces[agent]

    def observe(self, agent):
        """Returns the observation for the specified agent."""
        agent_color = self.color_map[agent]
        core_observation = self._get_core_observation(agent_color)

        legal_moves = (
            self.game.state.playable_actions
            if self.agent_map.get(self.game.state.current_color()) == agent
            else []
        )
        action_mask = self._get_action_mask(legal_moves)

        return {"observation": core_observation, "action_mask": action_mask}

    def _auto_advance(self):
        """
        If enabled, repeatedly execute the only legal action for the current player
        while that player has exactly 1 legal action.
        """
        if not self.auto_play_single_action or self.game is None:
            return

        while True:
            winning_color = self.game.winning_color()
            if winning_color is not None:
                break

            legal_moves = list(self.game.state.playable_actions)
            if len(legal_moves) != 1:
                break

            sole_action = legal_moves[0]
            self.game.execute(sole_action)

            winning_color = self.game.winning_color()
            is_terminated = winning_color is not None
            current_agent = (
                self.agent_map[self.game.state.current_color()]
                if self.game.state.current_color() in self.agent_map
                else None
            )
            is_truncated = self.game.state.num_turns >= TURNS_LIMIT or (
                current_agent is not None
                and self.invalid_actions_count.get(current_agent, 0)
                > self.max_invalid_actions
            )

            if is_terminated:
                winner_agent = self.agent_map[winning_color]
                for agent in self.agents:
                    self.rewards[agent] = 1 if agent == winner_agent else -1
                    self.terminations[agent] = True
                self.agent_selection = self._agent_selector.next()
                break
            elif is_truncated:
                for agent in self.agents:
                    self.truncations[agent] = True
                self.agent_selection = self._agent_selector.next()
                break
            else:
                self.agent_selection = self.agent_map[self.game.state.current_color()]

    def reset(self, seed=None, options=None):
        """Resets the environment to a starting state."""
        catan_map = build_map(self.map_type)
        for player in self.catan_players:
            player.reset_state()

        self.game = Game(
            players=self.catan_players,
            seed=seed,
            catan_map=catan_map,
            vps_to_win=self.vps_to_win,
        )
        self.port_coords = compute_port_coordinates(
            self.game.state.board.map, self.node_coords, offset_dist=0
        )

        self.invalid_actions_count = {agent: 0 for agent in self.possible_agents}

        # Map agents directly to the seating order (colors) decided by catanatron after the shuffle
        # this ensures player_0 is always the first player, player_1 the second, etc.
        shuffled_colors = self.game.state.colors
        self.color_map = {
            agent: shuffled_colors[i] for i, agent in enumerate(self.possible_agents)
        }
        self.agent_map = {color: agent for agent, color in self.color_map.items()}
        # Re-initialize catan_players to match the new mapping if necessary
        # However, catanatron already has the player objects, we just need to know which agent is which.
        # But for consistency with __init__:
        self.catan_players = [
            next(p for p in self.game.state.players if p.color == self.color_map[agent])
            for agent in self.possible_agents
        ]

        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self.agent_map[self.game.state.current_color()]

        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        if self.auto_play_single_action:
            self._auto_advance()

        for agent in self.agents:
            self._cumulative_rewards[agent] += self.rewards.get(agent, 0)
            self.infos[agent] = {
                "turn": self.game.state.num_turns,
                "player": self.possible_agents.index(agent),
                "legal_moves": (
                    list(self._get_valid_action_indices())
                    if agent == self.agent_selection
                    else []
                ),
            }

        if self.render_mode == "human":
            self.render()

    def step(self, action):
        """Executes one action for the current agent."""
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        current_agent = self.agent_selection
        is_action_valid = action in self._get_valid_action_indices()

        if not is_action_valid:
            self.invalid_actions_count[current_agent] += 1
            self.rewards = {agent: 0 for agent in self.agents}
            self.rewards[current_agent] = self.invalid_action_reward
        else:
            catan_action = from_action_space(action, self.game.state.playable_actions)
            self.game.execute(catan_action)
            self.rewards = {agent: 0 for agent in self.agents}

        # --- WIN CONDITION LOGIC ---
        winning_color = None

        if self.bandit_mode:
            # Custom Mode: Winner is whoever holds >= 10 resource cards first
            # We check the current player's resource count
            current_color = self.game.state.current_color()
            if current_color:
                # Note: You might need to import player_num_resource_cards or access it via state
                # Assuming player_num_resource_cards is available as imported or via helper
                num_cards = player_num_resource_cards(self.game.state, current_color)
                if num_cards >= self.vps_to_win:
                    winning_color = current_color
        else:
            # Standard Mode: Victory Points (handled by catanatron's game logic)
            winning_color = self.game.winning_color()
        # ---------------------------

        is_terminated = winning_color is not None
        is_truncated = (
            self.game.state.num_turns >= TURNS_LIMIT
            or self.invalid_actions_count[current_agent] > self.max_invalid_actions
        )

        if is_terminated:
            winner_agent = self.agent_map[winning_color]
            for agent in self.agents:
                self.rewards[agent] = 1 if agent == winner_agent else -1
                self.terminations[agent] = True
        elif is_truncated:
            for agent in self.agents:
                self.truncations[agent] = True
                self.rewards[agent] = 0  # -1

        if not (is_terminated or is_truncated):
            self.agent_selection = self.agent_map[self.game.state.current_color()]
            if self.auto_play_single_action:
                self._auto_advance()
        else:
            self.agent_selection = self._agent_selector.next()

        for agent in self.agents:
            self._cumulative_rewards[agent] += self.rewards[agent]
            self.infos[agent] = {
                "turn": self.game.state.num_turns,
                "player": self.possible_agents.index(agent),
                "legal_moves": (
                    list(self._get_valid_action_indices())
                    if agent == self.agent_selection
                    else []
                ),
            }

        if self.render_mode == "human":
            self.render()

    def _init_pygame(self):
        """Initializes pygame with default fonts. Handles headless (rgb_array) by using SDL dummy driver.
        Also aligns coordinates (hex geometry -> vertex/edge/port positions) so tiles/buildings/roads line up.
        """
        if self._pygame_initialized:
            return

        # Headless-friendly SDL driver for rgb_array
        if self.render_mode == "rgb_array" and "SDL_VIDEODRIVER" not in os.environ:
            os.environ["SDL_VIDEODRIVER"] = os.environ.get("SDL_VIDEODRIVER", "dummy")

        pygame.init()
        pygame.display.init()

        # Create onscreen window only for human mode; otherwise off-screen Surface
        if self.render_mode == "human":
            self.screen = pygame.display.set_mode(
                (self.screen_width, self.screen_height)
            )
            pygame.display.set_caption("Catanatron")
        else:
            self.screen = pygame.Surface((self.screen_width, self.screen_height))

        self._pygame_clock = pygame.time.Clock()

        # Fonts
        # NOTE: pygame.font.Font(None, size) uses the default system font
        self.font = pygame.font.Font(None, 18)
        self.font_bold = pygame.font.Font(None, 20)

        # Colors
        self.pygame_colors = {
            Color.RED: pygame.Color(228, 6, 6),
            Color.BLUE: pygame.Color(0, 128, 255),
            Color.ORANGE: pygame.Color(255, 128, 0),
            Color.WHITE: pygame.Color(230, 230, 230),
            "background": pygame.Color(245, 245, 245),
            "text": pygame.Color(30, 30, 30),
            "text_light": pygame.Color(100, 100, 100),
            "text_red": pygame.Color(228, 6, 6),
        }

        self.tile_colors = {
            SHEEP: pygame.Color(144, 238, 144),
            WOOD: pygame.Color(34, 139, 34),
            WHEAT: pygame.Color(255, 255, 0),
            ORE: pygame.Color(169, 169, 169),
            BRICK: pygame.Color(255, 140, 0),
            None: pygame.Color(245, 222, 179),
        }

        self._pygame_initialized = True

    def _draw_text(self, text, pos, color, center=False, bold=False):
        """Helper function to draw text on the screen. Accepts pygame.Color or (r,g,b) tuples."""
        # Ensure color is a pygame.Color or tuple acceptable by font.render
        if isinstance(color, tuple) and not isinstance(color, pygame.Color):
            color_val = pygame.Color(*color)
        else:
            color_val = color

        font = self.font_bold if bold else self.font
        # Ensure font exists
        if font is None:
            # fallback to pygame.font.SysFont
            font = pygame.font.SysFont(None, 18)

        # Render text; guard against None
        text_surface = font.render(str(text), True, color_val)
        if center:
            text_rect = text_surface.get_rect(center=pos)
            self.screen.blit(text_surface, text_rect)
        else:
            self.screen.blit(text_surface, pos)

    def render(self):
        """
        Renders the current game state.
        - "human" mode: Displays the game in a pygame window.
        - "rgb_array" mode: Returns a numpy array of the frame.
        """
        if self.render_mode is None:
            return

        if not self._pygame_initialized:
            self._init_pygame()

        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return

        self._render_frame()

        if self.render_mode == "human":
            pygame.display.flip()
            self._pygame_clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return pygame.surfarray.array3d(self.screen).transpose(1, 0, 2)

    def _render_frame(self):
        """Draws all game elements onto the self.screen surface. **Strict error checking.**"""
        if self.game is None or not self._pygame_initialized:
            return

        state = self.game.state
        board = state.board

        # Draw Background
        self.screen.fill(self.pygame_colors["background"])

        # Draw Tiles (hexagons)
        for tile_id, (x, y) in self.tile_coords.items():
            tile = board.map.tiles_by_id.get(tile_id)
            # If tiles_by_id fails, we now attempt the robust lookup *without* exceptions.
            # This part requires specific knowledge of the board/map structure to translate
            # the original robust logic into non-exception-based checks.

            # Simplified tile lookup to remove nested try/excepts:
            for coord, map_tile in getattr(board.map, "land_tiles", {}).items():
                if map_tile.id == tile_id:
                    tile = map_tile
                    break

            if tile is None:
                continue

            # Determine tile color based on resource (None for desert)
            color = self.tile_colors.get(
                getattr(tile, "resource", None), pygame.Color(200, 200, 200)
            )

            # Draw hexagon
            self._draw_hexagon(x, y, HEX_SIZE - RECTANGLE_WIDTH, color)
            # self._draw_text(
            #     tile_id, (x, y), self.pygame_colors["text"], center=True, bold=False
            # )

        # Draw Tile Numbers
        for tile_id, (x, y) in self.number_coords.items():
            tile = board.map.tiles_by_id.get(tile_id)

            # Simplified tile lookup for Tile Numbers
            if tile is None:
                for coord, map_tile in board.map.land_tiles.items():
                    if map_tile.id == tile_id:
                        tile = map_tile
                        break

            if (
                tile is not None and tile.resource is not None
            ):  # Don't draw number on desert
                number = tile.number
                text = str(number)
                color = (
                    self.pygame_colors["text_red"]
                    if number == 6 or number == 8
                    else self.pygame_colors["text"]
                )
                self._draw_text(text, (x, y), color, center=True, bold=True)

        # Draw Ports (robust)
        for port_id, (x, y) in self.port_coords.items():
            # print(port_id, x, y)
            pygame.draw.circle(
                self.screen,
                (200, 200, 255),
                (int(x), int(y)),
                10,
            )
            port = board.map.ports_by_id.get(port_id)
            if port is None:
                continue
            text = port.resource if port.resource else "3:1"
            self._draw_text(
                text,
                (x, y),
                self.pygame_colors["text"],
                center=True,
                bold=True,
            )

        # Draw Roads
        for edge_tuple, color in board.roads.items():
            # color is a Color enum; convert to pygame color if present
            pygame_color = self.pygame_colors.get(color, pygame.Color(0, 0, 0))
            edge_key = tuple(sorted(edge_tuple))

            if edge_key in self.edge_coords:
                p1_x, p1_y = self.edge_coords[edge_key][0]
                p2_x, p2_y = self.edge_coords[edge_key][1]

                # Calculate the vector from p1 to p2
                dx = p2_x - p1_x
                dy = p2_y - p1_y

                # Calculate the length and the normalized perpendicular vector
                length = math.hypot(dx, dy)

                if length == 0:
                    continue

                # Calculate for the main rectangle
                half_width = RECTANGLE_WIDTH / 2
                px = (dy / length) * half_width
                py = (-dx / length) * half_width

                # Main rectangle corner points
                c1 = (p1_x + px, p1_y + py)
                c2 = (p1_x - px, p1_y - py)
                c3 = (p2_x - px, p2_y - py)
                c4 = (p2_x + px, p2_y + py)

                rectangle_points = [c1, c4, c3, c2]

                # Draw the main colored rectangle (filled)
                pygame.draw.polygon(self.screen, pygame_color, rectangle_points)

                # --- Add the black border ---
                # Calculate for the border. The border's total width will be
                # RECTANGLE_WIDTH + (2 * BORDER_THICKNESS).
                # So, the half_width for the border calculation should be:
                border_half_width = (RECTANGLE_WIDTH + (2 * 2)) / 2

                border_px = (dy / length) * border_half_width
                border_py = (-dx / length) * border_half_width

                # Border corner points (slightly expanded)
                bc1 = (p1_x + border_px, p1_y + border_py)
                bc2 = (p1_x - border_px, p1_y - border_py)
                bc3 = (p2_x - border_px, p2_y - border_py)
                bc4 = (p2_x + border_px, p2_y + border_py)

                border_points = [bc1, bc4, bc3, bc2]

                # Draw the black border (outline only, by setting width=BORDER_THICKNESS)
                # Note: We draw the border *after* the filled rectangle.
                # This will draw an outline around the existing rectangle.
                pygame.draw.polygon(
                    self.screen, pygame.Color(0, 0, 0), border_points, 2
                )
        # for node_id, (x, y) in self.node_coords.items():
        #     self._draw_text(
        #         node_id, (x, y), self.pygame_colors["text"], center=True, bold=False
        #     )

        # Draw Settlements and Cities
        for node_id, (color, building_type) in board.buildings.items():
            pygame_color = self.pygame_colors.get(color, pygame.Color(0, 0, 0))
            if node_id not in self.node_coords:
                continue
            x, y = self.node_coords[node_id]

            if building_type == CITY:
                # City - rectangle with black fill
                width = 24
                height = 16
                rect = (x - width // 2, y - height // 2, width, height)
                pygame.draw.rect(self.screen, (0, 0, 0), rect)
                pygame.draw.rect(self.screen, pygame_color, rect, width=3)
            else:  # SETTLEMENT
                # Settlement - square
                size = 24
                rect = (x - size // 2, y - size // 2, size, size)
                pygame.draw.rect(self.screen, pygame_color, rect)
                pygame.draw.rect(self.screen, (0, 0, 0), rect, width=2)

        # Draw Robber (gray square) - more robust lookup
        robber_coord = board.robber_coordinate
        robber_tile = None
        # Try tiles_by_id first (tile objects with coordinates in tile_coords)
        for t_id, tile in board.map.tiles_by_id.items():
            # some tile objects may carry coordinate attribute or id; we check tile.id if present
            t_id_attr = tile.id
            # If robber_coord is a coordinate tuple, we must compare via land_tiles mapping if present
            # Try common pattern: board.map.land_tiles : { coord -> map_tile }
            found = False
            for coord, map_tile in board.map.land_tiles.items():
                if map_tile.id == t_id_attr and coord == robber_coord:
                    robber_tile = map_tile  # Use map_tile which has the coordinate
                    found = True
                    break
            if found:
                break

        if robber_tile and robber_tile.id in self.tile_coords:
            x, y = self.tile_coords[robber_tile.id]
            robber_size = 30
            pygame.draw.rect(
                self.screen,
                pygame.Color(128, 128, 128),
                (
                    x - robber_size // 2,
                    y - robber_size // 2,
                    robber_size,
                    robber_size,
                ),
            )
            pygame.draw.rect(
                self.screen,
                pygame.Color(0, 0, 0),
                (
                    x - robber_size // 2,
                    y - robber_size // 2,
                    robber_size,
                    robber_size,
                ),
                width=2,
            )

        # Draw Info Panel
        # All logic inside the original Info Panel try/except is now bare
        x = self.info_panel_x_start + 15
        y = 20

        self._draw_text(
            f"Turn: {state.num_turns}",
            (x, y),
            self.pygame_colors["text"],
            bold=True,
        )
        y += 25

        current_color = state.current_color()
        if current_color:
            current_agent = self.agent_map.get(current_color, "N/A")
            self._draw_text(
                f"Current Player:", (x, y), self.pygame_colors["text_light"]
            )
            y += 20
            self._draw_text(
                f"{current_agent}",
                (x + 10, y),
                self.pygame_colors.get(current_color, self.pygame_colors["text"]),
                bold=True,
            )
            y += 30

        last_roll = state.last_roll
        roll_text = (
            f"Roll: {sum(last_roll)} ({last_roll[0]}+{last_roll[1]})"
            if last_roll
            else "Roll: -"
        )
        self._draw_text(roll_text, (x, y), self.pygame_colors["text"])
        y += 35

        # Player Info
        for agent in self.agents:
            color = self.color_map.get(agent)

            # Search the list for the player state object matching the color.
            # Assuming the player state object has an attribute named 'color'.
            p_x, p_y = x, y
            # Header
            pygame.draw.rect(
                self.screen,
                self.pygame_colors[color],
                (p_x, p_y + 3, 15, 15),
            )
            self._draw_text(
                f"{agent}", (p_x + 22, p_y), self.pygame_colors["text"], bold=True
            )
            p_y += 25
            vp_text = f"VISIBLE VP: {get_visible_victory_points(state, color)} | ACTUAL VP: {get_actual_victory_points(state, color)}"
            self._draw_text(
                vp_text,
                (p_x, p_y),
                self.pygame_colors["text"],
                bold=True,
            )
            p_y += 25

            resource_freqdeck = get_player_freqdeck(state, color)
            res_texts = []
            for i, count in enumerate(resource_freqdeck):
                res_name = RESOURCES[i]
                res_texts.append(f"{res_name[:3]}: {count}")

            for i, text in enumerate(res_texts):
                col = i % 3
                row = i // 3
                self._draw_text(
                    text,
                    (p_x + col * 60, p_y + row * 18),
                    self.pygame_colors["text_light"],
                )
            p_y += (len(res_texts) // 3 + 1) * 18

            # --- Dev Cards & Stats ---
            # Total number of dev cards in hand
            dev_cards_count = get_dev_cards_in_hand(state, color)
            # Number of played Knights (used for Largest Army check)
            knights_played = get_played_dev_cards(state, color, "KNIGHT")

            dev_text = f"Dev Cards: {dev_cards_count}"
            knight_text = f"Knights Played: {knights_played}"
            self._draw_text(dev_text, (p_x, p_y), self.pygame_colors["text_light"])
            self._draw_text(
                knight_text, (p_x + 120, p_y), self.pygame_colors["text_light"]
            )
            p_y += 18

            # --- Road/Army Status ---
            stats = []
            # Check if this player has the Longest Road
            if get_longest_road_color(state) == color:
                stats.append("Longest Road")
            # Check if this player has the Largest Army
            if get_largest_army(state)[0] == color:
                stats.append("Largest Army")
            if stats:
                self._draw_text(
                    f"{', '.join(stats)}",
                    (p_x, p_y),
                    self.pygame_colors["text_light"],
                    bold=True,
                )
                p_y += 18

            y = p_y + 15
            # Draw separator line
            pygame.draw.line(
                self.screen,
                (220, 220, 220),
                (x, y - 8),
                (self.screen_width - 15, y - 8),
                1,
            )

        # Bank Info
        y = max(y, self.screen_height - 150)
        self._draw_text("Bank:", (x, y), self.pygame_colors["text"], bold=True)
        y += 22
        for i, res in enumerate(RESOURCES):
            self._draw_text(
                f"{res}: {freqdeck_count(state.resource_freqdeck, res)}",
                (x + (i % 2) * 100, y + (i // 2) * 20),
                self.pygame_colors["text_light"],
            )
        y += (len(RESOURCES) // 2 + 1) * 20 + 5
        self._draw_text(
            f"Dev Cards Left: {len(state.development_listdeck)}",
            (x, y),
            self.pygame_colors["text_light"],
        )

    def _draw_hexagon(self, x, y, size, color):
        """Draw a pointy-top hexagon (point facing up) centered at (x,y)."""
        angles = [-90 + 60 * i for i in range(6)]  # start at -90° => top point
        points = []
        for angle in angles:
            rad = math.radians(angle)
            px = x + size * math.cos(rad)
            py = y + size * math.sin(rad)
            points.append((int(px), int(py)))
        pygame.draw.polygon(self.screen, color, points)
        pygame.draw.polygon(self.screen, (0, 0, 0), points, 2)

    def _draw_rainbow_square(self, x, y, size):
        """Draw a rainbow colored square for 3:1 ports."""
        colors = [
            (255, 0, 0),  # Red
            (255, 127, 0),  # Orange
            (255, 255, 0),  # Yellow
            (0, 255, 0),  # Green
            (0, 0, 255),  # Blue
            (75, 0, 130),  # Indigo
            (148, 0, 211),  # Violet
        ]

        stripe_height = size // len(colors)
        for i, color in enumerate(colors):
            pygame.draw.rect(
                self.screen, color, (x, y + i * stripe_height, size, stripe_height)
            )

        # Border
        pygame.draw.rect(self.screen, (0, 0, 0), (x, y, size, size), 2)

    def close(self):
        """Closes the environment and cleans up pygame."""
        if self._pygame_initialized:
            pygame.display.quit()
            pygame.quit()
            self._pygame_initialized = False

    def _get_action_mask(self, legal_moves):
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int8)
        if not legal_moves:
            return mask
        legal_indices = [to_action_space(action) for action in legal_moves]
        mask[legal_indices] = 1
        return mask

    def _get_valid_action_indices(self):
        return {to_action_space(action) for action in self.game.state.playable_actions}

    def _get_core_observation(self, agent_color: Color) -> Union[np.ndarray, dict]:
        """Generates the core observation for a specific agent color."""
        sample = create_sample(self.game, agent_color)
        if self.representation == "mixed":
            board_tensor = create_board_tensor(
                self.game,
                agent_color,
                channels_first=True,
                spatial_encoding=self.spatial_encoding,
                include_validity_mask=self.include_validity_mask,
                include_last_roll=self.include_last_roll,
                include_game_phase=self.include_game_phase,
                include_bank_state=self.include_bank_state,
                include_road_distance=self.include_road_distance,
            ).astype(np.float32)
            numeric = np.array(
                [float(sample[i]) for i in self.numeric_features], dtype=np.float32
            )
            return {"board": board_tensor, "numeric": numeric}
        if self.representation == "image":
            board_tensor = create_board_tensor(
                self.game,
                agent_color,
                channels_first=True,
                spatial_encoding=self.spatial_encoding,
                include_validity_mask=self.include_validity_mask,
                include_last_roll=self.include_last_roll,
                include_game_phase=self.include_game_phase,
                include_bank_state=self.include_bank_state,
                include_road_distance=self.include_road_distance,
            ).astype(np.float32)

            numeric = np.array(
                [float(sample[i]) for i in self.numeric_features], dtype=np.float32
            )

            _, h, w = board_tensor.shape
            numeric_planes = np.tile(numeric[:, None, None], (1, h, w))

            return np.concatenate((board_tensor, numeric_planes), axis=0)
        else:
            return np.array([float(sample[i]) for i in self.features], dtype=np.float32)


def env(**kwargs):
    """Factory function for creating the AEC environment."""
    env_instance = CatanAECEnv(**kwargs)
    env_instance = wrappers.AssertOutOfBoundsWrapper(env_instance)
    env_instance = wrappers.OrderEnforcingWrapper(env_instance)
    return env_instance


def parallel_env(**kwargs):
    """Factory function for creating the parallel API version of the environment."""
    from pettingzoo.utils import aec_to_parallel

    aec_env_instance = env(**kwargs)
    parallel_env_instance = aec_to_parallel(aec_env_instance)
    return parallel_env_instance
