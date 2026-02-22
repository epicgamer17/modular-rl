# --- Catanatron Game Logic and Constants (from original code) ---
from catanatron.models.player import Color
from catanatron.models.map import BASE_MAP_TEMPLATE, NUM_NODES, LandTile
from catanatron.models.enums import RESOURCES, Action, ActionType
from catanatron.models.board import get_edges
import torch

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
    # (Function implementation from the original code)
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
    normalized = normalize_action(action)
    return ACTIONS_ARRAY.index((normalized.action_type, normalized.value))


def from_action_space(action_int, playable_actions):
    (action_type, value) = ACTIONS_ARRAY[action_int]
    for action in playable_actions:
        normalized = normalize_action(action)
        if normalized.action_type == action_type and normalized.value == value:
            return action
    raise ValueError(f"Action {action_int} not found in playable_actions")


HIGH = 19 * 5


class CatanPlayerWrapper:
    def __init__(self, player_class, color):
        # keep original initialization but we will overwrite color at decision time
        self.player = player_class(color)
        self.name = player_class.__name__
        # remember the initial color (not strictly necessary, but harmless)
        self.init_color = color

    def predict(self, observation, info, env=None, *args, **kwargs):
        # pass through; env will be available in select_actions via prediction[2]
        return observation, info, env

    def select_actions(self, prediction, info, *args, **kwargs):
        """
        prediction[2] is expected to be the env (per your usage).
        We fetch the game from the env and ensure the wrapped player's color
        equals the game's current_color() before decision/search.
        """
        # Unpack env from the prediction (per your predict() return)
        env = prediction[2]
        game = env.game

        # IMPORTANT: set the player's color to the game's current color so
        # the AlphaBeta (or other search) uses the correct color in its logic.
        # This ensures correctness even when the environment rotated color->agent mapping.
        self.player.color = game.state.current_color()

        # Now call the player's decision routine using the real game state.
        action = self.player.decide(game, game.state.playable_actions)

        # Convert to action-space integer
        action_int = to_action_space(action)

        # Return as a tensor (same as your original wrapper)
        return torch.tensor(action_int)
