# --- Catanatron Game Logic and Constants (from original code) ---
from catanatron.models.player import Color
from catanatron.models.map import BASE_MAP_TEMPLATE, NUM_NODES, LandTile
from catanatron.models.enums import RESOURCES, Action, ActionType
from catanatron.models.board import get_edges
import torch
from torch.distributions import Categorical
from modules.models.inference_output import InferenceOutput

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
    def __init__(self, player_class, color, **kwargs):
        # keep original initialization but we will overwrite color at decision time
        self.player = player_class(color, **kwargs)
        self.name = player_class.__name__
        # remember the initial color (not strictly necessary, but harmless)
        self.init_color = color

    def obs_inference(self, obs: torch.Tensor, **kwargs) -> InferenceOutput:
        """
        Standardized inference interface for CatanPlayerWrapper.
        Requires 'adapter' in kwargs to access the environment's game state.
        """
        adapter = kwargs.get("adapter")
        if adapter is None:
            raise ValueError("CatanPlayerWrapper requires 'adapter' in obs_inference kwargs")
        
        # PettingZoo's OrderEnforcingWrapper blocks unknown attrs like `game`.
        # Use the base env for direct Catan state access.
        # GymAdapter/PettingZooAdapter usually have .env
        env = getattr(adapter, "env", None)
        if env is None:
            # Fallback for other adapter types
            env = getattr(adapter, "vec_env", None)
            
        game = env.unwrapped.game

        # Ensure the player's color matches the current game state
        self.player.color = game.state.current_color()

        # Decide using the wrapped catanatron player
        action = self.player.decide(game, game.playable_actions)

        # Convert to action-space integer
        action_int = to_action_space(action)
        
        batch_size = obs.shape[0]
        probs = torch.zeros((batch_size, ACTION_SPACE_SIZE), device=obs.device)
        probs[:, action_int] = 1.0
        
        return InferenceOutput(
            policy=Categorical(probs=probs),
            value=torch.zeros((batch_size,), device=obs.device)
        )
