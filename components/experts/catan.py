"""
Catan Expert Component
======================
A pipeline component wrapping any ``catanatron`` player that implements
``decide(game, playable_actions)``.

This is a port of the logic from the retired
``actors/experts/catan_player_wrapper.py``.  It relies on the same
``catanatron`` action-space encoding used by the custom Catan environments.

Blackboard contract
-------------------
Reads:
    ``data["info"]`` – Dict that must contain ``"env"`` (the live PettingZoo /
        Gymnasium env with a ``game`` attribute accessible via
        ``env.unwrapped.game``).

Writes:
    ``meta["action"]`` – ``int`` action-space index into ``ACTIONS_ARRAY``.
"""

import torch
from typing import Any, Type, Set, Dict

from core import PipelineComponent, Blackboard
from core.contracts import Key, Action, SemanticType

# ---------------------------------------------------------------------------
# Action-space encoding (mirrors the original CatanPlayerWrapper constants)
# ---------------------------------------------------------------------------
from catanatron.models.player import Color  # type: ignore[import]
from catanatron.models.map import BASE_MAP_TEMPLATE, NUM_NODES, LandTile  # type: ignore[import]
from catanatron.models.enums import RESOURCES, Action, ActionType  # type: ignore[import]
from catanatron.models.board import get_edges  # type: ignore[import]

_BASE_TOPOLOGY = BASE_MAP_TEMPLATE.topology
_TILE_COORDINATES = [x for x, y in _BASE_TOPOLOGY.items() if y == LandTile]

ACTIONS_ARRAY = [
    (ActionType.ROLL, None),
    *[(ActionType.MOVE_ROBBER, tile) for tile in _TILE_COORDINATES],
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
ACTION_SPACE_SIZE: int = len(ACTIONS_ARRAY)


def _normalize_action(action: Action) -> Action:
    """
    Strip non-canonical fields so an action can be looked up in ACTIONS_ARRAY.

    Args:
        action: A raw ``catanatron`` ``Action``.

    Returns:
        Normalised ``Action`` ready for index lookup.
    """
    if action.action_type == ActionType.ROLL:
        return Action(action.color, action.action_type, None)
    if action.action_type == ActionType.MOVE_ROBBER:
        return Action(action.color, action.action_type, action.value[0])
    if action.action_type == ActionType.BUILD_ROAD:
        return Action(action.color, action.action_type, tuple(sorted(action.value)))
    if action.action_type in (ActionType.BUY_DEVELOPMENT_CARD, ActionType.DISCARD):
        return Action(action.color, action.action_type, None)
    return action


def _to_action_space(action: Action) -> int:
    """
    Convert a ``catanatron`` ``Action`` to its integer action-space index.

    Args:
        action: Raw ``catanatron`` action.

    Returns:
        Integer index into ``ACTIONS_ARRAY``.

    Raises:
        ValueError: If the normalised action is not found in ``ACTIONS_ARRAY``.
    """
    normalized = _normalize_action(action)
    return ACTIONS_ARRAY.index((normalized.action_type, normalized.value))


class CatanExpertComponent(PipelineComponent):
    """
    Pipeline component wrapping any catanatron ``Player`` as an expert.

    Instantiates one player internally and re-uses it for every step.
    The player's ``color`` is updated at decision time to match the game's
    current active color (so a single component instance is correct for
    both player seats in self-play).

    Args:
        player_class: Any ``catanatron.models.player.Player`` subclass
            that implements ``decide(game, playable_actions)``.
        init_color:   The ``catanatron.models.player.Color`` passed to the
            player constructor.  The color is overwritten before each
            decision so the exact initial value is arbitrary.
        **player_kwargs: Forwarded to ``player_class.__init__``.

    Example::

        from catanatron.players.minimax import AlphaBetaPlayer
        from catanatron.models.player import Color

        expert = CatanExpertComponent(AlphaBetaPlayer, Color.RED, depth=2)
    """

    def __init__(
        self,
        player_class: Type[Any],
        init_color: "Color",
        **player_kwargs: Any,
    ) -> None:
        self.player = player_class(init_color, **player_kwargs)

    @property
    def requires(self) -> Set[Key]:
        return {Key("data.info", SemanticType)}

    @property
    def provides(self) -> Set[Key]:
        return {Key("meta.action", Action)}

    def validate(self, blackboard: Blackboard) -> None:
        pass

    def execute(self, blackboard: Blackboard) -> Dict[str, Any]:
        """
        Query the catanatron player and publish the chosen action.

        Args:
            blackboard: The shared Blackboard for the current pipeline tick.

        Returns:
            Dictionary containing "meta.action".

        Raises:
            AssertionError: If ``blackboard.data["info"]["env"]`` is missing.
            ValueError: If the decided action is not in ``ACTIONS_ARRAY``.
        """
        info: dict = blackboard.data.get("info", {})
        assert info is not None and "env" in info, (
            "CatanExpertComponent: expected 'env' inside blackboard.data['info']. "
            "The Catan environment wrapper must store a reference to itself in "
            "info['env'] so the expert can access the live game state."
        )

        env = info["env"]
        # PettingZoo's OrderEnforcingWrapper blocks unknown attributes.
        game = env.unwrapped.game

        # Match the player color to whoever is currently active.
        self.player.color = game.state.current_color()

        action = self.player.decide(game, game.playable_actions)
        action_int: int = _to_action_space(action)

        return {"meta.action": action_int}
