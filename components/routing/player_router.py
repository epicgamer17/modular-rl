"""
Player Router Component
=======================
Routes the current pipeline tick to a player-specific sub-pipeline based on
``blackboard.data["player_id"]``.

Why this exists
---------------
In multi-player games (e.g. TicTacToe, Catan) different agents may control
different player seats.  A common pattern is:

* Player 0: neural-network action selector (learned policy).
* Player 1: minimax expert or random baseline.

``PlayerRoutingComponent`` encapsulates this dispatch, keeping each player's
component list self-contained and preventing bleeding of player-specific state
across seats.

Design notes
------------
* Each sub-list is executed **in order**, exactly as a ``BlackboardEngine``
  would execute a flat list.
* The Blackboard is shared, so outputs written by a player-specific component
  (e.g. ``meta["action"]``) are visible to all downstream components.
* An arbitrary number of players is supported; the constructor accepts a
  mapping ``{player_id: [components...]}``.

Blackboard contract
-------------------
Reads:
    ``data["player_id"]`` – ``int`` index of the currently active player.

Side-effects:
    Delegates to every ``PipelineComponent.execute()`` in the matching list.
"""

from typing import Dict, List, Optional, Set

from core import PipelineComponent, Blackboard
from core.contracts import Key, ToPlay


class PlayerRoutingComponent(PipelineComponent):
    """
    Dispatch pipeline execution to a player-specific sub-list of components.

    Args:
        player_components: Mapping from player-id (``int``) to an ordered list
            of ``PipelineComponent`` instances that should run for that player.
        default_components: Optional fallback list used when the current
            player_id is not found in ``player_components``.  If ``None``
            (the default) and an unknown player_id is encountered, an
            ``AssertionError`` is raised.

    Example usage (two-player TicTacToe)::

        from components.selectors import ActionSelectorComponent
        from components.experts.tictactoe import TicTacToeExpertComponent
        from components.routing.player_router import PlayerRoutingComponent

        router = PlayerRoutingComponent(
            player_components={
                0: [NetworkInferenceComponent(agent), ActionSelectorComponent(input_key="logits")],
                1: [TicTacToeExpertComponent()],
            }
        )
    """

    def __init__(
        self,
        player_components: Dict[int, List[PipelineComponent]],
        default_components: Optional[List[PipelineComponent]] = None,
    ) -> None:
        assert len(player_components) > 0, (
            "PlayerRoutingComponent: player_components must contain at least one "
            "player entry."
        )
        for pid, components in player_components.items():
            assert isinstance(pid, int), (
                f"PlayerRoutingComponent: all player-id keys must be int, "
                f"got {type(pid)} for key {pid!r}."
            )
            assert isinstance(components, list) and len(components) > 0, (
                f"PlayerRoutingComponent: component list for player {pid} must be "
                f"a non-empty list."
            )

        self.player_components = player_components
        self.default_components = default_components

    @property
    def requires(self) -> Set[Key]:
        r = {Key("data.player_id", ToPlay)}
        for comps in self.player_components.values():
            for c in comps:
                r.update(c.requires)
        if self.default_components:
            for c in self.default_components:
                r.update(c.requires)
        return r

    @property
    def provides(self) -> Set[Key]:
        p = set()
        for comps in self.player_components.values():
            for c in comps:
                p.update(c.provides)
        if self.default_components:
            for c in self.default_components:
                p.update(c.provides)
        return p

    def validate(self, blackboard: Blackboard) -> None:
        pass

    def execute(self, blackboard: Blackboard) -> None:
        """
        Look up the active player's component list and execute each component.

        Args:
            blackboard: The shared Blackboard for the current pipeline tick.

        Raises:
            AssertionError: If ``blackboard.data["player_id"]`` is missing.
            AssertionError: If the player_id is not in ``player_components``
                and no ``default_components`` were provided.
        """
        assert "player_id" in blackboard.data, (
            "PlayerRoutingComponent: 'player_id' missing from blackboard.data. "
            "An observation component must write the active player's index to "
            "blackboard.data['player_id'] before routing."
        )

        player_id: int = int(blackboard.data["player_id"])

        if player_id in self.player_components:
            components = self.player_components[player_id]
        elif self.default_components is not None:
            components = self.default_components
        else:
            registered = sorted(self.player_components.keys())
            assert False, (
                f"PlayerRoutingComponent: received player_id={player_id} which is "
                f"not in the registered player map {registered} and no "
                f"default_components were provided."
            )

        for component in components:
            component.execute(blackboard)
