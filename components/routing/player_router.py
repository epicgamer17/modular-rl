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

from typing import Dict, List, Optional, Set, Any

from core import PipelineComponent, Blackboard
from core.contracts import Key, ToPlay
from core.blackboard_engine import apply_updates


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
    def provides(self) -> Dict[Key, str]:
        p = {}
        for comps in self.player_components.values():
            for c in comps:
                child_provides = c.provides
                if isinstance(child_provides, dict):
                    p.update(child_provides)
                else:
                    for k in child_provides:
                        p[k] = "new"
        if self.default_components:
            for c in self.default_components:
                child_provides = c.provides
                if isinstance(child_provides, dict):
                    p.update(child_provides)
                else:
                    for k in child_provides:
                        p[k] = "new"
        return p

    def validate(self, blackboard: Blackboard) -> None:
        """Ensures 'player_id' exists and a valid component list is available."""
        from core.validation import assert_in_blackboard
        assert_in_blackboard(blackboard, "data.player_id")
        
        player_id = int(blackboard.data["player_id"])
        if player_id not in self.player_components and self.default_components is None:
            registered = sorted(self.player_components.keys())
            assert False, (
                f"PlayerRoutingComponent: received player_id={player_id} which is "
                f"not in the registered player map {registered} and no "
                f"default_components were provided."
            )

    def execute(self, blackboard: Blackboard) -> Dict[str, Any]:
        """
        Look up the active player's component list and execute each component.
        """
        player_id: int = int(blackboard.data["player_id"])

        if player_id in self.player_components:
            components = self.player_components[player_id]
        else:
            # If we reached here, validate() must have passed, 
            # meaning default_components is not None if player_id wasn't in player_components.
            components = self.default_components

        combined_updates = {}
        for component in components:
            # Execute sub-component and capture its returned mutations
            result = component.execute(blackboard)
            if result:
                # Meta-components must manually apply or return updates if they impact downstream routed nodes
                # However, for now, we assume routed nodes need results from previous routed nodes in-place to function.
                # To maintain side-effect safety, we'll manually apply them here for the NEXT component in the router's loop.
                # This mirrors BlackboardEngine logic.
                apply_updates(blackboard, result)
                combined_updates.update(result)
        
        return combined_updates
