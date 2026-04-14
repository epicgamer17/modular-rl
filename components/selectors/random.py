"""
Random Selector Component
=========================
A minimal baseline selector that picks a uniformly-random legal action.

Blackboard contract
-------------------
Reads:
    ``data["info"]`` – Dict containing an optional ``"legal_moves"`` key
        (``List[int]``).  If the key is absent or the list is empty every
        action index ``[0, num_actions)`` is treated as legal and one is
        chosen at random.

Writes:
    ``meta["action"]`` – ``int`` selected action index.
"""

import random
from typing import Optional, Set, Any, Dict

from core import PipelineComponent, Blackboard
from core.contracts import Key, Action, SemanticType

# Sentinel value used when no action could be determined (unreachable in
# normal operation; kept to make the assertion message informative).
_NO_ACTION = -1

# Fallback action when the info dict is entirely missing and ``num_actions``
# was not provided.  This mirrors the historical ``RandomAgent`` behaviour.
_FALLBACK_ACTION = 0


class RandomSelectorComponent(PipelineComponent):
    """
    Selects a uniformly-random legal action and writes it to the Blackboard.

    This component is a direct replacement for the legacy ``RandomAgent``
    class.  It reads ``blackboard.data["info"]["legal_moves"]`` and samples
    one index uniformly at random.

    Args:
        num_actions: Total size of the action space.  Used as a fallback when
            no legal-moves list is available in the info dict.  If ``None``
            and no legal-moves list is present, action ``0`` is returned.

    Example pipeline usage::

        pipeline = BlackboardEngine(components=[
            GymObservationComponent(env),
            RandomSelectorComponent(num_actions=env.action_space.n),
            GymStepComponent(env, obs_component),
        ])
    """

    def __init__(self, num_actions: Optional[int] = None) -> None:
        self.num_actions = num_actions
        self._requires = {Key("data.info", SemanticType)}
        self._provides = {Key("meta.action", Action): "new"}

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Dict[Key, str]:
        return self._provides

    def validate(self, blackboard: Blackboard) -> None:
        """Ensures info dict is accessible and legal moves or num_actions is available."""
        info = blackboard.data.get("info", {})
        if info is None:
            info = {}
        legal_moves = info.get("legal_moves", [])
        if not legal_moves:
            assert self.num_actions is not None, (
                "RandomSelectorComponent: no legal moves in info and num_actions not provided"
            )

    def execute(self, blackboard: Blackboard) -> Dict[str, Any]:
        """
        Sample a legal action uniformly at random and publish it.

        Args:
            blackboard: The shared Blackboard for the current pipeline tick.

        Raises:
            AssertionError: If ``num_actions`` is ``None`` and the info dict
                contains no ``"legal_moves"`` list (no action can be chosen).
        """
        info: dict = blackboard.data.get("info", {})
        if info is None:
            info = {}

        legal_moves = info.get("legal_moves", [])

        if legal_moves is not None and len(legal_moves) > 0:
            # Sample from provided legal-move indices.
            action: int = int(random.choice(legal_moves))
        else:
            # Fall back to the full action space.
            assert self.num_actions is not None, (
                "RandomSelectorComponent: no legal moves found in "
                "blackboard.data['info']['legal_moves'] and num_actions was "
                "not provided.  Pass num_actions to the constructor so a "
                "fallback action can be sampled."
            )
            action = random.randrange(self.num_actions)

        return {"meta.action": action}
