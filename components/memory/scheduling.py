from typing import Any, Set, Dict
import torch
from core import PipelineComponent, Blackboard
from core.contracts import Key, SemanticType, Priority, Metric


class PriorityUpdateComponent(PipelineComponent):
    """
    Updates priorities in the replay buffer based on priorities stored in the blackboard.
    This component is 'blind' to how priorities are computed; it simply reads
    blackboard.meta['priorities'] and sends them to the buffer.
    """
    def __init__(self, priority_update_fn: Any):
        self.priority_update_fn = priority_update_fn
        self._requires = {
            Key("meta.priorities", Priority),
            Key("data.indices", SemanticType),
        }
        self._provides = {}

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Dict[Key, str]:
        return self._provides

    def validate(self, blackboard: Blackboard) -> None:
        """Ensures priorities and indices are available."""
        assert blackboard.meta.get("priorities") is not None, (
            "PriorityUpdateComponent: 'priorities' missing from blackboard.meta"
        )
        assert blackboard.data.get("indices") is not None, (
            "PriorityUpdateComponent: 'indices' missing from blackboard.data"
        )

    def execute(self, blackboard: Blackboard) -> Dict[str, Any]:
        priorities = blackboard.meta.get("priorities")
        indices = blackboard.data.get("indices")
        ids = blackboard.data.get("ids")

        if priorities is not None and indices is not None:
            # We must move to CPU before sending to the buffer
            self.priority_update_fn(indices, priorities.detach().cpu(), ids=ids)
        
        return {}


class BetaScheduleComponent(PipelineComponent):
    """Steps the PER beta schedule."""
    def __init__(self, per_beta_schedule: Any, set_beta_fn: Any):
        self.per_beta_schedule = per_beta_schedule
        self.set_beta_fn = set_beta_fn
        self._requires = {Key("meta.training_step", Metric)}
        self._provides = {}

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Dict[Key, str]:
        return self._provides

    def validate(self, blackboard: Blackboard) -> None:
        """Ensures training step is available."""
        assert blackboard.meta.get("training_step") is not None, (
            "BetaScheduleComponent: 'training_step' missing from blackboard.meta"
        )

    def execute(self, blackboard: Blackboard) -> Dict[str, Any]:
        step = blackboard.meta.get("training_step")
        if step is not None:
            self.set_beta_fn(self.per_beta_schedule.get_value(step=step))
        return {}
