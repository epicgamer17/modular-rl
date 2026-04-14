from typing import Any, Set, Dict
import torch
from core import PipelineComponent, Blackboard
from core.contracts import Key, SemanticType


class PriorityUpdateComponent(PipelineComponent):
    """
    Updates priorities in the replay buffer based on priorities stored in the blackboard.
    This component is 'blind' to how priorities are computed; it simply reads
    blackboard.meta['priorities'] and sends them to the buffer.
    """
    def __init__(self, priority_update_fn: Any):
        self.priority_update_fn = priority_update_fn

    @property
    def requires(self) -> Set[Key]:
        return {
            Key("meta.priorities", SemanticType),
            Key("data.indices", SemanticType),
        }

    @property
    def provides(self) -> Set[Key]:
        return set()

    def validate(self, blackboard: Blackboard) -> None:
        pass

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

    @property
    def requires(self) -> Set[Key]:
        return {Key("meta.training_step", SemanticType)}

    @property
    def provides(self) -> Set[Key]:
        return set()

    def validate(self, blackboard: Blackboard) -> None:
        pass

    def execute(self, blackboard: Blackboard) -> Dict[str, Any]:
        step = blackboard.meta.get("training_step")
        if step is not None:
            self.set_beta_fn(self.per_beta_schedule.get_value(step=step))
        return {}
