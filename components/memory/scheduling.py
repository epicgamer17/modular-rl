import torch
from typing import Any
from core import PipelineComponent, Blackboard


class PriorityUpdateComponent(PipelineComponent):
    """
    Updates priorities in the replay buffer based on priorities stored in the blackboard.
    This component is 'blind' to how priorities are computed; it simply reads
    blackboard.meta['priorities'] and sends them to the buffer.
    """
    def __init__(self, priority_update_fn: Any):
        self.priority_update_fn = priority_update_fn

    def execute(self, blackboard: Blackboard) -> None:
        priorities = blackboard.meta.get("priorities")
        indices = blackboard.data.get("indices")
        ids = blackboard.data.get("ids")

        if priorities is not None and indices is not None:
            # We must move to CPU before sending to the buffer
            self.priority_update_fn(indices, priorities.detach().cpu(), ids=ids)


class BetaScheduleComponent(PipelineComponent):
    """Steps the PER beta schedule."""
    def __init__(self, per_beta_schedule: Any, set_beta_fn: Any):
        self.per_beta_schedule = per_beta_schedule
        self.set_beta_fn = set_beta_fn

    def execute(self, blackboard: Blackboard) -> None:
        step = blackboard.meta.get("training_step")
        self.set_beta_fn(self.per_beta_schedule.get_value(step=step))
