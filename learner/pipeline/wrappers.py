from typing import Optional, List
from learner.pipeline.base import PipelineComponent
from learner.core import Blackboard
from modules.agent_nets.base import BaseAgentNetwork
from learner.pipeline.target_builders import BaseTargetBuilder
from learner.pipeline.callbacks import CallbackList, Callback

class TargetBuilderComponent(PipelineComponent):
    """
    Wraps existing Target Builders into the new Pipeline format.
    Computes targets that require current network weights and writes to Blackboard.
    """
    def __init__(self, target_builder: BaseTargetBuilder, agent_network: BaseAgentNetwork):
        self.target_builder = target_builder
        self.agent_network = agent_network

    def execute(self, blackboard: Blackboard) -> None:
        if self.target_builder is not None:
            self.target_builder.build_targets(
                batch=blackboard.batch,
                predictions=blackboard.predictions,
                network=self.agent_network,
                current_targets=blackboard.targets,
            )

class ComponentCallbacks(PipelineComponent):
    """
    Temporary bridge that fires callback events before and after execution flow.
    Can be broken down into individual Pipeline Components later.
    """
    def __init__(self, callbacks: List[Callback], hook: str = "on_step_end"):
        self.callbacks = CallbackList(callbacks)
        self.hook = hook
        
    def execute(self, blackboard: Blackboard) -> None:
        if self.hook == "on_step_end":
            self.callbacks.on_step_end(
                learner=None, # Deprecated reference
                predictions=blackboard.predictions,
                targets=blackboard.targets,
                loss_dict=blackboard.loss_dict,
                priorities=blackboard.priorities,
                batch=blackboard.batch,
                meta=blackboard.meta,
            )
