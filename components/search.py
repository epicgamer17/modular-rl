import torch
from core import PipelineComponent
from core import Blackboard


# TODO: clean up some of the target builder stuff, it still feels too closely tied to this idea of "targets", which i feel bay not exist? idk.
class MCTSExtractorComponent(PipelineComponent):
    """Extracts MCTS search statistics from the batch into targets."""

    def execute(self, blackboard: Blackboard) -> None:
        data = blackboard.data
        target_keys = ["values", "rewards", "policies", "actions", "to_plays"]
        for key in target_keys:
            if key in data:
                blackboard.targets[key] = data[key]
