import torch
from typing import List
from core import PipelineComponent
from core import Blackboard


class PassThroughTargetComponent(PipelineComponent):
    """
    Generic whitelist-based component that passes specific keys
    from the batch through to the targets.
    """

    def __init__(self, keys_to_keep: List[str]):
        self.keys_to_keep = keys_to_keep

    def execute(self, blackboard: Blackboard) -> None:
        data = blackboard.data
        for key in self.keys_to_keep:
            if key in data:
                val = data[key]
                # Ensure T dimension [B, 1, ...] if not already present
                if torch.is_tensor(val) and (
                    val.ndim == 1 or (val.ndim >= 2 and val.shape[1] != 1)
                ):
                    blackboard.targets[key] = val.unsqueeze(1)
                else:
                    blackboard.targets[key] = val
