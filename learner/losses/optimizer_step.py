import torch
from typing import Dict, Optional
from learner.pipeline.base import PipelineComponent
from learner.core import Blackboard

class OptimizerStepComponent(PipelineComponent):
    """
    Sits at the end of the pipeline. Calls backward(), applies clipping, and steps.
    """
    def __init__(
        self,
        agent_network: torch.nn.Module,
        optimizers: Dict[str, torch.optim.Optimizer],
        max_grad_norm: Optional[float] = None,
    ):
        self.agent_network = agent_network
        self.optimizers = optimizers
        self.max_grad_norm = max_grad_norm

    def execute(self, blackboard: Blackboard) -> None:
        total_losses = blackboard.losses.get("total_loss", {})
        
        for opt_key, loss_tensor in total_losses.items():
            opt = self.optimizers[opt_key]
            
            opt.zero_grad(set_to_none=True)
            loss_tensor.backward()

            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.agent_network.parameters(), self.max_grad_norm
                )

            opt.step()
