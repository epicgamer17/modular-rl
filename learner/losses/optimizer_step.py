import torch
from typing import Dict, Any, Optional, Union
from learner.pipeline.base import PipelineComponent
from learner.core import Blackboard

class OptimizerStepComponent(PipelineComponent):
    """
    Sits at the end of the pipeline. Pulls total_loss from blackboard, 
    calls backward(), applies gradient clipping, and steps optimizers.
    Also manages learning rate schedulers and gradient accumulation.
    """
    def __init__(
        self,
        agent_network: torch.nn.Module,
        optimizer: Optional[Union[torch.optim.Optimizer, Dict[str, torch.optim.Optimizer]]] = None,
        lr_scheduler: Optional[Union[Any, Dict[str, Any]]] = None,
        clipnorm: Optional[float] = None,
        max_grad_norm: Optional[float] = None,
        gradient_accumulation_steps: int = 1,
    ):
        self.agent_network = agent_network
        self.clipnorm = clipnorm
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self.step_counter = 0

        # Normalize optimizers into dictionaries
        if isinstance(optimizer, dict):
            self.optimizers = optimizer
        elif optimizer is not None:
            self.optimizers = {"default": optimizer}
        else:
            self.optimizers = {}

        if isinstance(lr_scheduler, dict):
            self.lr_schedulers = lr_scheduler
        elif lr_scheduler is not None:
            self.lr_schedulers = {"default": lr_scheduler}
        else:
            self.lr_schedulers = {}

        # Ensure we start with clean decoupled memory
        for opt in self.optimizers.values():
            opt.zero_grad(set_to_none=True)

    def execute(self, blackboard: Blackboard) -> None:
        """
        Pulls total_loss dict from blackboard, calls backward, applies clipping and steps
        if at an accumulation boundary.
        """
        if not blackboard.losses:
            return  # No loss computed.

        self.step_counter += 1

        # 1. Backward Pass
        # Iterate over unique optimizers
        num_losses = len(blackboard.losses)
        for opt_key, total_loss in blackboard.losses.items():
            # Scale loss for gradient accumulation so accumulated gradients represent the mean over accumulation steps
            loss_tensor = total_loss / self.gradient_accumulation_steps
            # Keep graph attached if multiple optimizers share backward computation graph
            loss_tensor.backward(retain_graph=(num_losses > 1))

        # 2. Accumulation boundary
        if self.step_counter % self.gradient_accumulation_steps == 0:
            
            # Application of gradient clipping
            if self.clipnorm is not None and self.clipnorm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.agent_network.parameters(), self.clipnorm
                )
            elif self.max_grad_norm is not None and self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.agent_network.parameters(), self.max_grad_norm
                )

            # 3. Optimizer Step and memory clearing using standard PyTorch best practice
            for opt in self.optimizers.values():
                opt.step()
                opt.zero_grad(set_to_none=True)

            # 4. Step LR Scheduler
            for sched in self.lr_schedulers.values():
                sched.step()
