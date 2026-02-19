from typing import Callable, Tuple, Dict, Any
from configs.agents.ppo import PPOConfig
from torch import Tensor
import torch
import torch.nn as nn
from modules.heads.policy import PolicyHead
from modules.heads.value import ValueHead
from modules.heads.strategy_factory import OutputStrategyFactory


class PPONetwork(nn.Module):
    # This combines the two new base modules
    def __init__(self, config: PPOConfig, input_shape: Tuple[int]):
        super().__init__()
        self.input_shape = input_shape
        # TODO: SHARED BACKBONE?
        # Policy Head (Actor)
        strategy = None
        if hasattr(config.policy_head, "output_strategy"):
            strategy = OutputStrategyFactory.create(config.policy_head.output_strategy)

        self.policy = PolicyHead(
            arch_config=config.arch,
            input_shape=input_shape,
            neck_config=config.policy_head.neck,
            strategy=strategy,
        )

        # Value Head (Critic)
        val_strat = OutputStrategyFactory.create(config.value_head.output_strategy)
        self.value = ValueHead(
            arch_config=config.arch,
            input_shape=input_shape,
            strategy=val_strat,
            neck_config=config.value_head.neck,
        )

    @property
    def device(self) -> torch.device:
        return (
            next(self.parameters()).device
            if list(self.parameters())
            else torch.device("cpu")
        )

    def initialize(self, initializer: Callable[[Tensor], None]) -> None:
        self.policy.initialize(initializer)
        self.value.initialize(initializer)

    def initial_inference(self, inputs: Tensor) -> "InferenceOutput":
        from modules.world_models.inference_output import InferenceOutput

        # Ensure inputs is a tensor
        if not torch.is_tensor(inputs):
            inputs = torch.as_tensor(inputs, dtype=torch.float32, device=self.device)

        # Ensure input has batch dim
        # Assuming input_shape from config does NOT include batch
        if inputs.dim() == len(self.input_shape):
            inputs = inputs.unsqueeze(0)

        # Standard head outputs: (policy_logits, policy_state), (value_logits, value_state)
        # Assuming self.policy() returns distribution or we need to construct it?
        # self.policy is usually PolicyHead.
        # PolicyHead logic might need checking. It usually returns (dist, state).
        (policy_logits, _, policy_dist), (value, _) = self.policy(inputs), self.value(
            inputs
        )

        return InferenceOutput(
            policy=policy_dist, policy_logits=policy_logits, value=value
        )
