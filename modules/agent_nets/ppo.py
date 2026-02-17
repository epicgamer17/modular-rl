from typing import Callable, Tuple, Dict, Any
from configs.agents.ppo import PPOConfig
from torch import Tensor
import torch.nn as nn
from modules.heads.policy import PolicyHead
from modules.heads.value import ValueHead
from modules.heads.strategy_factory import OutputStrategyFactory


class PPONetwork(nn.Module):
    # This combines the two new base modules
    def __init__(self, config: PPOConfig, input_shape: Tuple[int]):
        super().__init__()
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

    def initialize(self, initializer: Callable[[Tensor], None]) -> None:
        self.policy.initialize(initializer)
        self.value.initialize(initializer)

    def forward(
        self, inputs: Tensor
    ) -> Tuple[Tuple[Tensor, Dict[str, Any]], Tuple[Tensor, Dict[str, Any]]]:
        """Returns standard head outputs: (policy_logits, policy_state), (value_logits, value_state)"""
        return self.policy(inputs), self.value(inputs)
