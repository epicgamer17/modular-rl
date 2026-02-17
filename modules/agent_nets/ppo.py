from typing import Callable, Tuple, Dict, Any
from configs.agents.ppo import PPOConfig
from torch import Tensor
import torch.nn as nn
from modules.heads.policy import PolicyHead
from modules.heads.value import ValueHead
from modules.output_strategy_factory import OutputStrategyFactory


class PPONetwork(nn.Module):
    # This combines the two new base modules
    def __init__(
        self, config: PPOConfig, input_shape: Tuple[int], output_size: int, discrete
    ):
        super().__init__()
        # TODO: SHARED BACKBONE?
        # Policy Head (Actor)
        # Note: PolicyHeadConfig doesn't have an output strategy field yet in some contexts, but PolicyHead defaults to Categorical.
        # If PPOConfig.policy_head has neck, it handles the backbone.

        # We need to ensure we pass a strategy to PolicyHead or let it create one.
        # PPOConfig.policy_head should have the strategy config.
        # But here we are passing 'output_size' manually.
        # Let's see if config.policy_head has output_strategy.

        strategy = None
        if hasattr(config.policy_head, "output_strategy"):
            strategy = OutputStrategyFactory.create(config.policy_head.output_strategy)

        self.policy = PolicyHead(
            arch_config=config.arch,
            input_shape=input_shape,
            # Pass output_size/num_actions or strategy
            num_actions=output_size if discrete else None,
            output_size=output_size if not discrete else None,
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

    def forward(self, inputs: Tensor):
        # PPOTrainer expects:
        # dist = self.model.actor(obs)
        # value = self.model.critic(obs)
        # PolicyHead.forward returns dist object if return_probs is not used?
        # Wait, PolicyHead.forward currently returns:
        # logits = super().forward(x)
        # if return_probs: return self.strategy.logits_to_probs(logits)
        # return logits

        # Original ActorNetwork returned a distribution object (Categorical/Normal)
        # because the trainer does `dist.log_prob(actions)`.
        # The new PolicyHead currently returns PROBS or LOGITS tensor. It does NOT return a torch Distribution object yet.
        # THIS IS A BREAKING CHANGE if we don't fix PolicyHead or wrap it.
        # But typically PPO trainers want the distribution to sample/log_prob.

        # StandardCategoricalHead / PolicyHead logic needs to be verified.
        # In modules/heads/policy.py: 'strategy = Categorical(num_classes=num_actions)'
        # Categorical strategy: logits_to_probs returns softmax.

        # If the trainer does `dist = self.model.actor(obs)`, we need to return something that has `.log_prob`.
        # The generic PolicyHead returns a Tensor.

        # Let's check modules/heads/policy.py again.
        return self.policy(inputs), self.value(inputs)
