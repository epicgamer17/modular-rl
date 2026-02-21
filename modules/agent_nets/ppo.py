from typing import Callable, Tuple, Dict, Any
from configs.agents.ppo import PPOConfig
from torch import Tensor
import torch
import torch.nn as nn
from modules.heads.policy import PolicyHead
from modules.heads.value import ValueHead
from modules.heads.strategy_factory import OutputStrategyFactory
from modules.agent_nets.base import BaseAgentNetwork
from modules.world_models.inference_output import InferenceOutput, LearningOutput


class PPONetwork(BaseAgentNetwork):
    """
    Actor-Critic network for Proximal Policy Optimization (PPO).

    Implements separate Policy (Actor) and Value (Critic) heads, both sharing
    the same architecture config but maintaining independent parameters.
    The BaseAgentNetwork API is satisfied via obs_inference (actor/selector loop)
    and learner_inference (PPO gradient updates).
    """

    def __init__(self, config: PPOConfig, input_shape: Tuple[int, ...]):
        """
        Args:
            config: PPOConfig containing architecture and head configurations.
            input_shape: Shape of a single observation (without batch dimension).
        """
        super().__init__()
        self.config = config
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
        """Returns the device on which the network parameters reside."""
        return (
            next(self.parameters()).device
            if list(self.parameters())
            else torch.device("cpu")
        )

    def initialize(self, initializer: Callable[[Tensor], None]) -> None:
        """
        Applies a custom weight initializer to all sub-modules.

        Args:
            initializer: A callable that modifies a tensor in-place
                         (e.g., orthogonal init).
        """
        self.policy.initialize(initializer)
        self.value.initialize(initializer)

    @torch.inference_mode()
    def obs_inference(self, obs: Tensor) -> InferenceOutput:
        """
        Actor API: translates a raw observation into a value estimate and
        an action distribution for use by ActionSelectors.

        Args:
            obs: Observation tensor. May be unbatched (shape == input_shape)
                 or batched (shape == (B, *input_shape)).

        Returns:
            InferenceOutput with ``policy`` (Distribution) and ``value``
            (expected scalar, computed via the value head's output strategy).
        """
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

        # Add batch dimension if the input is a single (unbatched) observation.
        if obs.dim() == len(self.input_shape):
            obs = obs.unsqueeze(0)

        # Independent forward passes through each head.
        policy_logits, _, policy_dist = self.policy(obs)
        value_logits, _ = self.value(obs)

        expected_value = self.value.strategy.to_expected_value(value_logits)

        return InferenceOutput(policy=policy_dist, value=expected_value)

    def learner_inference(self, batch: Dict[str, Any]) -> LearningOutput:
        """
        Learner API: computes raw policy and value logits for loss computation.

        PPO computes policy and value losses independently, so both heads are
        run here. Raw logits (not distributions) are returned so that the
        PPOPolicyLoss and PPOValueLoss modules can apply numerically stable
        log-probability computations themselves.

        Args:
            batch: Dict containing at minimum:
                - ``"observations"``: Float tensor of shape ``(B, *input_shape)``.

        Returns:
            LearningOutput with:
                - ``policies``: Raw policy logits ``(B, num_actions)``.
                - ``values``: Raw value logits ``(B, ...)`` (strategy-dependent).
        """
        obs = batch["observations"].to(self.device)

        policy_logits, _, _ = self.policy(obs)
        value_logits, _ = self.value(obs)

        return LearningOutput(
            values=value_logits,
            policies=policy_logits,
        )
