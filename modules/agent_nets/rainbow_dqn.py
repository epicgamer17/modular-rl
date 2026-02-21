from typing import Callable, Tuple, Any
import torch
from torch import nn, Tensor
from configs.agents.rainbow_dqn import RainbowConfig
from modules.backbones.factory import BackboneFactory
from modules.blocks.dense import DenseStack, build_dense
from modules.heads.q import QHead, DuelingQHead
from modules.heads.strategy_factory import OutputStrategyFactory
from utils.utils import to_lists  # Import the generalized block
from modules.agent_nets.base import BaseAgentNetwork
from modules.world_models.inference_output import InferenceOutput, LearningOutput


class RainbowNetwork(BaseAgentNetwork):
    """
    Rainbow DQN network combining Dueling architecture, Noisy nets, and
    distributional RL (C51).

    Implements the BaseAgentNetwork actor API via ``obs_inference`` and the
    learner API via ``learner_inference``. Planning methods
    (``hidden_state_inference``, ``afterstate_inference``) are not applicable
    and will raise ``NotImplementedError`` (handled by the base class).
    """

    def __init__(
        self,
        config: RainbowConfig,
        output_size: int,
        input_shape: Tuple[int, ...],
        *args,
        **kwargs,
    ):
        """
        Args:
            config: RainbowConfig containing architecture and head parameters.
            output_size: Number of discrete actions.
            input_shape: Shape of a single observation (without batch dimension).
        """
        super().__init__()
        self.config = config
        self.output_size = output_size
        self.atom_size = config.atom_size
        self.input_shape = input_shape

        # 1. Core Feature Extraction (Uses modular backbones)
        self.feature_block = BackboneFactory.create(config.backbone, input_shape)

        # Determine the final feature width/shape for heads
        current_shape = self.feature_block.output_shape

        # 2. Head (Dueling or Standard Q)
        strategy = OutputStrategyFactory.create(config.head.output_strategy)

        if self.config.dueling:
            self.head = DuelingQHead(
                arch_config=config.arch,
                input_shape=current_shape,
                strategy=strategy,
                value_hidden_widths=config.head.value_hidden_widths,
                advantage_hidden_widths=config.head.advantage_hidden_widths,
                num_actions=output_size,
                neck_config=config.head.neck,
            )
        else:
            self.head = QHead(
                arch_config=config.arch,
                input_shape=current_shape,
                strategy=strategy,
                hidden_widths=config.head.hidden_widths,
                num_actions=output_size,
                neck_config=config.head.neck,
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
            initializer: A callable that modifies a tensor in-place.
        """
        self.feature_block.initialize(initializer)
        self.head.initialize(initializer)

    def reset_noise(self) -> None:
        """Resamples NoisyNet parameters for exploration (no-op if noisy_sigma == 0)."""
        if self.config.noisy_sigma != 0:
            self.feature_block.reset_noise()
            self.head.reset_noise()

    @torch.inference_mode()
    def obs_inference(self, obs: Any) -> InferenceOutput:
        """
        Actor API: translates a raw observation into Q-values and a greedy
        distribution for use by ActionSelectors.

        Args:
            obs: Observation. May be a numpy array, list, or tensor. May be
                 unbatched (shape == input_shape) or batched.

        Returns:
            InferenceOutput with:
                - ``value``: Max Q-value (scalar per batch element).
                - ``q_values``: Expected Q-values ``(B, num_actions)`` for
                  action selection.
                - ``policy``: Greedy distribution derived from Q-values.
        """
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

        # Add batch dimension if the input is a single (unbatched) observation.
        if obs.dim() == len(self.input_shape):
            obs = obs.unsqueeze(0)

        # Forward pass: feature extraction → Q head
        x = self.feature_block(obs)
        Q = self.head(x)  # (B, num_actions, atom_size) or (B, num_actions)

        # Convert distributional output to expected Q-values: (B, num_actions)
        q_vals = self.head.strategy.to_expected_value(Q)

        # State value is max expected Q over all actions
        state_value = q_vals.max(dim=-1)[0]

        return InferenceOutput(
            value=state_value,
            q_values=q_vals,
            policy=self.head.strategy.get_distribution(Q),
        )

    def learner_inference(self, batch: Any) -> LearningOutput:
        """
        Learner API: computes raw Q-logits and expected Q-values for loss
        computation.

        For distributional RL (C51, atom_size > 1), ``q_logits`` contains the
        raw pre-softmax atom logits required by the numerically stable
        cross-entropy loss. For standard DQN (atom_size == 1), only ``q_values``
        is populated.

        Args:
            batch: Dict containing at minimum:
                - ``"observations"``: Float tensor of shape ``(B, *input_shape)``.

        Returns:
            LearningOutput with:
                - ``values``: Max expected Q-value per sample ``(B, 1)``.
                - ``q_values``: Expected Q-values ``(B, num_actions)``.
                - ``q_logits``: Raw atom logits ``(B, num_actions, atom_size)``
                  (only populated when using distributional RL).
        """
        obs = batch["observations"]
        # Assumes obs is already preprocessed (float32, normalised, on device)

        x = self.feature_block(obs)
        Q = self.head(x)  # (B, num_actions, atom_size) or (B, num_actions)

        q_vals = self.head.strategy.to_expected_value(Q)  # (B, num_actions)
        state_value = q_vals.max(dim=-1)[0]

        return LearningOutput(
            values=state_value.unsqueeze(-1),  # [B, 1] for unified pipeline
            q_values=q_vals,
            q_logits=Q,
        )
