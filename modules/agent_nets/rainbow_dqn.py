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
from modules.world_models.inference_output import LearningOutput


class RainbowNetwork(BaseAgentNetwork):
    def __init__(
        self,
        config: RainbowConfig,
        output_size: int,
        input_shape: Tuple[int],
        *args,
        **kwargs
    ):
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
                arch_config=config.arch,  # Assuming config has arch property or is compatible
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
        return (
            next(self.parameters()).device
            if list(self.parameters())
            else torch.device("cpu")
        )

    def initialize(self, initializer: Callable[[Tensor], None]) -> None:
        self.feature_block.initialize(initializer)
        self.head.initialize(initializer)

    def reset_noise(self):
        if self.config.noisy_sigma != 0:
            self.feature_block.reset_noise()
            self.head.reset_noise()

    @torch.inference_mode()
    def obs_inference(self, obs: Any) -> "InferenceOutput":
        from modules.world_models.inference_output import InferenceOutput

        # Ensure obs is a tensor
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

        # Ensure inputs have batch dimension
        if obs.dim() == len(self.input_shape):
            obs = obs.unsqueeze(0)

        # Forward pass
        x = self.feature_block(obs)
        Q = self.head(x)  # (B, actions, atoms) or (B, actions)

        # Q is (B, actions, atoms) or (B, actions)
        # Use strategy to convert to expected value (B, actions)
        q_vals = self.head.strategy.to_expected_value(Q)

        # For DQN, state value is max(Q)
        state_value = q_vals.max(dim=-1)[0]

        return InferenceOutput(
            value=state_value,
            q_values=q_vals,
            policy=self.head.strategy.get_distribution(Q),
        )

    def learner_inference(self, batch: Any) -> LearningOutput:
        """
        Calculates Q logits/atoms for the batch.
        """
        obs = batch["observations"]
        # Assumes obs is already preprocessed (float32, normalized)

        x = self.feature_block(obs)
        Q = self.head(x)  # (B, actions, atoms) or (B, actions)

        # Q is (B, actions, atoms) or (B, actions)
        # Use strategy to convert to expected value (B, actions)
        q_vals = self.head.strategy.to_expected_value(Q)

        # For DQN, state value is max(Q)
        state_value = q_vals.max(dim=-1)[0]

        return LearningOutput(
            values=state_value.unsqueeze(-1),  # [B, 1] for unified pipeline
            q_values=q_vals,
            q_logits=Q,
        )
