from typing import Callable, Tuple
from torch import nn, Tensor
from configs.agents.rainbow_dqn import RainbowConfig
from modules.backbones.factory import BackboneFactory
from modules.blocks.dense import DenseStack, build_dense
from modules.heads.q import QHead, DuelingQHead
from modules.heads.strategy_factory import OutputStrategyFactory
from utils.utils import to_lists  # Import the generalized block


class RainbowNetwork(nn.Module):
    def __init__(
        self,
        config: RainbowConfig,
        output_size: int,
        input_shape: Tuple[int],
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.config = config
        self.output_size = output_size
        self.atom_size = config.atom_size

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

    def initialize(self, initializer: Callable[[Tensor], None]) -> None:
        self.feature_block.initialize(initializer)
        self.head.initialize(initializer)

    def forward(self, inputs: Tensor) -> Tensor:
        # Pass through core layers
        x = self.feature_block(inputs)

        # Head outputs (B, actions, atoms)
        Q = self.head(x)

        if self.atom_size == 1:
            return Q.squeeze(-1)
        else:
            return Q.softmax(dim=-1)

    def reset_noise(self):
        if self.config.noisy_sigma != 0:
            self.feature_block.reset_noise()
            self.head.reset_noise()
