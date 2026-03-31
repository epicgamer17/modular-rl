from typing import Optional, Dict, Any, TypeVar, Type
import torch
from configs.modules.architecture_config import ArchitectureConfig
from configs.base import (
    ConfigBase,
    OptimizationConfig,
    ReplayConfig,
    RecordConfig,
    DistributionalConfig,
    NoisyConfig,
    ExecutionConfig,
    CompilationConfig,
)
import torch.nn.functional as F
from modules.utils import (
    prepare_activations,
    prepare_kernel_initializers,
    kernel_initializer_wrapper,
)
from configs.games.game import GameConfig
from configs.selectors import SelectorConfig


class AgentConfig(
    ConfigBase, OptimizationConfig, ReplayConfig, RecordConfig, ExecutionConfig
):
    """
    Base configuration for all agents.
    Inherits from various mixins to provide standard capabilities.
    Also manages the 'arch' attribute for modular network components.
    """

    def __init__(self, config_dict: dict, game_config: GameConfig) -> None:
        super().__init__(config_dict)
        self.game = game_config

        self.agent_type: str = self.parse_field("agent_type", required=True)
        self.results_path: str = self.parse_field("results_path", "results")
        self._verify_game()

        # Initialize Mixins
        self.parse_optimization_params()
        self.parse_replay_params()
        self.parse_record_params()

        # Execution Config
        self.parse_execution_params()

        # Core/Common Params
        self.save_intermediate_weights: bool = self.parse_field(
            "save_intermediate_weights", False
        )
        self.test_trials: int = self.parse_field("test_trials", 5)

        # Compilation Config
        compilation_dict = self.parse_field("compilation", default={}, required=False)
        self.compilation = CompilationConfig(compilation_dict)

        # Loss & Activation
        self.loss_function = self.parse_field("loss_function", F.mse_loss)
        self.activation = self.parse_field(
            "activation", "relu", wrapper=prepare_activations
        )

        # Initializers
        self.kernel_initializer = self.parse_field(
            "kernel_initializer",
            None,
            required=False,
            wrapper=kernel_initializer_wrapper,
        )
        self.prob_layer_initializer = self.parse_field(
            "prob_layer_initializer",
            None,
            required=False,
            wrapper=kernel_initializer_wrapper,
        )

        # Architecture Config (The 'arch' attribute)
        # This provides a unified view of architecture params for heads/backbones
        self.arch = ArchitectureConfig(config_dict)

        # Legacy/Compatibility params (handled by mixins usually, but some were in Config)
        self.norm_type: str = self.parse_field("norm_type", "none")
        # Action Selector
        # NO DEFAULT SELECTORS ALL AGENTS SHOULD DEFINE THEIR OWN IN THEIR CONFIGS
        selector_dict = self.parse_field("action_selector", required=True)
        self.action_selector = SelectorConfig(selector_dict)

    def _verify_game(self):
        assert (
            self.game is not None
        ), "Config requires a game config to be provided in 'game' field"
        assert (
            self.game.env_factory is not None
        ), "Game config must provide a valid environment factory (env_factory)"
