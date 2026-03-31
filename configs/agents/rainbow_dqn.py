from typing import Type, Union
from .base import AgentConfig
from old_muzero.configs.base import (
    DistributionalConfig,
    NoisyConfig,
    EpsilonGreedyConfig,
)
from old_muzero.configs.modules.backbones.base import BackboneConfig
from old_muzero.configs.modules.backbones.factory import BackboneConfigFactory
from old_muzero.configs.modules.heads.base import HeadConfig
from old_muzero.configs.modules.heads.q import QHeadConfig, DuelingQHeadConfig
from old_muzero.configs.modules.output_strategies import OutputStrategyConfigFactory
from old_muzero.configs.modules.backbones.factory import BackboneConfigFactory
from old_muzero.configs.modules.architecture_config import ArchitectureConfig
from old_muzero.utils.utils import tointlists


class RainbowConfig(
    AgentConfig, DistributionalConfig, NoisyConfig, EpsilonGreedyConfig
):
    def __init__(self, config_dict: dict, game_config):
        if "agent_type" not in config_dict:
            config_dict["agent_type"] = "rainbow"
        super(RainbowConfig, self).__init__(config_dict, game_config)

        # Parse shared architecture defaults
        arch_dict = self.parse_field("architecture", default={}, required=False)
        self.arch = ArchitectureConfig(arch_dict)
        print("RainbowConfig")

        # Mixin: Noisy
        self.parse_noisy_params()

        # Overwriting default parse behavior for Noisy to match Rainbow default 0.5 if not present
        if "noisy_sigma" not in self.config_dict:
            self.noisy_sigma = 0.5  # Restore Rainbow Default

        self.backbone: BackboneConfig = self.parse_backbone_config("backbone")

        # Mixin: Epsilon Greedy
        self.parse_epsilon_greedy_params()

        self.dueling: bool = self.parse_field("dueling", True)

        # self.discount_factor parsed in Config
        self.soft_update: bool = self.parse_field("soft_update", False)
        self.transfer_interval: int = self.parse_field(
            "transfer_interval", 512, wrapper=int
        )
        self.ema_beta: float = self.parse_field("ema_beta", 0.99)
        self.replay_interval: int = self.parse_field("replay_interval", 1, wrapper=int)

        # Mixin: Distributional (Atom Size)
        self.parse_distributional_params()

        # Logic moved to DistributionalConfig, but verifying assignment
        self.v_min = game_config.min_score
        self.v_max = game_config.max_score

        if self.atom_size != 1:
            assert self.v_min != None and self.v_max != None

        # --- Head Configuration ---
        head_dict = self.parse_field("head", {}, required=False)

        # 1. Inject Global Noisy Sigma
        if "noisy_sigma" not in head_dict:
            head_dict["noisy_sigma"] = self.noisy_sigma

        # 2. Handle Output Strategy & Sync Atom Size
        # If user defines strategy in head, we prioritize it and sync back to self.atom_size
        if "output_strategy" in head_dict:
            strat = head_dict["output_strategy"]
            if strat.get("type") == "c51":
                # Check for atom_size update
                if "num_atoms" in strat:
                    self.atom_size = strat["num_atoms"]

                # Check for bounds update (Sync Game Defaults -> Strategy)
                if "v_min" not in strat and self.v_min is not None:
                    strat["v_min"] = self.v_min
                if "v_max" not in strat and self.v_max is not None:
                    strat["v_max"] = self.v_max

                # Update self bounds from strategy if strategy provided them
                if "v_min" in strat:
                    self.v_min = strat["v_min"]
                if "v_max" in strat:
                    self.v_max = strat["v_max"]

        # If user DID NOT define strategy, infer from self.atom_size (Legacy/Fallback)
        else:
            if self.atom_size > 1:
                head_dict["output_strategy"] = {
                    "type": "c51",
                    "num_atoms": self.atom_size,
                    "v_min": self.v_min,
                    "v_max": self.v_max,
                }
            else:
                head_dict["output_strategy"] = {"type": "scalar"}

        # 3. Construct Head Config
        if self.dueling:
            self.head = DuelingQHeadConfig(head_dict)
        else:
            self.head = QHeadConfig(head_dict)

    def _verify_game(self):
        assert self.game.is_discrete, "Rainbow only supports discrete action spaces"

    def parse_backbone_config(self, field_name: str) -> BackboneConfig:
        bb_dict = self.parse_field(field_name, default=None, required=False)
        if bb_dict is None:
            return None

        # Propagate noisy_sigma if not explicitly set in backbone config
        if "noisy_sigma" not in bb_dict:
            bb_dict["noisy_sigma"] = self.noisy_sigma

        return BackboneConfigFactory.create(bb_dict)
