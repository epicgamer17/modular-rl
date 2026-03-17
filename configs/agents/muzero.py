from typing import Callable, Type, Optional, List, Dict
from .base import AgentConfig
from configs.base import (
    SearchConfig,
    ValuePrefixConfig,
    ConsistencyConfig,
    DistributionalConfig,
    NoisyConfig,
)
from configs.modules.backbones.base import BackboneConfig
from configs.modules.backbones.factory import BackboneConfigFactory
from configs.modules.heads.to_play import ToPlayHeadConfig
from configs.modules.heads.policy import PolicyHeadConfig
from configs.modules.heads.chance_probability import ChanceProbabilityHeadConfig
from configs.modules.heads.value import ValueHeadConfig
from configs.modules.heads.reward import RewardHeadConfig, ValuePrefixRewardHeadConfig
from configs.modules.heads.base import HeadConfig
from configs.modules.architecture_config import ArchitectureConfig
import torch.nn.functional as F
from utils.utils import tointlists
from utils.schedule import ScheduleConfig
import copy


class MuZeroConfig(
    AgentConfig,
    SearchConfig,
    ValuePrefixConfig,
    ConsistencyConfig,
    DistributionalConfig,
    NoisyConfig,
):
    def __init__(self, config_dict, game_config):
        if "agent_type" not in config_dict:
            config_dict["agent_type"] = "muzero"
        super(MuZeroConfig, self).__init__(config_dict, game_config)

        # Initialize Architecture Config handled by AgentConfig

        # Mixin: Distributional (Support Range) - Parse early for head validation
        self.parse_distributional_params()

        self.stochastic: bool = self.parse_field("stochastic", False)
        self.use_true_chance_codes: bool = self.parse_field(
            "use_true_chance_codes", False
        )
        self.num_chance: int = self.parse_field("num_chance", 32)
        self.sigma_loss = self.parse_field("sigma_loss", F.cross_entropy)
        self.vqvae_commitment_cost_factor: float = self.parse_field(
            "vqvae_commitment_cost_factor", 1.0
        )

        # Mixin: Value Prefix
        self.parse_value_prefix_params()

        # Backbone Configurations
        self.representation_backbone: BackboneConfig = self.parse_backbone_config(
            "representation_backbone"
        )
        self.dynamics_backbone: BackboneConfig = self.parse_backbone_config(
            "dynamics_backbone"
        )
        self.prediction_backbone: BackboneConfig = self.parse_backbone_config(
            "prediction_backbone"
        )
        self.afterstate_dynamics_backbone: BackboneConfig = self.parse_backbone_config(
            "afterstate_dynamics_backbone"
        )
        self.chance_encoder_backbone: BackboneConfig = self.parse_backbone_config(
            "chance_encoder_backbone"
        )

        reward_head_cls = (
            ValuePrefixRewardHeadConfig if self.use_value_prefix else RewardHeadConfig
        )
        # Inject LSTM params into reward head dict if using Value Prefix
        # Helper to process reward dict
        rh_dict = self.parse_field("reward_head", default={}, required=False) or {}

        if self.use_value_prefix:
            if "lstm_hidden_size" not in rh_dict:
                rh_dict["lstm_hidden_size"] = self.lstm_hidden_size
            if "lstm_horizon_len" not in rh_dict:
                rh_dict["lstm_horizon_len"] = self.lstm_horizon_len

        # Enforce strategy if distributional
        if self.atom_size > 1:
            rew_strat = rh_dict.get("output_strategy", None)
            if rew_strat is None:
                rew_strat = {"type": "muzero", "support_range": self.support_range}
            # Force num_classes and support_range to match atom_size
            rew_strat["num_classes"] = self.atom_size
            rew_strat["support_range"] = self.support_range
            rh_dict["output_strategy"] = rew_strat

        self.reward_head: RewardHeadConfig = reward_head_cls(rh_dict)

        # Value Head - Enforce strategy if distributional
        value_dict = self.parse_field("value_head", default={}, required=False) or {}
        if self.atom_size > 1:
            val_strat = value_dict.get("output_strategy", None)
            if val_strat is None:
                val_strat = {"type": "muzero", "support_range": self.support_range}
            # Force num_classes and support_range to match atom_size
            val_strat["num_classes"] = self.atom_size
            val_strat["support_range"] = self.support_range
            value_dict["output_strategy"] = val_strat

        self.value_head: ValueHeadConfig = ValueHeadConfig(value_dict)
        # To Play Head - Custom parsing to inject num_players from game config
        tp_dict = self.parse_field("to_play_head", default={}, required=False) or {}
        if "num_players" not in tp_dict:
            tp_dict["num_players"] = self.game.num_players

        # Ensure output_strategy has num_classes (needed for CategoricalConfig)
        tp_strat = tp_dict.get("output_strategy", {"type": "categorical"})
        if "num_classes" not in tp_strat:
            tp_strat["num_classes"] = self.game.num_players
        tp_dict["output_strategy"] = tp_strat

        self.to_play_head: ToPlayHeadConfig = ToPlayHeadConfig(tp_dict)
        # Policy Head - Inject num_classes from game
        poly_dict = self.parse_field("policy_head", default={}, required=False) or {}
        # Determine num_actions dynamically
        num_actions = self.game.num_actions

        if num_actions is not None:
            pol_strat = poly_dict.get("output_strategy", {"type": "categorical"})
            if "num_classes" not in pol_strat:
                pol_strat["num_classes"] = num_actions
            poly_dict["output_strategy"] = pol_strat

        self.policy_head: PolicyHeadConfig = PolicyHeadConfig(poly_dict)

        # Chance Probability Head - Inject num_chance
        chance_dict = (
            self.parse_field("chance_probability_head", default={}, required=False)
            or {}
        )
        chance_strat = chance_dict.get("output_strategy", {"type": "categorical"})
        if "num_classes" not in chance_strat:
            chance_strat["num_classes"] = self.num_chance
        chance_dict["output_strategy"] = chance_strat

        self.chance_probability_head: ChanceProbabilityHeadConfig = (
            ChanceProbabilityHeadConfig(chance_dict)
        )

        # Support a shared 'backbone' field as a fallback
        shared_bb = self.parse_field("backbone", default=None, required=False)
        if shared_bb:
            default_bb_cfg = BackboneConfigFactory.create(shared_bb)
            if self.representation_backbone is None:
                self.representation_backbone = default_bb_cfg
            if self.dynamics_backbone is None:
                self.dynamics_backbone = default_bb_cfg
            if self.prediction_backbone is None:
                self.prediction_backbone = default_bb_cfg
            if self.afterstate_dynamics_backbone is None:
                self.afterstate_dynamics_backbone = default_bb_cfg
            if self.chance_encoder_backbone is None:
                self.chance_encoder_backbone = default_bb_cfg
        else:
            # Final defaults if nothing provided
            dense_default = {"type": "dense"}
            if self.representation_backbone is None:
                self.representation_backbone = BackboneConfigFactory.create(
                    dense_default
                )
            if self.dynamics_backbone is None:
                self.dynamics_backbone = BackboneConfigFactory.create(dense_default)
            if self.prediction_backbone is None:
                self.prediction_backbone = BackboneConfigFactory.create(dense_default)
            if self.afterstate_dynamics_backbone is None:
                self.afterstate_dynamics_backbone = BackboneConfigFactory.create(
                    dense_default
                )
            if self.chance_encoder_backbone is None:
                self.chance_encoder_backbone = BackboneConfigFactory.create(
                    dense_default
                )
        # Mixin: Noisy
        self.parse_noisy_params()

        # Training
        self.games_per_generation: int = self.parse_field("games_per_generation", 100)
        self.value_loss_factor: float = self.parse_field("value_loss_factor", 1.0)
        self.to_play_loss_factor: float = self.parse_field("to_play_loss_factor", 1.0)
        self.train_value_iterations: int = self.parse_field("train_value_iterations", 1)
        self.train_policy_iterations: int = self.parse_field(
            "train_policy_iterations", 1
        )

        # Mixin: Search (MCTS)
        self.parse_search_params()

        self.temperature_schedule = self.parse_schedule_config(
            "temperature_schedule",
            defaults={"type": "stepwise", "steps": [5], "values": [1.0, 0.0]},
        )

        self.clip_low_prob: float = self.parse_field("clip_low_prob", 0.0)

        self.value_loss_function = self.parse_field("value_loss_function", F.mse_loss)
        self.reward_loss_function = self.parse_field("reward_loss_function", F.mse_loss)
        self.policy_loss_function = self.parse_field(
            "policy_loss_function", F.cross_entropy
        )
        self.to_play_loss_function = self.parse_field(
            "to_play_loss_function", F.cross_entropy
        )

        self.unroll_steps: int = self.parse_field("unroll_steps", 5)

        # Mixin: Distributional (Support Range) - Moved to top

        self.lr_ratio: float = self.parse_field("lr_ratio", float("inf"))
        self.transfer_interval: int = self.parse_field("transfer_interval", 1000)

        self.reanalyze_ratio: float = self.parse_field("reanalyze_ratio", 0.0)
        self.reanalyze_method: bool = self.parse_field("reanalyze_method", "mcts")
        self.reanalyze_tau: float = self.parse_field("reanalyze_tau", 0.3)
        self.injection_frac: float = self.parse_field(
            "injection_frac", 0.0
        )  # 0.25 for unplugged
        self.reanalyze_noise: bool = self.parse_field(
            "reanalyze_noise", False
        )  # true for gumbel
        self.reanalyze_update_priorities: bool = self.parse_field(
            "reanalyze_update_priorities", False
        )  # default false for most implementations

        # Mixin: Consistency
        self.parse_consistency_params()

        self.mask_absorbing = self.parse_field("mask_absorbing", False)

        # Mixin: Value Prefix
        # Moved up (stochastic, num_chance, etc.)

        self.action_embedding_dim = self.parse_field("action_embedding_dim", 32)
        self.single_action_plane = self.parse_field("single_action_plane", False)

        self.latent_viz_method = self.parse_field("latent_viz_method", "pca")
        self.latent_viz_interval = self.parse_field(
            "latent_viz_interval", 1
        )  # how often within learn() to update buffer

    def _verify_game(self):
        pass

    def parse_backbone_config(self, field_name: str) -> BackboneConfig:
        bb_dict = self.parse_field(field_name, default=None, required=False)
        if bb_dict is None:
            return None
        return BackboneConfigFactory.create(bb_dict)

    def parse_head_config(
        self, field_name: str, head_cfg_cls: Type[HeadConfig]
    ) -> HeadConfig:
        head_dict = self.parse_field(field_name, default=None, required=False)
        if head_dict is None:
            # Return a default config for that head type if not specified?
            # Or just empty dict
            return head_cfg_cls({})
        return head_cfg_cls(head_dict)
