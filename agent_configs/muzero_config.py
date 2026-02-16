from typing import Callable

from torch import Tensor

# from muzero.muzero_world_model import MuzeroWorldModel
from .base_config import (
    Config,
    SearchConfig,
    ValuePrefixConfig,
    ConsistencyConfig,
    DistributionalConfig,
    NoisyConfig,
)
from configs.modules.backbones.base import BackboneConfig
from configs.modules.backbones.factory import BackboneConfigFactory
from losses.basic_losses import CategoricalCrossentropyLoss, MSELoss
from utils.utils import tointlists
import copy


class MuZeroConfig(
    Config,
    SearchConfig,
    ValuePrefixConfig,
    ConsistencyConfig,
    DistributionalConfig,
    NoisyConfig,
):
    def __init__(self, config_dict, game_config):
        super(MuZeroConfig, self).__init__(config_dict, game_config)

        self.world_model_cls = self.parse_field("world_model_cls", None, required=True)
        # self.norm_type parsed in Config
        # SAME AS VMIN AND VMAX?
        self.known_bounds = self.parse_field(
            "known_bounds", default=None, required=False
        )

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
        self.reward_backbone: BackboneConfig = self.parse_backbone_config(
            "reward_backbone"
        )
        self.to_play_backbone: BackboneConfig = self.parse_backbone_config(
            "to_play_backbone"
        )

        # Actor/Critic Backbones (for Prediction Head)
        # TODO: right now the afterstate backbones use these as well, maybe switch that for easier readability
        self.actor_backbone: BackboneConfig = self.parse_backbone_config(
            "actor_backbone"
        )
        self.critic_backbone: BackboneConfig = self.parse_backbone_config(
            "critic_backbone"
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
        # self.weight_decay parsed in Config

        # Mixin: Search (MCTS)
        self.parse_search_params()

        self.temperatures = self.parse_field("temperatures", [1.0, 0.0])
        self.temperature_updates = self.parse_field("temperature_updates", [5])
        self.temperature_with_training_steps = self.parse_field(
            "temperature_with_training_steps", False
        )
        assert len(self.temperatures) == len(self.temperature_updates) + 1

        self.clip_low_prob: float = self.parse_field("clip_low_prob", 0.0)

        self.value_loss_function = self.parse_field("value_loss_function", MSELoss())

        self.reward_loss_function = self.parse_field("reward_loss_function", MSELoss())

        self.policy_loss_function = self.parse_field(
            "policy_loss_function", CategoricalCrossentropyLoss()
        )

        self.to_play_loss_function = self.parse_field(
            "to_play_loss_function", CategoricalCrossentropyLoss()
        )

        # self.n_step parsed in Config
        # self.discount_factor parsed in Config
        self.unroll_steps: int = self.parse_field("unroll_steps", 5)

        # self.per_alpha, beta, epsilon etc parsed in Config

        # Mixin: Distributional (Support Range)
        self.parse_distributional_params()

        self.multi_process: bool = self.parse_field("multi_process", True)
        self.num_workers: int = self.parse_field("num_workers", 4)
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
        self.parse_value_prefix_params()

        self.q_estimation_method: str = self.parse_field("q_estimation_method", "v_mix")

        self.stochastic: bool = self.parse_field("stochastic", False)
        self.use_true_chance_codes: bool = self.parse_field(
            "use_true_chance_codes", False
        )
        self.num_chance: int = self.parse_field("num_chance", 32)
        self.sigma_loss = self.parse_field("sigma_loss", CategoricalCrossentropyLoss())
        self.vqvae_commitment_cost_factor: float = self.parse_field(
            "vqvae_commitment_cost_factor", 1.0
        )

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
