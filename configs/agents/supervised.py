from configs.base import (
    ConfigBase,
    OptimizationConfig,
    ReplayConfig,
)
from configs.modules.backbones.base import BackboneConfig
from configs.modules.backbones.factory import BackboneConfigFactory
from configs.modules.architecture_config import ArchitectureConfig
from modules.utils import (
    prepare_activations,
    prepare_kernel_initializers,
    kernel_initializer_wrapper,
)
from torch.optim import Optimizer, Adam


# TODO: MAKE THIS CLEANER AND DONT HAVE THE PREFIX EVERYWHERE
class SupervisedConfig(ConfigBase, OptimizationConfig, ReplayConfig):
    def __init__(self, config_dict):
        if "agent_type" not in config_dict:
            config_dict["agent_type"] = "supervised"
        super().__init__(config_dict)
        self.agent_type = self.parse_field("agent_type", "supervised")
        print("SupervisedConfig")
        # Parse shared architecture defaults
        arch_dict = self.parse_field("architecture", default={}, required=False)
        self.arch = ArchitectureConfig(arch_dict)
        self.learning_rate = self.parse_field("sl_learning_rate", 0.005)
        self.momentum = self.parse_field("sl_momentum", 0.9)
        self.loss_function = self.parse_field("sl_loss_function", required=True)
        self.clipnorm = self.parse_field("sl_clipnorm", 0)
        self.optimizer: Optimizer = self.parse_field("sl_optimizer", Adam)
        self.adam_epsilon = self.parse_field("sl_adam_epsilon", 1e-8)
        self.weight_decay = self.parse_field("sl_weight_decay", 0.0)
        self.training_steps = self.parse_field("training_steps", required=True)
        self.training_iterations = self.parse_field("sl_training_iterations", 1)
        self.num_minibatches = self.parse_field("sl_num_minibatches", 1)
        self.minibatch_size = self.parse_field("sl_minibatch_size", 32)
        self.min_replay_buffer_size = self.parse_field(
            "sl_min_replay_buffer_size", self.minibatch_size
        )
        self.replay_buffer_size = self.parse_field(
            "sl_replay_buffer_size", self.training_steps
        )
        self.activation = self.parse_field(
            "sl_activation", "relu", wrapper=prepare_activations
        )
        self.kernel_initializer = self.parse_field(
            "sl_kernel_initializer",
            None,
            required=False,
            wrapper=kernel_initializer_wrapper,
        )

        self.clip_low_prob = self.parse_field("sl_clip_low_prob", 0.00)

        self.noisy_sigma = self.parse_field("sl_noisy_sigma", 0)
        self.lr_schedule = self.parse_schedule_config(
            "sl_lr_schedule",
            defaults={"type": "constant", "initial": self.learning_rate},
        )

        # Backbone Configuration
        self.backbone: BackboneConfig = self.parse_backbone_config("sl_backbone")

        # Fallback/Default logic
        if self.backbone is None:
            # Default to a simple dense backbone if nothing provided
            self.backbone = BackboneConfigFactory.create(
                {
                    "type": "mlp",
                    "widths": self.parse_field("sl_dense_layer_widths", [128]),
                }
            )

        self.game = None

        # Backward compatibility for buffer factories if they look for standard names (without sl_ prefix)
        # We manually map them here so `create_standard_buffer` works
        self.n_step = 1
        self.discount_factor = 1.0
        self.per_alpha = 0
        self.per_beta_schedule = None

    def parse_backbone_config(self, field_name: str) -> BackboneConfig:
        bb_dict = self.parse_field(field_name, default=None, required=False)
        if bb_dict is None:
            return None
        return BackboneConfigFactory.create(bb_dict)
