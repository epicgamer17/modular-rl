from typing import Type
from .base import AgentConfig
from configs.base import DistributionalConfig, NoisyConfig
from configs.modules.heads.policy import PolicyHeadConfig
from configs.modules.heads.value import ValueHeadConfig
from configs.modules.heads.base import HeadConfig
from configs.modules.backbones.base import BackboneConfig
from configs.modules.backbones.factory import BackboneConfigFactory
from configs.modules.architecture_config import ArchitectureConfig
from configs.base import ActorConfig
from configs.base import CriticConfig


class PPOConfig(AgentConfig, DistributionalConfig, NoisyConfig):
    def __init__(
        self,
        config_dict,
        game_config,
    ):
        super(PPOConfig, self).__init__(config_dict, game_config)
        self.parse_distributional_params()
        self.parse_noisy_params()

        # Parse shared architecture defaults
        arch_dict = self.parse_field("architecture", default={}, required=False)
        self.arch = ArchitectureConfig(arch_dict)

        # Parse Actor/Critic configs from dict
        actor_dict = self.parse_field("actor_config", default={}, required=False)
        actor_config = ActorConfig(actor_dict)

        critic_dict = self.parse_field("critic_config", default={}, required=False)
        critic_config = CriticConfig(critic_dict)

        self.actor = actor_config

        self.critic = critic_config

        # Policy Head - Inject num_classes from game/config
        policy_dict = self.parse_field("policy_head", default={}, required=False) or {}

        # Determine num_actions dynamically
        num_actions = self.game.num_actions

        if num_actions is not None:
            # Inject into output_strategy
            pol_strat = policy_dict.get("output_strategy", {"type": "categorical"})
            if "num_classes" not in pol_strat:
                pol_strat["num_classes"] = num_actions
            policy_dict["output_strategy"] = pol_strat

        self.policy_head: PolicyHeadConfig = PolicyHeadConfig(policy_dict)

        # Value Head - Inject num_atoms if distributional
        value_dict = self.parse_field("value_head", default={}, required=False) or {}

        if self.atom_size > 1:
            val_strat = value_dict.get("output_strategy", None)
            if val_strat is None:
                raise ValueError(
                    f"Distributional PPO (atom_size={self.atom_size}) requires an explicit output_strategy for the value head."
                )

            # Force num_classes to be atom_size
            val_strat["num_classes"] = self.atom_size
            value_dict["output_strategy"] = val_strat

        self.value_head: ValueHeadConfig = ValueHeadConfig(value_dict)

        self.clip_param = self.parse_field("clip_param", 0.2)
        self.steps_per_epoch = self.parse_field("steps_per_epoch", 4800)
        self.num_minibatches = self.parse_field("num_minibatches", 4)

        self.train_policy_iterations = self.parse_field("train_policy_iterations", 5)
        self.train_value_iterations = self.parse_field("train_value_iterations", 5)

        # Calculate properties for UniversalLearner
        config_dict["minibatch_size"] = self.steps_per_epoch // self.num_minibatches
        config_dict["replay_buffer_size"] = self.steps_per_epoch
        config_dict["training_iterations"] = (
            max(self.train_policy_iterations, self.train_value_iterations)
            * self.num_minibatches
        )

        # Assign to self properties
        self.minibatch_size = config_dict["minibatch_size"]
        self.replay_buffer_size = config_dict["replay_buffer_size"]
        self.training_iterations = config_dict["training_iterations"]

        self.target_kl = self.parse_field("target_kl", 0.02)
        # self.discount_factor parsed in Config/AgentConfig
        self.gae_lambda = self.parse_field("gae_lambda", 0.98)
        self.entropy_coefficient = self.parse_field("entropy_coefficient", 0.01)
        self.critic_coefficient = self.parse_field("critic_coefficient", 0.5)

        self.clip_low_prob = self.parse_field("clip_low_prob", 0.00)

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
            return head_cfg_cls({})
        return head_cfg_cls(head_dict)
