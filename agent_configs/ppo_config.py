from .base_config import Config, DistributionalConfig, NoisyConfig
from .actor_config import ActorConfig
from .critic_config import CriticConfig


class PPOConfig(Config, DistributionalConfig, NoisyConfig):
    def __init__(
        self,
        config_dict,
        game_config,
        actor_config: ActorConfig,
        critic_config: CriticConfig,
    ):
        super(PPOConfig, self).__init__(config_dict, game_config)
        self.parse_distributional_params()
        self.parse_noisy_params()
        # config_dict["optimizer"] = -1
        # config_dict["adam_epsilon"] = -1
        # config_dict["learning_rate"] = -1
        # config_dict["clipnorm"] = -1
        # config_dict["loss_function"] = -1
        # config_dict["training_iterations"] = -1
        # config_dict["min_replay_buffer_size"] = -1

        assert (
            not "replay_buffer_size" in config_dict
        ), "Replay buffer size must not be set for PPO as it is the same as steps per epoch"
        config_dict["replay_buffer_size"] = config_dict[
            "steps_per_epoch"
        ]  # times number of agents
        assert (
            not "minibatch_size" in config_dict
        ), "Minibatch size must not be set for PPO as it is the same as steps per epoch"
        config_dict["minibatch_size"] = config_dict["steps_per_epoch"]

        assert (
            "training_iterations" not in config_dict
        ), "Please set train_policy_iterations and train_value_iterations instead of training_iterations"

        # could change to just storing a config and accessing it as ppo_config.actor_config.learning_rate etc
        # self.actor_optimizer = actor_config.optimizer
        # self.actor_epsilon = actor_config.adam_epsilon
        # self.actor_learning_rate = actor_config.learning_rate
        # self.actor_clipnorm = actor_config.clipnorm
        self.actor = (
            actor_config  # maybe go back since it is not inherriting anything anymore
        )

        # could change to just storing a config and accessing it as ppo_config.critic_config.learning_rate etc
        # self.critic_optimizer = critic_config.optimizer
        # self.critic_epsilon = critic_config.adam_epsilon
        # self.critic_learning_rate = critic_config.learning_rate
        # self.critic_clipnorm = critic_config.clipnorm
        self.critic = (
            critic_config  # maybe go back since it is not inherriting anything anymore
        )

        # Network Arcitecture
        # COULD SET ALL ACTOR STUFF IN ACTOR CONFIG AND CRITIC STUFF IN CRITIC CONFIG FOR NETWORK ARCHITECTURE
        self.critic_conv_layers = self.parse_field("conv_layers", [])
        self.actor_conv_layers = self.parse_field("conv_layers", [])
        self.critic_dense_layer_widths = self.parse_field("critic_dense_layers", [])
        self.actor_dense_layer_widths = self.parse_field("actor_dense_layers", [])

        self.clip_param = self.parse_field("clip_param", 0.2)
        self.steps_per_epoch = self.parse_field("steps_per_epoch", 4800)

        # Override ReplayConfig params
        self.replay_buffer_size = self.parse_field(
            "replay_buffer_size", self.steps_per_epoch
        )

        self.train_policy_iterations = self.parse_field("train_policy_iterations", 5)
        self.train_value_iterations = self.parse_field("train_value_iterations", 5)
        self.target_kl = self.parse_field("target_kl", 0.02)
        # self.discount_factor parsed in Config
        self.gae_lambda = self.parse_field("gae_lambda", 0.98)
        self.entropy_coefficient = self.parse_field("entropy_coefficient", 0.01)
        self.critic_coefficient = self.parse_field("critic_coefficient", 0.5)

        self.clip_low_prob = self.parse_field("clip_low_prob", 0.00)

        assert not (
            self.game.is_image
            and self.actor_conv_layers is not None
            and self.critic_conv_layers is not None
        ), "Convolutional layers must be defined for image based games"

    def _verify_game(self):
        pass
