from configs.agents.rainbow_dqn import RainbowConfig
from utils.schedule import ScheduleConfig


class DuelingDQNConfig(RainbowConfig):
    def __init__(self, config_dict, game_config):
        super(DuelingDQNConfig, self).__init__(config_dict, game_config)

        self.width: int = self.parse_field("width", 128)
        self.noisy_sigma: float = 0
        self.deuling: bool = self.parse_field("dueling", False)

        self.discount_factor: float = self.parse_field("discount_factor", 0.99)
        self.soft_update: bool = False
        self.transfer_interval: int = 1
        self.replay_interval: int = self.parse_field("replay_interval", 4)
        self.per_alpha: float = 0
        self.per_beta_schedule = ScheduleConfig.constant(0.0)
        self.per_epsilon: float = 0
        self.n_step: int = 1
        self.atom_size: int = 1

        self.v_min = game_config.min_score
        self.v_max = game_config.max_score

    def _verify_game(self):
        assert self.game.is_discrete, "Rainbow only supports discrete action spaces"
