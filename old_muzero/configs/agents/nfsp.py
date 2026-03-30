from old_muzero.configs.agents.rainbow_dqn import RainbowConfig
from old_muzero.configs.agents.supervised import SupervisedConfig
from old_muzero.configs.base import Config, ConfigBase
from torch.optim import Optimizer, Adam


class NFSPDQNConfig(ConfigBase):
    def __init__(self, config_dict, game_config):
        # Config type should be a DQN Type
        if "agent_type" not in config_dict:
            config_dict["agent_type"] = "nfsp"
        super(NFSPDQNConfig, self).__init__(config_dict)
        print("NFSPDQNConfig")
        self.game = game_config
        self.num_players = self.game.num_players
        self.rl_configs = [
            RainbowConfig(config_dict, game_config) for _ in range(self.num_players)
        ]
        self.sl_configs = [
            SupervisedConfig(config_dict) for _ in range(self.num_players)
        ]
        self.training_steps = self.parse_field("training_steps", 100000)

        self.replay_interval = self.parse_field("replay_interval", 16)
        self.num_minibatches = self.parse_field("num_minibatches", 1)

        self.anticipatory_param = self.parse_field("anticipatory_param", 0.1)

        # Whether all players share the same networks and buffers
        self.shared_networks_and_buffers = self.parse_field(
            "shared_networks_and_buffers", True
        )

        # Multi-processing settings
        self.multi_process = self.parse_field("multi_process", False)
        self.num_workers = self.parse_field("num_workers", 1)

        # Optional observation dtype override (defaults to None, auto-detected)
        self.observation_dtype = self.parse_field(
            "observation_dtype", None, required=False
        )

        self._verify_game()

    def _verify_game(self):
        assert self.game.is_discrete, "NFSP only supports discrete action spaces"
        assert (
            self.game.make_env is not None
        ), "NFSP requires a valid environment factory (make_env) in the game config"
