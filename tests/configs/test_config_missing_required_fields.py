import copy
import pytest

pytestmark = pytest.mark.unit

from configs.agents.ppo import PPOConfig
from configs.agents.muzero import MuZeroConfig
from configs.agents.rainbow_dqn import RainbowConfig
from configs.agents.supervised import SupervisedConfig
from configs.selectors import SelectorConfig, BaseSelectorConfig, DecoratorConfig
from configs.modules.output_strategies import CategoricalConfig
from configs.games.cartpole import CartPoleConfig
from configs.games.tictactoe import TicTacToeConfig


class TestPPOConfigMissingFields:
    def test_ppo_config_missing_action_selector_raises_value_error(
        self, make_ppo_config_dict, cartpole_game_config
    ):
        config_dict = make_ppo_config_dict()
        del config_dict["action_selector"]
        with pytest.raises(ValueError, match="action_selector"):
            PPOConfig(config_dict, cartpole_game_config)

    def test_ppo_config_distributional_missing_value_head_output_strategy(
        self, make_ppo_config_dict, cartpole_game_config
    ):
        config_dict = make_ppo_config_dict()
        config_dict["atom_size"] = 51
        config_dict["value_head"] = {}
        with pytest.raises(ValueError, match="output_strategy"):
            PPOConfig(config_dict, cartpole_game_config)

    def test_ppo_config_missing_steps_per_epoch_raises_key_error(
        self, make_ppo_config_dict, cartpole_game_config
    ):
        config_dict = make_ppo_config_dict()
        del config_dict["steps_per_epoch"]
        with pytest.raises(KeyError):
            PPOConfig(config_dict, cartpole_game_config)


class TestMuZeroConfigMissingFields:
    def test_muzero_config_missing_action_selector_raises_value_error(
        self, make_muzero_config_dict, tictactoe_game_config
    ):
        config_dict = make_muzero_config_dict()
        del config_dict["action_selector"]
        with pytest.raises(ValueError, match="action_selector"):
            MuZeroConfig(config_dict, tictactoe_game_config)


class TestRainbowConfigMissingFields:
    def test_rainbow_config_missing_action_selector_raises_value_error(
        self, make_rainbow_config_dict, cartpole_game_config
    ):
        config_dict = make_rainbow_config_dict()
        del config_dict["action_selector"]
        with pytest.raises(ValueError, match="action_selector"):
            RainbowConfig(config_dict, cartpole_game_config)


class TestSelectorConfigMissingFields:
    def test_selector_config_missing_base_raises_value_error(self):
        config_dict = {"decorators": []}
        with pytest.raises(ValueError, match="base"):
            SelectorConfig(config_dict)

    def test_base_selector_config_missing_type_raises_value_error(self):
        config_dict = {"kwargs": {}}
        with pytest.raises(ValueError, match="type"):
            BaseSelectorConfig(config_dict)

    def test_decorator_config_missing_type_raises_value_error(self):
        config_dict = {"kwargs": {}}
        with pytest.raises(ValueError, match="type"):
            DecoratorConfig(config_dict)

    def test_selector_config_missing_decorators_not_required(self):
        config_dict = {"base": {"type": "categorical"}}
        selector = SelectorConfig(config_dict)
        assert selector.base is not None
        assert selector.decorators == []


class TestCategoricalConfigMissingFields:
    def test_categorical_config_missing_num_classes_raises_value_error(self):
        config_dict = {"type": "categorical"}
        with pytest.raises(ValueError, match="num_classes"):
            CategoricalConfig(config_dict)

    def test_categorical_config_with_num_classes_succeeds(self):
        config_dict = {"type": "categorical", "num_classes": 10}
        categorical = CategoricalConfig(config_dict)
        assert categorical.num_classes == 10


class TestSupervisedConfigMissingFields:
    def test_supervised_config_missing_sl_loss_function_raises_value_error(
        self, make_supervised_config_dict
    ):
        config_dict = make_supervised_config_dict()
        del config_dict["sl_loss_function"]
        with pytest.raises(ValueError, match="sl_loss_function"):
            SupervisedConfig(config_dict)

    def test_supervised_config_missing_training_steps_raises_value_error(
        self, make_supervised_config_dict
    ):
        config_dict = make_supervised_config_dict()
        del config_dict["training_steps"]
        with pytest.raises(ValueError, match="training_steps"):
            SupervisedConfig(config_dict)


class TestNFSPConfigMissingFields:
    def test_nfsp_config_missing_action_selector_raises_value_error(
        self, make_nfsp_config_dict, cartpole_game_config
    ):
        from configs.agents.nfsp import NFSPDQNConfig

        config_dict = make_nfsp_config_dict()
        del config_dict["action_selector"]
        with pytest.raises(ValueError, match="action_selector"):
            NFSPDQNConfig(config_dict, cartpole_game_config)


class TestSelectorDecoratorChainMissingFields:
    def test_selector_with_decorator_missing_type_raises_error(self):
        config_dict = {
            "base": {"type": "categorical"},
            "decorators": [{"kwargs": {}}]
        }
        with pytest.raises(ValueError, match="type"):
            SelectorConfig(config_dict)
