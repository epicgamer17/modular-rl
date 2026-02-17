import unittest
import copy
import gymnasium as gym
from configs.games.game_config import GameConfig
from configs.agents.ppo import PPOConfig
from configs.agents.muzero import MuZeroConfig
from modules.world_models.muzero_world_model import MuzeroWorldModel


class MockEnv:
    def close(self):
        pass


class TestStrictConfig(unittest.TestCase):
    def setUp(self):
        self.game = GameConfig(
            max_score=100,
            min_score=0,
            is_discrete=True,
            is_image=False,
            is_deterministic=True,
            has_legal_moves=False,
            perfect_information=True,
            multi_agent=False,
            num_players=1,
            num_actions=5,
            make_env=lambda: MockEnv(),
        )

    def test_ppo_strict_value_head(self):
        # Case 1: atom_size > 1 but no output_strategy -> Should Raise ValueError
        base_dict = {
            "atom_size": 51,
            "support_range": 10,
            "value_head": {"neck": {"type": "identity"}},  # No output_strategy
            "policy_head": {"neck": {"type": "identity"}},
            "steps_per_epoch": 100,
            "train_policy_iterations": 1,
            "train_value_iterations": 1,
        }

        with self.assertRaises(ValueError) as cm:
            PPOConfig(copy.deepcopy(base_dict), self.game)
        self.assertIn("requires an explicit output_strategy", str(cm.exception))

        # Case 2: atom_size > 1, output_strategy provided with WRONG num_classes -> Should Override
        case2_dict = copy.deepcopy(base_dict)
        case2_dict["value_head"]["output_strategy"] = {
            "type": "categorical",
            "num_classes": 999,
        }
        config = PPOConfig(case2_dict, self.game)
        # Use attribute access if it's a config object, or dict access if it's a dict.
        # PPOConfig parses strategy into a dict for ValueHeadConfig?
        # No, ValueHeadConfig likely converts it to an OutputStrategyConfig object.
        # Let's check config.value_head.output_strategy type.
        # Typically configurations recursively parse into XConfig objects.
        # If accessing via config.value_head.output_strategy, it's NOT a dict.
        # It's a CategoricalConfig object. It has .num_classes attribute.
        self.assertEqual(config.value_head.output_strategy.num_classes, 51)

        # Case 3: atom_size = 1 (Scalar) -> No error
        case3_dict = {
            "atom_size": 1,
            "value_head": {"output_strategy": {"type": "scalar"}},
            "policy_head": {"neck": {"type": "identity"}},
            "steps_per_epoch": 100,
            "train_policy_iterations": 1,
            "train_value_iterations": 1,
        }
        config = PPOConfig(case3_dict, self.game)

    def test_muzero_strict_value_head(self):
        # Case 1: atom_size > 1 but no output_strategy -> Should Raise ValueError
        base_dict = {
            "world_model_cls": MuzeroWorldModel,
            "atom_size": 51,
            "support_range": 10,
            "value_head": {"neck": {"type": "identity"}},  # No output_strategy
            "policy_head": {"neck": {"type": "identity"}},
            "representation_backbone": {"type": "identity"},
            "dynamics_backbone": {"type": "identity"},
            "prediction_backbone": {"type": "identity"},
            "afterstate_dynamics_backbone": {"type": "identity"},
            "chance_encoder_backbone": {"type": "identity"},
        }
        with self.assertRaises(ValueError) as cm:
            MuZeroConfig(copy.deepcopy(base_dict), self.game)
        self.assertIn("requires an explicit output_strategy", str(cm.exception))

        # Case 2: atom_size > 1, output_strategy provided with WRONG num_classes -> Should Override
        case2_dict = copy.deepcopy(base_dict)
        case2_dict["value_head"]["output_strategy"] = {
            "type": "categorical",
            "num_classes": 999,
        }
        config = MuZeroConfig(case2_dict, self.game)
        self.assertEqual(config.value_head.output_strategy.num_classes, 51)


if __name__ == "__main__":
    unittest.main()
