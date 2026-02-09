from .game_config import GameConfig
import gymnasium as gym
import custom_gym_envs


def make_env(render_mode=None):
    env = gym.make("custom_gym_envs/Game2048-v0")
    return env


class Game2048Config(GameConfig):
    def __init__(self, make_env=make_env):
        super(Game2048Config, self).__init__(
            max_score=2**16,
            min_score=0,
            is_discrete=True,
            is_image=True,
            is_deterministic=False,
            has_legal_moves=True,
            perfect_information=True,
            multi_agent=False,
            num_players=1,
            # has_intermediate_rewards=True,
            make_env=make_env,
        )
