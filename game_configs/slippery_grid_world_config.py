import gymnasium as gym
from .game_config import GameConfig
import custom_gym_envs


def make_env(render_mode=None):
    env = gym.make("custom_gym_envs/SlipperyGrid-v0", render_mode=render_mode)
    return env


class SlipperyGridWorldConfig(GameConfig):
    def __init__(self):
        super(SlipperyGridWorldConfig, self).__init__(
            max_score=10,
            min_score=-20,
            is_discrete=True,
            is_image=True,
            is_deterministic=False,  # i think it is deterministic (pretty sure if you input the same actions the same thing will happen, it just has a random start state)
            has_legal_moves=True,
            perfect_information=True,
            multi_agent=False,
            num_players=1,
            # has_intermediate_rewards=True,
            make_env=make_env,
        )
