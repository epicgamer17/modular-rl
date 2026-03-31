from .game import GameConfig
import gymnasium as gym
import custom_gym_envs


def env_factory(render_mode=None):
    env = gym.make("custom_gym_envs/Game2048-v0")
    return env


class Game2048Config(GameConfig):
    def __init__(self, env_factory=env_factory):
        super(Game2048Config, self).__init__(
            num_actions=4,
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
            env_factory=env_factory,
        )
