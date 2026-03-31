from .game import GameConfig
import gymnasium as gym


def env_factory(render_mode=None):
    env = gym.make("CartPole-v1", render_mode=render_mode)
    return env


class CartPoleConfig(GameConfig):
    def __init__(self, env_factory=env_factory):
        super(CartPoleConfig, self).__init__(
            num_actions=2,
            max_score=500,
            min_score=0,
            is_discrete=True,
            is_image=False,
            is_deterministic=True,  # i think it is deterministic (pretty sure if you input the same actions the same thing will happen, it just has a random start state)
            has_legal_moves=False,
            perfect_information=True,
            multi_agent=False,
            num_players=1,
            # has_intermediate_rewards=True,
            env_factory=env_factory,
        )
