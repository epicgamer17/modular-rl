from .game import GameConfig


import gymnasium as gym


def env_factory(render_mode=None):
    return gym.make("Blackjack-v1", render_mode=render_mode)


class BlackjackConfig(GameConfig):
    def __init__(self, env_factory=env_factory):
        super(BlackjackConfig, self).__init__(
            num_actions=2,
            max_score=1,
            min_score=-1,
            is_discrete=True,
            is_image=False,
            is_deterministic=False,
            has_legal_moves=False,
            perfect_information=False,
            multi_agent=False,
            num_players=2,
            # has_intermediate_rewards=False,
            env_factory=env_factory,
        )
