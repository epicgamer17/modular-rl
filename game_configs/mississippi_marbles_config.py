from .game_config import SequenceConfig


from pettingzoo.classic import mississippi_marbles_v3


def make_env(render_mode=None):
    return mississippi_marbles_v3.env(render_mode=render_mode)


class MississippiMarblesConfig(SequenceConfig):
    def __init__(self, make_env=make_env):
        super(MississippiMarblesConfig, self).__init__(
            max_score=20000,  # technically infinite, but this is a good enough approximation
            min_score=0,
            is_discrete=True,
            is_image=False,
            is_deterministic=False,
            has_legal_moves=True,
            perfect_information=False,
            multi_agent=True,
            num_players=2,  # could be more
            # has_intermediate_rewards=False,
            make_env=make_env,
        )
