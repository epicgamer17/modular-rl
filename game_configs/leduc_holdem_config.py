from .game_config import SequenceConfig


from pettingzoo.classic import leduc_holdem_v4


def make_env(render_mode=None):
    return leduc_holdem_v4.env(render_mode=render_mode)


class LeducHoldemConfig(SequenceConfig):
    def __init__(self, make_env=make_env):
        super(LeducHoldemConfig, self).__init__(
            max_score=10,
            min_score=-10,
            is_discrete=True,
            is_image=False,
            is_deterministic=False,
            has_legal_moves=False,
            perfect_information=False,
            multi_agent=True,
            num_players=2,
            # has_intermediate_rewards=False,
            make_env=make_env,
        )
