from .game_config import GameConfig


from pettingzoo.classic import connect4_v3


def make_env(render_mode=None):
    return connect4_v3.env(render_mode=render_mode)


class Connect4Config(GameConfig):
    def __init__(self, make_env=make_env):
        super(Connect4Config, self).__init__(
            max_score=1,
            min_score=-1,
            is_discrete=True,
            is_image=True,
            is_deterministic=True,
            has_legal_moves=True,
            perfect_information=True,
            multi_agent=True,
            num_players=2,
            # has_intermediate_rewards=False,
            make_env=make_env,
        )
