from .game_config import GameConfig
import sys

sys.path.append("../../")
from wrappers import (
    ActionMaskInInfoWrapper,
    ChannelLastToFirstWrapper,
    FrameStackWrapper,
    InitialMovesWrapper,
    TwoPlayerPlayerPlaneWrapper,
)
from pettingzoo.classic import tictactoe_v3


def make_env(render_mode="rgb_array"):
    env = tictactoe_v3.env(render_mode=render_mode)
    env = ActionMaskInInfoWrapper(env)
    env = FrameStackWrapper(env, 4, channel_first=False)
    env = TwoPlayerPlayerPlaneWrapper(env, channel_first=False)
    env = ChannelLastToFirstWrapper(env)
    # env = InitialMovesWrapper(env, [0, 1, 2, 3, 4, 6, 5])
    return env


class TicTacToeConfig(GameConfig):
    def __init__(self, make_env=make_env):
        super(TicTacToeConfig, self).__init__(
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
