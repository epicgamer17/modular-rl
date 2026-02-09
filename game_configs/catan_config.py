from wrappers import (
    ActionMaskInInfoWrapper,
    AppendAgentSelectionWrapper,
    FrameStackWrapper,
    TwoPlayerPlayerPlaneWrapper,
)
from .game_config import GameConfig
from custom_gym_envs.envs.catan import (
    env as catan_env,
    CatanAECEnv,
)


def make_env(
    num_players=2,
    map_type="TOURNAMENT",
    vps_to_win=10,
    representation="vector",
    invalid_action_reward=-10,
    render_mode="rgb_array",
    auto_play_single_action=False,
):
    env = catan_env(
        render_mode=render_mode,
        num_players=num_players,
        map_type=map_type,
        vps_to_win=vps_to_win,
        representation=representation,
        invalid_action_reward=invalid_action_reward,
        auto_play_single_action=auto_play_single_action,
    )
    env = ActionMaskInInfoWrapper(env)
    env = FrameStackWrapper(env, 4, channel_first=False)
    env = AppendAgentSelectionWrapper(env)
    return env


class CatanConfig(GameConfig):
    def __init__(self, make_env=make_env):
        super(CatanConfig, self).__init__(
            max_score=1,
            min_score=-1,
            is_discrete=True,
            is_image=False,
            is_deterministic=False,
            has_legal_moves=True,
            perfect_information=False,
            multi_agent=True,
            num_players=2,
            make_env=make_env,
        )


class SinglePlayerCatanConfig(GameConfig):
    def __init__(self, make_env=None):
        super(SinglePlayerCatanConfig, self).__init__(
            max_score=1,
            min_score=-1,
            is_discrete=True,
            is_image=False,
            is_deterministic=False,
            has_legal_moves=True,
            perfect_information=False,
            multi_agent=False,
            num_players=1,
            make_env=make_env,
        )
