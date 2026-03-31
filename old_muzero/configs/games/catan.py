from old_muzero.utils.wrappers import (
    ActionMaskInInfoWrapper,
    AppendAgentSelectionWrapper,
    FrameStackWrapper,
    TwoPlayerPlayerPlaneWrapper,
)
from old_muzero.configs.games.game import GameConfig
from custom_gym_envs.envs.catan import (
    env as catan_env,
    CatanAECEnv,
    ACTION_SPACE_SIZE,
    SpatialEncoding,
)

from custom_gym_envs.envs.catan_placement import (
    env as catan_placement_env,
    # TODO: This is not correct technically if we toggle on road placing
    PLACEMENT_SETTLEMENT_ACTIONS as CATAN_PLACEMENT_ACTION_SPACE_SIZE,
)


def catan_aec_env_factory(
    num_players=2,
    map_type="BASE",
    vps_to_win=10,
    representation="image",
    invalid_action_reward=-10,
    render_mode="rgb_array",
    auto_play_single_action=False,
    bandit_mode=False,
    spatial_encoding: SpatialEncoding = "axial",
    include_validity_mask: bool = True,
    include_last_roll: bool = True,
    include_game_phase: bool = True,
    include_bank_state: bool = True,
    include_road_distance: bool = True,
):
    env = catan_env(
        render_mode=render_mode,
        num_players=num_players,
        map_type=map_type,
        vps_to_win=vps_to_win,
        representation=representation,
        invalid_action_reward=invalid_action_reward,
        auto_play_single_action=auto_play_single_action,
        bandit_mode=bandit_mode,
        spatial_encoding=spatial_encoding,
        include_validity_mask=include_validity_mask,
        include_last_roll=include_last_roll,
        include_game_phase=include_game_phase,
        include_bank_state=include_bank_state,
        include_road_distance=include_road_distance,
    )
    env = ActionMaskInInfoWrapper(env)
    env = FrameStackWrapper(env, 4, channel_first=False)
    env = AppendAgentSelectionWrapper(env)
    return env


def placement_catan_env_factory(
    num_players=2,
    map_type="BASE",
    vps_to_win=10,
    representation="image",
    invalid_action_reward=-10,
    render_mode="rgb_array",
    auto_play_single_action=False,
    bandit_mode=False,
    spatial_encoding: SpatialEncoding = "axial",
    include_validity_mask: bool = True,
    include_last_roll: bool = False,
    include_game_phase: bool = False,
    include_bank_state: bool = False,
    include_road_distance: bool = False,
    include_roads_in_action_space: bool = False,
    auto_play_roads: bool = True,
):
    env = catan_placement_env(
        render_mode=render_mode,
        num_players=num_players,
        map_type=map_type,
        vps_to_win=vps_to_win,
        representation=representation,
        invalid_action_reward=invalid_action_reward,
        auto_play_single_action=auto_play_single_action,
        bandit_mode=bandit_mode,
        spatial_encoding=spatial_encoding,
        include_validity_mask=include_validity_mask,
        include_last_roll=include_last_roll,
        include_game_phase=include_game_phase,
        include_bank_state=include_bank_state,
        include_road_distance=include_road_distance,
        include_roads_in_action_space=include_roads_in_action_space,
        auto_play_roads=auto_play_roads,
    )
    env = ActionMaskInInfoWrapper(env)
    env = FrameStackWrapper(env, 4, channel_first=False)
    env = AppendAgentSelectionWrapper(env)
    return env


class CatanConfig(GameConfig):
    def __init__(self, env_factory=catan_aec_env_factory):
        super(CatanConfig, self).__init__(
            num_actions=ACTION_SPACE_SIZE,
            max_score=1,
            min_score=-1,
            is_discrete=True,
            is_image=False,
            is_deterministic=False,
            has_legal_moves=True,
            perfect_information=False,
            multi_agent=True,
            num_players=2,
            env_factory=catan_aec_env_factory,
        )


class PlacementCatanConfig(GameConfig):
    def __init__(self, env_factory=placement_catan_env_factory):
        super(PlacementCatanConfig, self).__init__(
            num_actions=len(CATAN_PLACEMENT_ACTION_SPACE_SIZE),
            max_score=1,
            min_score=-1,
            is_discrete=True,
            is_image=False,
            is_deterministic=False,
            has_legal_moves=True,
            perfect_information=False,
            multi_agent=True,
            num_players=2,
            env_factory=placement_catan_env_factory,
        )


class SinglePlayerCatanConfig(GameConfig):
    def __init__(self, env_factory=None):
        super(SinglePlayerCatanConfig, self).__init__(
            num_actions=ACTION_SPACE_SIZE,
            max_score=1,
            min_score=-1,
            is_discrete=True,
            is_image=False,
            is_deterministic=False,
            has_legal_moves=True,
            perfect_information=False,
            multi_agent=False,
            num_players=1,
            env_factory=env_factory,
        )
