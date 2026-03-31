from .game import GameConfig


def env_factory(render_mode=None, max_cycles=100):
    from custom_gym_envs_pkg.custom_gym_envs.envs.matching_pennies import (
        env as matching_pennies_env,
    )

    return matching_pennies_env(render_mode=render_mode, max_cycles=max_cycles)


class MatchingPenniesConfig(GameConfig):
    def __init__(self, env_factory=env_factory):
        super(MatchingPenniesConfig, self).__init__(
            num_actions=2,
            max_score=1,
            min_score=-1,
            is_discrete=True,
            is_image=False,
            is_deterministic=False,
            has_legal_moves=False,
            perfect_information=False,
            multi_agent=True,
            num_players=2,
            # has_intermediate_rewards=False,
            env_factory=env_factory,
        )
