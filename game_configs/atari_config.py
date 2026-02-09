from .game_config import GameConfig


def make_env(render_mode=None):
    # This is a placeholder, as Atari usually requires a game name.
    # We'll assume the user provides make_env if they use AtariConfig directly,
    # but we can try to make a default one if we had a default game.
    # For now, we'll just raise an error if called without a specific env.
    raise NotImplementedError(
        "AtariConfig requires a specific make_env for the chosen game."
    )


class AtariConfig(GameConfig):
    def __init__(self, make_env=make_env):
        super(AtariConfig, self).__init__(
            max_score=10,  # FROM CATEGORICAL DQN PAPER
            min_score=-10,
            is_discrete=True,
            is_image=True,
            is_deterministic=False,  # if no frameskip, then deterministic
            has_legal_moves=False,
            perfect_information=True,  # although it is not deterministic, it is so close to it that it is considered perfect information
            multi_agent=False,
            num_players=1,
            # has_intermediate_rewards=True,
            make_env=make_env,
        )
