class GameConfig:
    def __init__(
        self,
        max_score,
        min_score,
        is_discrete,
        is_image,
        is_deterministic,
        has_legal_moves,
        perfect_information,
        multi_agent,
        num_players,
        # has_intermediate_rewards,
        num_actions,
        make_env=None,
    ):
        self.max_score = max_score
        self.min_score = min_score
        self.is_discrete = is_discrete  # can just check the action space type instead of setting manually if the env is passed in (ALSO COULD DO THIS IN THE BASE GAME CONFIG)
        self.num_actions = num_actions
        # self.observation_space = observation_space
        self.is_image = is_image
        self.is_deterministic = is_deterministic
        # self.num_players = num_players (might not need this idk) <- it would likely be for muzero but could also be for rainbow and stuff when they play multiplayer games (like connect 4)
        self.has_legal_moves = has_legal_moves
        self.perfect_information = perfect_information
        self.multi_agent = multi_agent
        self.num_players = num_players
        # self.has_intermediate_rewards = has_intermediate_rewards
        self.make_env = make_env

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, GameConfig):
            return False

        return (
            self.max_score == o.max_score
            and self.min_score == o.min_score
            and self.is_discrete == o.is_discrete
            and self.is_image == o.is_image
            and self.is_deterministic == o.is_deterministic
            and self.has_legal_moves == o.has_legal_moves
            and self.perfect_information == o.perfect_information
            and self.multi_agent == o.multi_agent
            and self.num_players == o.num_players
            # and self.has_intermediate_rewards == o.has_intermediate_rewards
        )
