from gymnasium.envs.registration import register

register(
    id="custom_gym_envs/Catan-v0",
    entry_point="custom_gym_envs.envs:CatanAECEnv",
    max_episode_steps=1000,
)

register(
    id="custom_gym_envs/MatchingPennies-v0",
    entry_point="custom_gym_envs.envs:MatchingPenniesEnv",
    max_episode_steps=100,
)

register(
    id="custom_gym_envs/Connect4-v0",
    entry_point="custom_gym_envs.envs:Connect4Env",
    max_episode_steps=300,
    reward_threshold=1.0,
    kwargs={"size": (6, 7), "win_length": 4},
)


register(
    id="custom_gym_envs/TicTacToe-v0",
    entry_point="custom_gym_envs.envs:TicTacToeEnv",
    max_episode_steps=300,
    reward_threshold=1.0,
    kwargs={"size": 3, "win_length": 3},
)


register(
    id="custom_gym_envs/GridWorld-v0",
    entry_point="custom_gym_envs.envs:TicTacToeEnv",
    max_episode_steps=300,
    reward_threshold=1.0,
    kwargs={"size": 3, "win_length": 3},
)

register(
    id="custom_gym_envs/MississippiMarbles-v0",
    entry_point="custom_gym_envs.envs:MississippiMarblesEnv",
    max_episode_steps=30000,
    reward_threshold=1.0,
    kwargs={"players": 6},
)

register(
    id="custom_gym_envs/LeducHoldem-v0",
    entry_point="custom_gym_envs.envs:LeducHoldemEnv",
)

register(
    id="custom_gym_envs/ArmedBandits-v0",
    entry_point="custom_gym_envs.envs:ArmedBanditsEnv",
)

register(
    id="custom_gym_envs/NonstationairyArmedBandits-v0",
    entry_point="custom_gym_envs.envs:NonstationairyArmedBanditsEnv",
    kwargs={"mode": "random"},
)

register(
    id="custom_gym_envs/Game2048-v0",
    entry_point="custom_gym_envs.envs:Game2048Env",
)

register(
    id="custom_gym_envs/SlipperyGrid-v0",
    entry_point="custom_gym_envs.envs:SlipperyGridEnv",
)
