from gymnasium.envs.registration import register

register(
    id="custom_gym_envs/Catan-v0",
    entry_point="atomic_rl.envs.custom.envs.catan:CatanAECEnv",
    max_episode_steps=1000,
)

register(
    id="custom_gym_envs/MatchingPennies-v0",
    entry_point="atomic_rl.envs.custom.envs.matching_pennies:MatchingPenniesEnv",
    max_episode_steps=100,
)

register(
    id="custom_gym_envs/LeducHoldem-v0",
    entry_point="atomic_rl.envs.custom.envs.leduc_holdem:LeducHoldemEnv",
)
