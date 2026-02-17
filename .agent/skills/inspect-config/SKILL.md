---
name: inspect-config
description: Locates and reads configuration files for agents (e.g., MuZero, PPO) or games (e.g., CartPole, Atari). Use this to check hyperparameters or network settings.
---

# Goal
Find and display the content of configuration files in `agent_configs/` or `configs/games/`.

# Instructions
1.  Determine if the user is asking about an "agent" config or a "game" config.
2.  Search the respective directory (`agent_configs` or `game_configs`) for matching files.
3.  Read and return the content of the best match.

# Examples
**User:** "What are the learning rate settings for MuZero?"
**Agent:** (Calls skill with `config_type="agent"`, `query="muzero"`)

**User:** "Check the CartPole environment config."
**Agent:** (Calls skill with `config_type="game"`, `query="cartpole"`)