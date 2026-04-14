import numpy as np
from envs.factories.tictactoe import tictactoe_factory
from components.experts.tictactoe import TicTacToeBestAgent
from pettingzoo.utils.env import AECEnv

env = tictactoe_factory()
agent_0 = TicTacToeBestAgent()
agent_1 = TicTacToeBestAgent()

def simulate(env, p0, p1):
    env.reset()
    total_rewards = {"player_1": 0.0, "player_2": 0.0}
    for active_agent in env.agent_iter():
        obs, reward, terminated, truncated, info = env.last()
        total_rewards[active_agent] += reward
        if terminated or truncated:
            env.step(None)
            continue
            
        if active_agent == "player_1":
            action = p0.select_actions((obs, info), info)
        else:
            action = p1.select_actions((obs, info), info)
        env.step(action)
    return total_rewards

p0_wins=0; p1_wins=0; draws=0
for _ in range(1000):
    r = simulate(env, agent_0, agent_1)
    if r["player_1"] > 0: p0_wins += 1
    elif r["player_2"] > 0: p1_wins += 1
    else: draws += 1
print(f"Expert vs Expert -> P0 Win: {p0_wins/1000:.2f}, P1 Win: {p1_wins/1000:.2f}, Draw: {draws/1000:.2f}")

from components.experts.tictactoe import TicTacToeRandomAgent
rand_agent = TicTacToeRandomAgent()

p0_wins=0; p1_wins=0; draws=0
for _ in range(1000):
    r = simulate(env, agent_0, rand_agent)
    if r["player_1"] > 0: p0_wins += 1
    elif r["player_2"] > 0: p1_wins += 1
    else: draws += 1
print(f"Expert(P0) vs Random(P1) -> P0 Win: {p0_wins/1000:.2f}, P1 Win: {p1_wins/1000:.2f}, Draw: {draws/1000:.2f}")

p0_wins=0; p1_wins=0; draws=0
for _ in range(1000):
    r = simulate(env, rand_agent, agent_1)
    if r["player_1"] > 0: p0_wins += 1
    elif r["player_2"] > 0: p1_wins += 1
    else: draws += 1
print(f"Random(P0) vs Expert(P1) -> P0 Win: {p0_wins/1000:.2f}, P1 Win: {p1_wins/1000:.2f}, Draw: {draws/1000:.2f}")
