import torch
import torch.multiprocessing as mp
import gymnasium as gym
import sys
import os

# Add the project root to sys.path
sys.path.append(os.getcwd())

from registries.muzero import make_muzero_mlp_network, make_muzero_actor_engine, make_muzero_search_engine
from core import infinite_ticks

print("Imports done")
DEVICE = torch.device("cpu")
env = gym.make("CartPole-v1")
num_actions = env.action_space.n
obs_dim = env.observation_space.shape

print("Creating network...")
agent_network = make_muzero_mlp_network(obs_dim, num_actions, device=DEVICE)

print("Creating search engine...")
search_engine = make_muzero_search_engine(num_actions, num_simulations=5, device=DEVICE)

print("Creating actor engine...")
engine = make_muzero_actor_engine(env, agent_network, search_engine, None, obs_dim, num_actions, 1, device=DEVICE)

print("Engine created, running first step...")
for res in engine.step(infinite_ticks()):
    print("Step done")
    break
print("Done")
