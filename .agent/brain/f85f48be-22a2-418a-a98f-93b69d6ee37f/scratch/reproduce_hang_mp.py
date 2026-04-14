import torch
import torch.multiprocessing as mp
import gymnasium as gym
import sys
import os
import time

# Add the project root to sys.path
sys.path.append(os.getcwd())

from registries.muzero import make_muzero_mlp_network, make_muzero_actor_engine, make_muzero_search_engine
from core import infinite_ticks
from executors.workers.actor_worker import ActorWorker

def worker_loop(actor_engine):
    print("Worker loop started")
    worker = ActorWorker(actor_engine)
    print("Worker initialized, playing sequence...")
    worker.play_sequence()
    print("Worker finished sequence")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    print("Imports done")
    DEVICE = torch.device("cpu")
    ENV_ID = "CartPole-v1"
    env = gym.make(ENV_ID)
    num_actions = env.action_space.n
    obs_dim = env.observation_space.shape

    print("Creating network...")
    agent_network = make_muzero_mlp_network(obs_dim, num_actions, device=DEVICE)
    agent_network.share_memory()

    print("Creating search engine...")
    search_engine = make_muzero_search_engine(num_actions, num_simulations=5, device=DEVICE)

    print("Creating actor engine...")
    actor_engine = make_muzero_actor_engine(env, agent_network, search_engine, None, obs_dim, num_actions, 1, device=DEVICE)

    print("Launching process...")
    p = mp.Process(target=worker_loop, args=(actor_engine,))
    p.start()
    print("Process started, waiting...")
    p.join(timeout=30)
    if p.is_alive():
        print("Process HUNG!")
        p.terminate()
    else:
        print("Process finished successfully")
