import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from typing import Tuple
import numpy as np
import random
import wandb
from functools import partial

from functional.buffer import init_buffer, circular_write_strategy, uniform_sample
from functional.losses import bellman_error, cross_entropy_loss
from functional.action_selection import (
    standard_selector,
    categorical_extractor,
    with_epsilon_greedy,
    get_linear_epsilon,
)
from functional.optimizer import apply_gradients
from functional.network import hard_update_target_network
from functional.visualization import log_distributional_metrics

from functional.targets import categorical_td_target

# Constants
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY_FRAMES = 50000
LEARNING_RATE = 1e-3
MAX_STEPS = 120_000
UPDATE_FREQ = 4
BUFFER_CAPACITY = 50000
MIN_BUFFER_SIZE = 500
TARGET_NET_UPDATE_FREQ = 100
V_MIN = 0
V_MAX = 500
ATOM_SIZE = 51
SUPPORT = torch.linspace(V_MIN, V_MAX, ATOM_SIZE)

SEED = 42

# Seeding for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class CategoricalDQN(nn.Module):
    def __init__(self, input_shape: Tuple, num_actions: int, atom_size: int = 51):
        super().__init__()
        self.l1 = nn.Linear(input_shape[0], 512)
        self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, num_actions * atom_size)
        self.num_actions = num_actions
        self.atom_size = atom_size

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x.view(-1, self.num_actions, self.atom_size)


# --- 1. Initialization (Defining the State) ---
env = gym.make("CartPole-v1")
obs_shape = env.observation_space.shape
num_actions = env.action_space.n
device = torch.device("cpu")

model = CategoricalDQN(obs_shape, num_actions).to(device)
target_model = CategoricalDQN(obs_shape, num_actions).to(device)
target_model.load_state_dict(model.state_dict())

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

buffer_state = init_buffer(
    capacity=BUFFER_CAPACITY,
    shapes={
        "obs": obs_shape,
        "action": (1,),
        "reward": (1,),
        "terminated": (1,),
        "truncated": (1,),
        "next_obs": obs_shape,
        "gamma": (1,),
    },
    device=device,
)

obs, info = env.reset(seed=SEED)
stat_episode_return = 0.0
rng_key = torch.Generator(device=device)
rng_key.manual_seed(SEED)

categorical_selector = partial(
    standard_selector, extractor_fn=partial(categorical_extractor, support=SUPPORT)
)
action_selector = with_epsilon_greedy(categorical_selector)

# Initialize W&B
wandb.init(
    project="categorical-dqn-cartpole",
    config={
        "batch_size": BATCH_SIZE,
        "gamma": GAMMA,
        "learning_rate": LEARNING_RATE,
        "atom_size": ATOM_SIZE,
    }
)

# --- 2. The Monolithic Loop (The Imperative Shell) ---

for step in range(MAX_STEPS):

    # 1. Calculate Epsilon dynamically for this step
    current_epsilon = get_linear_epsilon(step, EPS_START, EPS_END, EPS_DECAY_FRAMES)

    # 2. Act (Pure function)
    with torch.inference_mode():
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)

        _, action, rng_key = action_selector(
            model=model,
            target_model=None,
            obs=obs_tensor,
            epsilon=current_epsilon,
            num_actions=num_actions,
            generator=rng_key,
        )
        action = action.item()

    # 2. Step Env
    next_obs, reward, terminated, truncated, info = env.step(action)
    stat_episode_return += reward

    # 3. Add to Buffer
    transition = {
        "obs": obs,
        "action": [action],
        "reward": [reward],
        "terminated": [terminated],
        "truncated": [truncated],
        "next_obs": next_obs,
        "gamma": [GAMMA],
    }
    buffer_state, _ = circular_write_strategy(buffer_state, transition)

    # Update state for next tick
    obs = next_obs

    if terminated or truncated:
        wandb.log({"episode_return": stat_episode_return}, step=step)
        obs, info = env.reset()
        stat_episode_return = 0.0

    # --- 3. The Update Loop ---
    if step > MIN_BUFFER_SIZE and step % UPDATE_FREQ == 0:
        # Sample
        batch = uniform_sample(buffer_state, rng_key, BATCH_SIZE)

        # Calculate Loss & Gradients
        loss, info_dict = bellman_error(
            model,
            target_model,
            batch,
            categorical_selector,
            partial(
                categorical_td_target,
                gamma=batch["gamma"].to(device),
                support=SUPPORT,
                v_min=V_MIN,
                v_max=V_MAX,
                atom_size=ATOM_SIZE,
            ),
            loss_fn=cross_entropy_loss,
        )
        loss = loss.mean()

        # Apply Updates
        optimizer = apply_gradients(optimizer, loss)

        if step % 100 == 0:
            # Log all metrics from info_dict. W&B handles scalars and histograms of tensors.
            # We exclude 'predictions' and 'priorities' from the direct log to handle them specially.
            log_dict = {k: v for k, v in info_dict.items() if k not in ["predictions", "priorities"]}
            log_dict.update({"epsilon": current_epsilon})

            # Add distributional metrics (Expected Q every 100 steps, Chart every 1000 steps)
            log_dict.update(log_distributional_metrics(info_dict, SUPPORT, step, log_chart=(step % 1000 == 0)))
            
            wandb.log(log_dict, step=step)

    # 4. Target Network Update
    if step % TARGET_NET_UPDATE_FREQ == 0:
        hard_update_target_network(model, target_model)
