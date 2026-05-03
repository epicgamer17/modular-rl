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

from functional.buffer import (
    init_per_buffer,
    sample_per,
    update_priorities,
    circular_write_strategy,
    with_per_tracking,
    get_linear_beta,
)
from functional.losses import bellman_error, with_per_weights, mse_loss
from functional.targets import standard_td_target
from functional.action_selection import (
    standard_selector,
    scalar_extractor,
    with_epsilon_greedy,
    get_linear_epsilon,
)
from functional.optimizer import apply_gradients
from functional.network import hard_update_target_network

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
SEED = 42

# PER Constants
ALPHA = 0.6
BETA_START = 0.4
BETA_FRAMES = 100000

# Seeding for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class DQN(nn.Module):
    def __init__(self, input_shape: Tuple, num_actions: int):
        super().__init__()
        self.l1 = nn.Linear(input_shape[0], 512)
        self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


# --- 1. Initialization (Defining the State) ---
env = gym.make("CartPole-v1")
obs_shape = env.observation_space.shape
num_actions = env.action_space.n
device = torch.device("cpu")

model = DQN(obs_shape, num_actions).to(device)
target_model = DQN(obs_shape, num_actions).to(device)
target_model.load_state_dict(model.state_dict())

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

buffer_state = init_per_buffer(
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
# Compose circular write with PER tracking
per_add_transition = with_per_tracking(circular_write_strategy)

obs, info = env.reset(seed=SEED)
stat_episode_return = 0.0
rng_key = torch.Generator(device=device)
rng_key.manual_seed(SEED)

dqn_selector = partial(standard_selector, extractor_fn=scalar_extractor)
action_selector = with_epsilon_greedy(dqn_selector)

# Initialize W&B
wandb.init(
    project="prioritized-replay-dqn-cartpole",
    config={
        "batch_size": BATCH_SIZE,
        "gamma": GAMMA,
        "learning_rate": LEARNING_RATE,
        "alpha": ALPHA,
    }
)

# --- 2. The Monolithic Loop (The Imperative Shell) ---
for step in range(MAX_STEPS):

    # 1. Calculate Epsilon dynamically for this step
    current_epsilon = get_linear_epsilon(step, EPS_START, EPS_END, EPS_DECAY_FRAMES)

    # 2. Act
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
    buffer_state = per_add_transition(buffer_state, transition)

    # Update state for next tick
    obs = next_obs

    if terminated or truncated:
        wandb.log({"episode_return": stat_episode_return}, step=step)
        obs, info = env.reset()
        stat_episode_return = 0.0

    # --- 3. The Update Loop ---
    if step > MIN_BUFFER_SIZE and step % UPDATE_FREQ == 0:
        # Anneal Beta
        beta = get_linear_beta(step, BETA_START, 1.0, BETA_FRAMES)
        beta_tensor = torch.tensor(beta, dtype=torch.float32, device=device)

        # Sample with PER
        batch, tree_indices, is_weights = sample_per(
            buffer_state, BATCH_SIZE, beta=beta_tensor
        )

        # Wrap loss function with PER weights
        per_loss_fn = with_per_weights(mse_loss, is_weights.to(device))

        # Calculate Loss & Gradients
        # bellman_error returns whatever loss_fn returns (loss, info)
        loss, info_dict = bellman_error(
            model,
            target_model,
            batch.to(device),
            dqn_selector,
            partial(standard_td_target, gamma=batch["gamma"].to(device)),
            loss_fn=per_loss_fn,
        )

        # Apply Updates
        optimizer = apply_gradients(optimizer, loss)

        # Update PER Priorities
        buffer_state = update_priorities(
            buffer_state, tree_indices, info_dict["priorities"], alpha=ALPHA
        )

        if step % 100 == 0:
            # W&B handles scalars and histograms of tensors (like priorities) automatically.
            log_dict = info_dict.copy()
            log_dict.update({"loss": loss.item(), "epsilon": current_epsilon, "beta": beta})
            wandb.log(log_dict, step=step)

    # 4. Target Network Update
    if step % TARGET_NET_UPDATE_FREQ == 0:
        hard_update_target_network(model, target_model)
