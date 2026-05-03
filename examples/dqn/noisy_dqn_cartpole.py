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
from functional.losses import bellman_error, mse_loss
from functional.targets import standard_td_target
from functional.action_selection import (
    standard_selector,
    scalar_extractor,
    with_epsilon_greedy,
    get_linear_epsilon,
)
from functional.optimizer import apply_gradients
from functional.network import hard_update_target_network
from networks.noisy_linear import NoisyLinear

# Constants
BATCH_SIZE = 128
GAMMA = 0.99
NOISY_SIGMA = 0.5
LEARNING_RATE = 1e-3
MAX_STEPS = 120_000
UPDATE_FREQ = 4
BUFFER_CAPACITY = 50000
MIN_BUFFER_SIZE = 500
TARGET_NET_UPDATE_FREQ = 100
SEED = 42

# Seeding for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class DQN(nn.Module):
    def __init__(self, input_shape: Tuple, num_actions: int):
        super().__init__()
        self.l1 = NoisyLinear(input_shape[0], 512, sigma_init=NOISY_SIGMA)
        self.l2 = NoisyLinear(512, 512, sigma_init=NOISY_SIGMA)
        self.l3 = NoisyLinear(512, num_actions, sigma_init=NOISY_SIGMA)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

    def reset_noise(self):
        """Propagates the noise reset to all NoisyLinear layers."""
        for module in self.children():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


# --- 1. Initialization (Defining the State) ---
env = gym.make("CartPole-v1")
obs_shape = env.observation_space.shape
num_actions = env.action_space.n
device = torch.device("cpu")

model = DQN(obs_shape, num_actions).to(device)
target_model = DQN(obs_shape, num_actions).to(device)
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

action_selector = partial(standard_selector, extractor_fn=scalar_extractor)

# Initialize W&B
wandb.init(
    project="noisy-dqn-cartpole",
    config={
        "batch_size": BATCH_SIZE,
        "gamma": GAMMA,
        "learning_rate": LEARNING_RATE,
        "buffer_capacity": BUFFER_CAPACITY,
    }
)

# --- 2. The Monolithic Loop (The Imperative Shell) ---

for step in range(MAX_STEPS):
    # 2. Act (Pure function)
    with torch.inference_mode():
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)

        # Resample noisy nets if using them!
        model.reset_noise()

        _, action = action_selector(
            model=model,
            target_model=None,
            obs=obs_tensor,
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

        # Resample noise before calculating loss to prevent overfitting to a single noise vector during the batch loss calculation
        model.reset_noise()
        target_model.reset_noise()

        # Calculate Loss & Gradients
        loss, info_dict = bellman_error(
            model,
            target_model,
            batch,
            action_selector,
            partial(standard_td_target, gamma=batch["gamma"].to(device)),
            loss_fn=mse_loss,
        )
        loss = loss.mean()

        # Apply Updates
        optimizer = apply_gradients(optimizer, loss)

        if step % 100 == 0:
            # W&B handles scalars and histograms of tensors (like priorities) automatically.
            log_dict = info_dict.copy()
            log_dict.update({"loss": loss.item()})
            wandb.log(log_dict, step=step)

    # 4. Target Network Update
    if step % TARGET_NET_UPDATE_FREQ == 0:
        hard_update_target_network(model, target_model)
