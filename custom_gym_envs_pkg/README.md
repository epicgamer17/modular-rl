# Custom Gym Environments

A standalone package of custom Gymnasium environments for games and problems used in RL research.

## Installation

```bash
cd custom_gym_envs_pkg
pip install -e .
```

## Available Environments

| Environment | Description | Observation | Action Space |
|-------------|-------------|-------------|--------------|
| TicTacToe | Classic 3x3 game | Board state (3x3) | Discrete(9) |
| Connect4 | Four-in-a-row | Board state (6x7) | Discrete(7) |
| Checkers | Draughts variant | Board state (8x8) | Discrete(32) |
| Catan | Resource trading game | Multiple boards | Discrete(actions) |
| Game2048 | Number merging puzzle | Grid (4x4) | Discrete(4) |
| LeducHoldem | Simplified poker | Card state | Discrete(actions) |
| MatchingPennies | Game theory classic | Binary | Discrete(2) |
| MississippiMarbles | Custom card game | Card state | Discrete(actions) |
| SlipperyGridWorld | Stochastic navigation | Position | Discrete(4) |
| GridWorld | Deterministic navigation | Position | Discrete(4) |
| ArmedBandits | Classic MAB | None | Discrete(n) |
| NonstationaryArmedBandits | Time-varying MAB | None | Discrete(n) |
| Wardrobe | Classification task | Image | Discrete(classes) |

## Usage

```python
import gymnasium as gym
import custom_gym_envs

# Create environment
env = gym.make('TicTacToe-v0')

# Standard Gym interface
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

## Environment Features

All environments support:
- Standard Gymnasium API
- Proper seeding for reproducibility
- Multi-agent mode (where applicable)
- Rendering (where applicable)

## PettingZoo Integration

Multi-agent games (TicTacToe, Connect4, etc.) implement the PettingZoo API:

```python
from custom_gym_envs.envs import tictactoe

env = tictactoe.env()
env.reset()

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    action = policy(observation, agent)
    env.step(action)
```

## Adding New Environments

1. Create environment class in `custom_gym_envs/envs/`
2. Register in `custom_gym_envs/envs/__init__.py`
3. Add entry point in `setup.py`
4. Create config in `game_configs/`

Example:
```python
from gymnasium import Env
from gymnasium.spaces import Discrete, Box

class MyEnv(Env):
    def __init__(self):
        self.action_space = Discrete(4)
        self.observation_space = Box(low=0, high=1, shape=(4,))
    
    def reset(self, seed=None, options=None):
        # Initialize environment
        return observation, info
    
    def step(self, action):
        # Execute action
        return observation, reward, terminated, truncated, info
```
