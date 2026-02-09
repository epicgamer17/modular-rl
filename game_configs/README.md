# Game Configurations

Environment-specific configurations that define observation spaces, action spaces, rewards, and environment parameters for each game.

## Purpose

Game configs provide a standardized way to describe environments to agents. They handle:
- Observation preprocessing (normalization, stacking)
- Action space specifications
- Reward shaping and clipping
- Episode termination conditions
- Environment-specific hyperparameters

## Available Configs

| Config | Game | Type |
|--------|------|------|
| atari_config.py | Atari games | Image |
| cartpole_config.py | CartPole | Classic control |
| tictactoe_config.py | Tic-Tac-Toe | Discrete board |
| catan_config.py | Catan | Complex multi-agent |
| connect4_config.py | Connect 4 | Discrete board |
| game_2048_config.py | 2048 | Grid puzzle |
| leduc_holdem_config.py | Leduc Hold'em | Card game |
| matching_pennies_config.py | Matching Pennies | Game theory |
| mississippi_marbles_config.py | Mississippi Marbles | Card game |
| slippery_grid_world_config.py | Slippery Grid | Stochastic nav |
| blackjack_config.py | Blackjack | Card game |
| classiccontrol_config.py | Classic Control | Continuous |

## Base Game Config

All configs inherit from `SequenceConfig`:

```python
@dataclass
class SequenceConfig:
    name: str
    observation_shape: Tuple[int, ...]
    action_space: int
    max_episode_steps: int
    gamma: float = 0.99
    reward_clip: Optional[Tuple[float, float]] = None
```

## Usage

Pass game config when creating an agent:

```python
from game_configs.tictactoe_config import TicTacToeConfig
from agents.muzero import MuZero

game_config = TicTacToeConfig()
agent = MuZero(config, game_config)
```

## Atari Config

Special handling for Atari games with preprocessing options:

```python
from game_configs.atari_config import AtariConfig

config = AtariConfig(
    game='Pong',
    frame_skip=4,
    grayscale=True,
    frame_stack=4,
    noop_max=30
)
```

## Multi-Agent Configs

Games like Catan and Tic-Tac-Toe specify agent configuration:

```python
# TicTacToeConfig
num_agents = 2
agent_names = ['player_1', 'player_2']
observation_mode = 'both'  # What each player sees
```

## Observation Preprocessing

Configs can include observation transformations:

```python
class Game2048Config(SequenceConfig):
    def preprocess_observation(self, obs):
        # Normalize grid values
        return np.log2(obs + 1) / 16.0
```

## Adding New Games

1. Create config class inheriting from `SequenceConfig`
2. Define observation_shape, action_space, max_episode_steps
3. Add any game-specific parameters
4. Implement preprocess_observation if needed
