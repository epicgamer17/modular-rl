# Agent Configurations

Configuration classes for all RL agents in the project. These dataclass-based configs provide type safety, validation, and easy parameter tuning.

## Structure

```
agent_configs/
├── base_config.py              # Base configuration with common parameters
├── actor_config.py             # Actor component configuration
├── critic_config.py            # Critic network configuration
├── muzero_config.py            # MuZero-specific settings
├── ppo_config.py               # PPO hyperparameters
├── sl_config.py                # Supervised learning config
└── dqn/                        # DQN variant configurations
    ├── dqn_config.py
    ├── double_dqn_config.py
    ├── dueling_dqn_config.py
    ├── categorical_dqn_config.py
    ├── noisy_dqn_config.py
    ├── per_dqn_config.py
    ├── n_step_dqn_config.py
    ├── rainbow_config.py
    └── nfsp_config.py
```

## Usage

Create a config and pass it to an agent:

```python
from agent_configs.rainbow_config import RainbowConfig
from agents.rainbow_dqn import RainbowDQN

config = RainbowConfig(
    learning_rate=0.00025,
    buffer_size=100000,
    batch_size=32
)
agent = RainbowDQN(config, game_config)
```

## Configuration Inheritance

All configs inherit from `BaseConfig` which provides:
- Device selection (CPU/CUDA)
- Logging and checkpointing settings
- Common training parameters (discount factor, etc.)

Example hierarchy:
```
BaseConfig
├── DQNConfig
│   ├── DoubleDQNConfig
│   ├── DuelingDQNConfig
│   └── ...
├── MuZeroConfig
├── PPOConfig
└── NFSPConfig
```

## Key Parameters by Algorithm

### Rainbow DQN
- `buffer_size` - Experience replay capacity
- `batch_size` - Training batch size
- `learning_rate` - Adam learning rate
- `gamma` - Discount factor
- `n_step` - Multi-step return length
- `v_min`, `v_max` - Categorical distribution bounds
- `atom_size` - Number of distribution atoms

### MuZero
- `num_simulations` - MCTS simulations per action
- `td_steps` - Temporal difference horizon
- `unroll_steps` - Unroll length for dynamics model
- `support_size` - Value/reward support size

### PPO
- `n_epochs` - Update epochs per batch
- `clip_epsilon` - Clipping parameter
- `gae_lambda` - GAE lambda for advantage estimation

## Experiment Tracking

Configs can be saved and loaded for reproducibility:

```python
# Save config
config.save('experiment_config.yaml')

# Load config
loaded_config = RainbowConfig.load('experiment_config.yaml')
```
