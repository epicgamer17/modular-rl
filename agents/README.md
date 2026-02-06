# Agents

Core RL agent implementations. Each agent inherits from `BaseAgent` and implements the standard interface for training and inference.

## Available Agents

| Agent | File | Algorithm Type |
|-------|------|----------------|
| Rainbow DQN | `rainbow_dqn.py` | Value-based |
| MuZero | `muzero.py` | Model-based planning |
| MuZero Actor | `muzero_actor.py` | Distributed MuZero actor |
| MuZero Learner | `muzero_learner.py` | Distributed MuZero learner |
| NFSP | `nfsp.py` | Fictitious self-play |
| PPO | `ppo.py` | Policy gradient |
| Policy Imitation | `policy_imitation.py` | Behavioral cloning |
| Random | `random.py` | Random baseline |

## Base Agent Interface

All agents implement:

```python
class BaseAgent:
    def __init__(self, config, game_config):
        """Initialize agent with configuration."""
        pass
    
    def select_action(self, observation, training=True):
        """Select action given observation."""
        pass
    
    def train(self, episodes):
        """Train agent for specified episodes."""
        pass
    
    def save(self, path):
        """Save agent checkpoint."""
        pass
    
    def load(self, path):
        """Load agent checkpoint."""
        pass
```

## Agent Selection Guide

**Single-player, perfect information**: Rainbow DQN, PPO, MuZero
**Multi-player, perfect information**: AlphaZero, MuZero with multi-agent wrapper
**Imperfect information**: NFSP
**Continuous control**: PPO
**Sample efficiency matters**: MuZero (learns model)
**Training speed matters**: Rainbow DQN (simplest)

## Usage Examples

### Rainbow DQN
```python
from agents.rainbow_dqn import RainbowDQN
from agent_configs.rainbow_config import RainbowConfig
from game_configs.cartpole_config import CartPoleConfig

agent = RainbowDQN(RainbowConfig(), CartPoleConfig())
agent.train(episodes=1000)
```

### MuZero
```python
from agents.muzero import MuZero
from agent_configs.muzero_config import MuZeroConfig
from game_configs.tictactoe_config import TicTacToeConfig

agent = MuZero(MuZeroConfig(), TicTacToeConfig())
agent.train(episodes=500)
```

### NFSP (Multi-agent)
```python
from agents.nfsp import NFSP
from agent_configs.dqn.nfsp_config import NFSPConfig

agent = NFSP(NFSPConfig(), game_config)
agent.train(episodes=10000)  # Learns against itself
```

## Distributed Training

MuZero supports distributed training via separate actor and learner processes:

```python
# Actor process
from agents.muzero_actor import MuZeroActor
actor = MuZeroActor(config, game_config, replay_buffer)
actor.run()

# Learner process
from agents.muzero_learner import MuZeroLearner
learner = MuZeroLearner(config, replay_buffer)
learner.train()
```

## Implementation Notes

- Agents handle their own network architectures through `modules/`
- Experience replay is managed by agents using `replay_buffers/`
- MuZero uses `search/` for MCTS planning
- Checkpoints include model weights, optimizer state, and training statistics
