# Agents

Core RL agent implementations. Each agent inherits from `BaseAgent` and implements the standard interface for training and inference.

## Installation

Agents are installed automatically with the main package:

```bash
pip install -e .
```

## Available Agents

| Agent | File | Type | Status |
|-------|------|------|--------|
| Rainbow DQN | `rainbow_dqn.py` | Value-based | ✅ |
| NFSP | `nfsp.py` | Fictitious self-play | ✅ |
| PPO | `ppo.py` | Policy gradient | ✅ |
| Policy Imitation | `policy_imitation.py` | Behavioral cloning | ✅ |
| Random | `random.py` | Baseline | ✅ |

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

**Single-player, perfect information**: Rainbow DQN, PPO
**Multi-player, perfect information**: AlphaZero (using MCTS wrapper)
**Imperfect information**: NFSP
**Continuous control**: PPO
**Training speed matters**: Rainbow DQN (simplest)

## Usage Examples

### Rainbow DQN
```python
from agents.rainbow_dqn import RainbowDQN
from configs.base import RainbowConfig
from configs.games.cartpole_config import CartPoleConfig

agent = RainbowDQN(RainbowConfig(), CartPoleConfig())
agent.train(episodes=1000)
```

### NFSP (Multi-agent)
```python
from agents.nfsp import NFSP
from configs.base.nfsp_config import NFSPConfig

agent = NFSP(NFSPConfig(), game_config)
agent.train(episodes=10000)  # Learns against itself
```


## Implementation Notes

- Agents handle their own network architectures through `modules/`
- Experience replay is managed by agents using `replay_buffers/`
- Checkpoints include model weights, optimizer state, and training statistics
