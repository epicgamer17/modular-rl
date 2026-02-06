# RL Research

Recreations and implementations of state-of-the-art reinforcement learning algorithms from research papers. This repository contains working implementations of DQN variants, MuZero, PPO, NFSP, and more.

## Implemented Algorithms

### DQN Variants (via RainbowDQN)
These components can be mixed and matched through configuration:

1. **DQN** - Deep Q-Network with experience replay
2. **Double DQN** - Reduces overestimation bias by decoupling action selection and evaluation
3. **Prioritized Experience Replay** - Samples important transitions more frequently
4. **Dueling DQN** - Separates state value and action advantage estimation
5. **Noisy Networks** - Adaptive exploration through learned noise parameters
6. **N-Step DQN** - Multi-step bootstrap targets for faster propagation
7. **Categorical DQN (C51)** - Learns distribution over returns instead of point estimates
8. **Rainbow DQN** - Combines all six improvements above

### Other Algorithms

9. **Ape-X** - Distributed prioritized experience replay with multiple actors
10. **NFSP (Neural Fictitious Self-Play)** - Learns approximate Nash equilibrium in imperfect information games
    - Can also be used to train Rainbow on multi-agent deterministic games (Tic-Tac-Toe, Connect 4) by setting anticipatory parameter to 1.0
11. **PPO** - Proximal Policy Optimization for stable policy gradient learning
12. **AlphaZero** - MCTS combined with deep neural networks for perfect information games
13. **MuZero** - Model-based planning without requiring environment dynamics

## Environments

### Currently Implemented
- Tic-Tac-Toe
- CartPole
- Connect 4
- Mississippi Marbles
- LeDuc Hold'em Poker

### Future Directions
See `papers.txt` for the full list of potential environments and research challenges.

## Quick Start

### Installation

```bash
pip install -r requirements.txt
pip install -e .
```

### Training an Agent

```python
from agents.rainbow_dqn import RainbowDQN
from agent_configs.rainbow_config import RainbowConfig
from game_configs.cartpole_config import CartPoleConfig

config = RainbowConfig()
agent = RainbowDQN(config, CartPoleConfig())
agent.train(episodes=1000)
```

## Project Structure

- `agents/` - Agent implementations (Rainbow DQN, MuZero, PPO, NFSP, etc.)
- `agent_configs/` - Configuration classes for all algorithms
- `modules/` - Neural network building blocks
- `replay_buffers/` - Experience replay implementations with prioritized sampling
- `search/` - MCTS and tree search algorithms
- `losses/` - Loss functions for different training objectives
- `game_configs/` - Environment-specific configurations
- `custom_gym_envs_pkg/` - Custom Gymnasium environments
- `experiments/` - Training runs, checkpoints, and results
- `tests/` - Unit and integration tests

## Key Papers

This codebase implements techniques from these papers (see `papers/` for PDFs):

- **MuZero**: https://arxiv.org/pdf/1911.08265.pdf
- **Rainbow**: https://arxiv.org/pdf/1710.02298.pdf
- **AlphaZero**: https://arxiv.org/pdf/1712.01815.pdf
- **PPO**: https://arxiv.org/pdf/1707.06347.pdf
- **NFSP**: https://arxiv.org/pdf/1603.01121.pdf

Full paper list in `papers.txt`.

## Configuration System

All agents use a consistent dataclass-based configuration system. Example:

```python
from dataclasses import dataclass

@dataclass
class RainbowConfig:
    learning_rate: float = 0.00025
    buffer_size: int = 100_000
    batch_size: int = 32
    gamma: float = 0.99
    n_step: int = 3
    v_min: float = -10.0
    v_max: float = 10.0
```

## Testing

Run the test suite:

```bash
pytest tests/
```

Interactive notebooks for debugging are available in `test_notebooks/`.

## Research Directions

Algorithms to explore next:
- Muesli
- DreamerV3
- R2D2
- NGU (Never Give Up)
- Agent 57
- CFR / DeepCFR for imperfect information
- StarCraft League training

See `papers.txt` for complete list of research ideas and paper links.
