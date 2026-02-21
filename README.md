# Modular RL Architecture

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A high-performance, modular framework for Reinforcement Learning research, designed with **Strict Separation of Concerns** and **Perfect Polymorphism**. This repository provides robust implementations of state-of-the-art RL algorithms, including MuZero, PPO, and Rainbow DQN, optimized for both research flexibility and execution speed.

## 🌟 Overview

The "Modular RL Architecture" project solves the complexity of modern RL development by isolating neural network math from environment logic and optimization pipelines. Whether you are experimenting with latent world models or tuning proximal policy updates, this framework ensures that components are pluggable, testable, and scalable.

### Key Philosophy
- **The Blind Learner**: Optimization is decoupled from network architecture.
- **Game Logic Isolation**: Neural networks are pure mathematical functions; rules live in action selectors.
- **Opaque Tokens**: Actor/Search trees treat network states as opaque, preserving abstraction boundaries.

---

## 📸 Visuals

### Rainbow DQN Performance
![CartPole Training](file:///Users/jonathanlamontange-kratz/Documents/GitHub/rl-stuff/readme-figs/Rainbow_ClassicControl_CartPole-v1-episode-154.mp4)
*Example: Rainbow DQN mastering CartPole-v1.*

---

## 🚀 Features & Algorithms

### 🌈 Rainbow DQN (Integrated Improvements)
Mix and match components via configuration:
- **DQN** & **Double DQN** (Overestimation bias reduction)
- **Prioritized Experience Replay** (Important transition sampling)
- **Dueling DQN** (Value/Advantage separation)
- **Noisy Networks** (Adaptive exploration)
- **N-Step Returns** & **Categorical DQN (C51)**

### ♟️ Model-Based & Policy Gradient
- **MuZero**: Planning with learned environment dynamics.
- **AlphaZero**: MCTS integrated with deep policy/value networks.
- **PPO (Proximal Policy Optimization)**: Stable and reliable policy gradients.
- **NFSP (Neural Fictitious Self-Play)**: Nash equilibrium approximation for multi-agent games.

### 🎮 Supported Environments
- **Classic Control**: CartPole, Acrobot, LunarLander.
- **Board Games**: Tic-Tac-Toe, Connect 4.
- **Card Games**: LeDuc Hold'em.
- **Custom**: Easy to plug in any Gymnasium-compatible environment.

---

## 🛠️ Installation

### Prerequisites
- **OS**: macOS (fully supported), Linux, or Windows.
- **Python**: 3.10 or higher.
- **Hardware**: CUDA/MPS supported (automatically detected).

### Setup Commands
```bash
# Clone the repository
git clone https://github.com/epicgamer17/modular-rl.git
cd modular-rl

# Install dependencies
pip install -r requirements.txt

# Install as editable package
pip install -e .
```

---

## 📖 Usage Examples

### Training a Rainbow Agent
```python
from agents.rainbow_dqn import RainbowDQN
from agent_configs.rainbow_config import RainbowConfig
from configs.games.cartpole_config import CartPoleConfig

# 1. Initialize configurations
config = RainbowConfig(learning_rate=0.00025, buffer_size=100_000)
game_config = CartPoleConfig()

# 2. Setup and train
agent = RainbowDQN(config, game_config)
agent.train(total_steps=100_000)
```

### Running MuZero via Launcher
For performance-critical tasks, use the `launcher.py` pattern to ensure optimal memory and thread affinity:
```bash
python launcher.py run_muzero --env CartPole-v1 --config configs/muzero_default.yaml
```

---

## 🧪 Testing

We maintain a rigorous test suite using `pytest`.

```bash
# Run all core tests
pytest tests/

# Run a specific algorithm smoke test
pytest tests/test_muzero_smoke.py
```

---

## 🤝 Contributing

We are currently **not accepting external contributions** on this specific branch. This repository serves as a focused research environment. For major suggestions, please open an issue for discussion.

---

## 🏗️ Project Structure

```text
├── agents/             # Optimization (Learners) & Action Selectors
├── modules/            # Pure PyTorch Compute Graphs (Backbones, Heads)
├── replay_buffers/     # High-performance Transition Fact Stores
├── search/             # CPU-bound Imagination (MCTS)
├── losses/             # Vectorized Loss Pipelines
├── configs/            # Game and Agent hyperparameters
└── custom_gym_envs/    # Specialized RL environments
```

---

## 🩺 Support & Community

For bug reports, feature requests, or questions regarding the implementation details, please use the **[GitHub Issue Tracker](https://github.com/epicgamer17/modular-rl/issues)**. 

---

## 🎓 Authors & Credits

- **Primary Author**: epicgamer17
- **Contributors**: Ezra Huang (minor early contributions).
- **Core Resources**: 
    - [Rainbow is all you need](https://github.com/Curt-Park/rainbow-is-all-you-need)
    - [37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
    - See the `papers/` directory for full academic references.

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details (or standard MIT terms if file is missing).
