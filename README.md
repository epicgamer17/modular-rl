# Modular RL Architecture

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A high-performance, modular framework for Reinforcement Learning research, designed with **Strict Separation of Concerns** and **Perfect Polymorphism**. This repository provides robust implementations of state-of-the-art RL algorithms, including MuZero, PPO, and Rainbow DQN, optimized for both research flexibility and execution speed.

## 🌟 Overview

The "Modular RL Architecture" project solves the complexity of modern RL development by isolating neural network math from environment logic and optimization pipelines. Whether you are experimenting with latent world models or tuning proximal policy updates, this framework ensures that components are pluggable, testable, and scalable.

### 🧠 Core Philosophy
This framework is built on the principle of **Strict Separation of Concerns**. We maintain a "Blind" boundary between components:
- **The Blind Learner**: The optimization pipeline (Learner) is completely blind to the neural network's architecture (LSTM, ResNet, etc.). It asks for raw math and computes the loss.
- **The Blind Actor/Tree**: The MCTS Tree and Actors treat the network's recurrent state as an **Opaque Token**, storing and passing it back without inspecting its contents.
- **Game Logic Isolation**: Neural networks are pure mathematical functions. Action masking and legal move filtering happen strictly *outside* the network in action selectors or search trees.

---

## 🏗️ Project Structure & Domain Boundaries

| Directory | Domain | Responsibility |
|---|---|---|
| `modules/` | **The Compute Graph** | Pure PyTorch Neural Networks. No knowledge of RL logic. |
| `agents/trainers/` | **Orchestration** | Interfaces with all components to run the training loop. |
| `agents/learner/` | **Optimization** | The Loss Pipeline and Optimizer (e.g., `UniversalLearner`). |
| `agents/action_selectors/`| **The Bridge** | Translates network math into environment actions; handles masking. |
| `replay_buffers/` | **The Fact Store** | High-performance transition storage with vectorized processing. |
| `search/` | **The Imagination** | CPU-bound MCTS (Backends: Python, C++, AOS). |
| `losses/` | **Math Kernels** | Vectorized loss computations (e.g., C51, TD, Value). |
| `custom_gym_envs_pkg/` | **Environments** | Repository for specialized and custom Gymnasium environments. |

---

## 🚀 Features & Algorithms

### 🌈 Rainbow DQN (Value-Based)
Integrated improvements including **Double DQN**, **Prioritized Experience Replay**, **Dueling Architectures**, **Noisy Networks**, **N-Step Returns**, and **Categorical DQN (C51)**.

### ♟️ MuZero & AlphaZero (Model-Based)
State-of-the-art planning with learned environment dynamics. Supports **Stochastic World Models**, **Gumbel MuZero**, and **EfficientZero** optimizations.

### 🎯 Policy Gradient & Others
- **PPO (Proximal Policy Optimization)**: Stable and reliable policy gradients with generalized advantage estimation.
- **NFSP (Neural Fictitious Self-Play)**: Nash equilibrium approximation for multi-agent games.
- **Imitation Learning**: Behavior cloning and policy imitation.

---

## 🛠️ Installation

### Prerequisites
- **Python**: 3.10 or higher.
- **Hardware**: CUDA/MPS supported (automatically detected).

### Setup Commands
```bash
# Clone the repository
git clone https://github.com/epicgamer17/modular-rl.git
cd modular-rl

# Install core dependencies
pip install -r requirements.txt

# Install the framework and custom environments in editable mode
pip install -e .
pip install -e custom_gym_envs_pkg/
```

---

## 📖 Usage Example: MuZero on Tic-Tac-Toe

The framework uses a `Trainer` + `Config` paradigm. Below is a minimal example of running a MuZero smoke test:

```python
import torch
from agents.trainers.muzero_trainer import MuZeroTrainer
from configs.agents.muzero import MuZeroConfig
from configs.games.tictactoe import TicTacToeConfig

# 1. Initialize configurations
game_config = TicTacToeConfig()
params = {
    "training_steps": 1000,
    "num_simulations": 50,
    "unroll_steps": 5,
    "batch_size": 256,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}
config = MuZeroConfig(config_dict=params, game_config=game_config)

# 2. Setup and train
trainer = MuZeroTrainer(
    config=config,
    env=game_config.env_factory(),
    device=torch.device(params["device"])
)
trainer.train()
```

---

## 🧪 Testing

We maintain a rigorous test suite using `pytest` markers for isolation:

```bash
# Run fast unit tests
pytest tests/ -m unit

# Run integration tests (component interactions)
pytest tests/ -m integration

# Run long training smoke tests
pytest tests/ -m slow
```

---

## 🎓 Authors & Credits

- **Primary Author**: epicgamer17
- **Contributors**: Ezra Huang
- **Core Resources**: 
    - [Rainbow is all you need](https://github.com/Curt-Park/rainbow-is-all-you-need)
    - [37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
    - [MuZero: Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/abs/1911.08265)

---

## 📄 License

This project is licensed under the **MIT License**.
