# Neural Network Modules

Reusable neural network components and building blocks for RL agents. Provides modular architectures that can be composed for different algorithms.

## Installation

Modules are included in the main package:

```bash
pip install -e .
```

## Structure

```
modules/
├── utils.py                     # Module utilities and helpers
├── backbones/                   # Feature extraction backbones (ResNet, Conv, MLP)
├── heads/                       # Semantic heads (Policy, Value, Q)
├── models/                      # Architectural Routers
│   ├── agent_network.py         # Unified Agent Network (Switchboard)
│   ├── world_model.py           # Unified World Model (Physics Engine)
│   └── inference_output.py      # Type-hinted output structures
└── ...
```

## Core Components

### Convolutional Layers (`backbones/conv.py`)
- `ConvBackbone` - Multi-stage Conv2d + BatchNorm + Activation ✅
- `DeconvBackbone` - ConvTranspose2d for decoding/generation ✅
- Built natively with `nn.Sequential` for performance.

### Dense Layers (`backbones/dense.py`)
- `MLPBackbone` - Multi-layer perceptron with configurable depth/width ✅
- `NoisyLinear` - Linear layer with learned noise (Noisy Nets) ✅
- Direct construction with `nn.Sequential` eliminates redundant wrappers.

### Residual Blocks (`backbones/resnet.py`)
- `ResidualBlock` - Standard residual connection with configurable norm ✅
- `ResNetBackbone` - High-performance residual stack constructed via `nn.Sequential` ✅

### Network Heads (`heads/`)
- `PolicyHead` - Outputs action probabilities/logits
- `ValueHead` - Outputs state value
- `RewardHead` - Predicts environment rewards

## Logical Routers (`models/`)

This framework avoids hardcoded agent-specific architectures (e.g., `RainbowNet.py`). Instead, it uses a dynamic "Switchboard" pattern where backbones and heads are assembled based on configuration.

### AgentNetwork (`agent_network.py`)
The central orchestrator that routes data through Environment, Spatial, and Temporal phases to produce semantic outputs.

```python
from modules.models import AgentNetwork
from configs.agents.muzero import MuZeroConfig

# Assembled dynamically from config
config = MuZeroConfig(...)
network = AgentNetwork(config, input_shape=(3, 96, 96), num_actions=4)
```

### WorldModel (`world_model.py`)
A modular physics engine for model-based planning (MCTS).

```python
from modules.models import WorldModel

# Encapsulates Representation, Dynamics, and Stochastic components
world_model = WorldModel(config, observation_dimensions=(3, 96, 96), num_actions=4)
```

## Utility Functions

`utils.py` provides:
- Layer initialization schemes (orthogonal, xavier)
- Activation function helpers
- Shape calculation utilities
- Parameter count functions
- Mixed-precision/Device helpers
