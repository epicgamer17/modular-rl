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
├── conv.py                      # Convolutional layers
├── dense.py                     # Fully connected layers
├── residual.py                  # Residual blocks
├── heads.py                     # Network heads (policy, value)
├── encoder_decoder.py           # Encoder-decoder architectures
├── action_encoder.py            # Action encoding utilities
├── network_block.py             # Generic network blocks
├── distributions.py             # Distribution utilities
├── actor.py                     # Actor network component
├── critic.py                    # Critic network component
├── base_stack.py                # Base network stack
├── sim_siam_projector_predictor.py  # Self-supervised learning
├── agent_nets/                  # Agent-specific networks
│   ├── rainbow_dqn.py
│   ├── muzero.py
│   ├── ppo.py
│   └── policy_imitation.py
└── world_models/                # World model implementations
    ├── world_model.py
    └── muzero_world_model.py
```

## Core Components

### Convolutional Layers (`conv.py`)
- `ConvBlock` - Conv2d + BatchNorm + Activation ✅
- `ConvStack` - Stack of convolutional blocks ✅
- Supports common RL architectures (Nature DQN, IMPALA)

### Dense Layers (`dense.py`)
- `MLP` - Multi-layer perceptron with configurable depth/width ✅
- `NoisyLinear` - Linear layer with learned noise (Noisy Nets) ✅
- `DuelingHead` - Separates value and advantage streams ✅

### Residual Blocks (`residual.py`)
- `ResidualBlock` - Standard residual connection ✅
- `ResNetStack` - Multiple residual blocks ✅
- Used in IMPALA-style architectures

### Network Heads (`heads.py`)
- `PolicyHead` - Outputs action probabilities/logits ✅
- `ValueHead` - Outputs state value ✅
- `CategoricalValueHead` - Outputs value distribution (C51) ✅

## Agent-Specific Networks

### Rainbow DQN Network
```python
from modules.agent_nets.rainbow_dqn import RainbowDQNNetwork

network = RainbowDQNNetwork(
    input_shape=(4, 84, 84),
    action_space=18,
    atom_size=51,
    dueling=True,
    noisy=True
)
```

### MuZero Network
```python
from modules.agent_nets.muzero import MuZeroNetwork

network = MuZeroNetwork(
    input_shape=(3, 96, 96),
    action_space=4,
    support_size=10,
    hidden_size=256
)
```

### PPO Network
```python
from modules.agent_nets.ppo import PPONetwork

network = PPONetwork(
    input_shape=(4,),
    action_space=2,
    continuous=True,
    hidden_dims=[64, 64]
)
```

## World Models

MuZero dynamics model for model-based planning:

```python
from modules.world_models.muzero_world_model import MuZeroWorldModel

model = MuZeroWorldModel(
    action_space=9,
    hidden_size=256,
    support_size=10
)

# Dynamics: (hidden_state, action) -> (new_state, reward)
next_state, reward_logits = model.dynamics(state, action)

# Prediction: (state) -> (policy_logits, value_logits)
policy, value = model.prediction(state)

# Representation: (observation) -> (state)
state = model.representation(obs)
```

## Building Custom Networks

Compose modules to create custom architectures:

```python
from modules.conv import ConvStack
from modules.dense import MLP
from modules.heads import PolicyHead, ValueHead

class CustomNetwork(nn.Module):
    def __init__(self, input_shape, action_space):
        super().__init__()
        self.encoder = ConvStack(input_shape, channels=[32, 64, 64])
        self.shared = MLP(input_dim=3136, hidden_dims=[512])
        self.policy_head = PolicyHead(512, action_space)
        self.value_head = ValueHead(512)
```

## Utility Functions

`utils.py` provides:
- Layer initialization schemes (orthogonal, xavier)
- Activation function helpers
- Shape calculation utilities
- Parameter count functions
