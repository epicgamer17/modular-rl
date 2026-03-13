# Loss Functions

Modular loss function implementations for RL training objectives. Supports value losses, policy losses, and auxiliary losses.

## Installation

Losses are included in the main package:

```bash
pip install -e .
```

## Structure

```
losses/
├── __init__.py           # Loss registry and exports
├── losses.py             # Main loss implementations
├── losses.py       # Basic loss definitions
└── loss_manager.py       # Loss management utilities
```

## Available Losses

### Value Losses
- **MSELoss** - Mean squared error for value estimation ✅
- **SmoothL1Loss** - Huber loss (less sensitive to outliers) ✅
- **CategoricalCrossEntropy** - For distributional value learning (C51) ✅

### Policy Losses
- **PolicyGradientLoss** - Vanilla policy gradient ✅
- **PPOLoss** - Proximal policy optimization with clipping ✅
- **CrossEntropyLoss** - For supervised policy learning ✅

### MuZero Losses
- **MuZeroLoss** - Combined value, policy, reward, and consistency loss ✅
- **ConsistencyLoss** - Observation reconstruction consistency ✅

### Auxiliary Losses
- **EntropyBonus** - Entropy regularization for exploration ✅
- **L2Regularization** - Weight decay ✅

## Usage

### Basic Loss
```python
from losses import MSELoss

loss_fn = MSELoss()
loss = loss_fn(predictions, targets)
```

### Rainbow DQN Loss
```python
from losses import CategoricalLoss

loss_fn = CategoricalLoss(v_min=-10, v_max=10, num_atoms=51)
loss = loss_fn(current_dist, target_dist)
```

### PPO Loss
```python
from losses import PPOLoss

loss_fn = PPOLoss(epsilon=0.2, value_coef=0.5, entropy_coef=0.01)
total_loss = loss_fn(
    new_log_probs, old_log_probs, advantages,
    new_values, returns, entropy
)
```

### MuZero Loss
```python
from losses import MuZeroLoss

loss_fn = MuZeroLoss(
    value_coef=0.25,
    reward_coef=1.0,
    policy_coef=1.0,
    consistency_coef=0.5
)
loss = loss_fn(predictions, targets, priorities)
```

## Loss Manager

Combine multiple losses with weights:

```python
from losses.loss_manager import LossManager

manager = LossManager()
manager.add_loss('value', MSELoss(), weight=1.0)
manager.add_loss('policy', CrossEntropyLoss(), weight=1.0)
manager.add_loss('entropy', EntropyBonus(), weight=0.01)

total_loss = manager.compute(predictions, targets)
```

## Custom Losses

Create custom losses by inheriting from `BaseLoss`:

```python
from losses.losses import BaseLoss

class CustomLoss(BaseLoss):
    def __init__(self, param=1.0):
        self.param = param
    
    def forward(self, predictions, targets):
        return torch.mean((predictions - targets) ** 2 * self.param)
```
