# Utilities

General utility functions and helpers used across the project.

## Structure

```
utils/
├── __init__.py
├── utils.py              # Main utilities
└── lr_schedulers/        # Learning rate schedules
    ├── lr_schedule.py
    └── linear_annealed_lr.py
```

## Main Utilities (`utils.py`)

### Device Management
```python
from utils import get_device

device = get_device()  # Returns cuda if available, else cpu
```

### Seed Setting
```python
from utils import set_seed

set_seed(42)  # Reproducibility across numpy, torch, random
```

### Checkpointing
```python
from utils import save_checkpoint, load_checkpoint

save_checkpoint(model, optimizer, path='checkpoint.pt')
model, optimizer = load_checkpoint(path='checkpoint.pt', model=model, optimizer=optimizer)
```

### Logging
```python
from utils import setup_logger

logger = setup_logger('training', log_file='train.log')
logger.info('Training started')
```

## Learning Rate Schedulers

### Linear Annealed LR
```python
from utils.lr_schedulers.linear_annealed_lr import LinearAnnealedLR

scheduler = LinearAnnealedLR(
    optimizer,
    start_lr=0.001,
    end_lr=0.00001,
    total_steps=100000
)
```

### Usage with Agents

Most agents automatically handle LR scheduling based on config:

```python
config.learning_rate = 0.00025
config.lr_schedule = 'linear'
config.lr_decay_steps = 100000
```

## Helper Functions

- `compute_returns()` - Calculate discounted returns
- `normalize_observations()` - Observation normalization
- `action_mask()` - Valid action masking
- `entropy()` - Distribution entropy calculation

## Adding Utilities

When adding new utilities:
1. Add to `utils.py` for general functions
2. Create subdirectory for specialized utilities
3. Add tests in `tests/`
4. Import in `__init__.py` if widely used
