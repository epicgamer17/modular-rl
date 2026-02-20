# Search Algorithms

Monte Carlo Tree Search (MCTS) and related planning algorithms. Provides modular components that can be combined for different search strategies.

## Installation

Search algorithms are included in the main package:

```bash
pip install -e .
```

## Structure

```
search/
├── modular_search.py       # Main MCTS implementation
├── nodes.py                # Search tree node types
├── action_selectors.py     # Action selection strategies
├── scoring_methods.py      # Node scoring functions
├── backpropagation.py         # Value backpropagation
├── root_policies.py        # Root action policies
├── initial_searchsets.py  # Initial action selection
├── prior_injectors.py      # Prior injection mechanisms
├── pruners.py              # Tree pruning strategies
├── min_max_stats.py        # Min-max normalization
├── search_factories.py     # Search algorithm factories
└── utils.py                # Search utilities
```

## Core MCTS Components

### Search Algorithm

`ModularMCTS` provides a composable MCTS implementation:

```python
from search.modular_search import ModularMCTS
from search.action_selectors import UCBSelector
from search.scoring_methods import QValueScorer

mcts = ModularMCTS(
    action_selector=UCBSelector(c_puct=1.25),
    num_simulations=800,
    root_noise=True,
    add_exploration_noise=True
)

# Run search
root = mcts.run(model, observation)
action = root.best_action()
```

### Node Types

- `Node` - Standard MCTS node with visit count, value sum
- `MuZeroNode` - Extends Node with latent state and reward
- `StochasticNode` - Handles chance nodes for stochastic games

### Action Selectors

- **UCBSelector** - Upper Confidence Bound (AlphaZero/MuZero)
- **PUCTSelector** - Predictor + UCB (traditional)
- **GumbelSelector** - Gumbel sampling for action selection

### Scoring Methods

- **QValueScorer** - Mean action value
- **VisitCountScorer** - Visit count for temperature-based selection
- **MuZeroScorer** - MuZero-specific scoring with min-max scaling

## Usage Examples

### AlphaZero Search
```python
from search import AlphaZeroSearch

search = AlphaZeroSearch(
    num_simulations=800,
    c_puct=1.25,
    dirichlet_alpha=0.3,
    dirichlet_epsilon=0.25
)

policy, value = search.run(model, board_state)
```

### MuZero Search
```python
from search import MuZeroSearch

search = MuZeroSearch(
    num_simulations=50,
    max_depth=10,
    discount=0.997
)

root = search.run(representation_model, dynamics_model, prediction_model, obs)
action = select_action(root.visit_counts, temperature=1.0)
```

### Gumbel MuZero
```python
from search.action_selectors import GumbelSelector

selector = GumbelSelector(
    max_num_considered_actions=16,
    gumbel_scale=1.0
)

mcts = ModularMCTS(action_selector=selector, num_simulations=64)
```

## Backpropagation

Update node statistics after simulation:

```python
from search.backpropagation import StandardBackprop

backprop = StandardBackprop(discount=0.997)
backprop.update(search_path, value)
```

## Min-Max Statistics

Normalize Q-values for stable learning:

```python
from search.min_max_stats import MinMaxStats

stats = MinMaxStats()
stats.update(q_value)
normalized_q = stats.normalize(q_value)
```

## Pruning

Remove unlikely actions from search:

```python
from search.pruners import LowVisitPruner

pruner = LowVisitPruner(min_visits=2)
pruner.prune(root)
```

## Factory Methods

Create pre-configured search algorithms:

```python
from search.search_factories import create_muzero_search

search = create_muzero_search(
    config={'num_simulations': 50, 'use_gumbel': True}
)
```

## Testing

Run search algorithm tests:

```python
pytest tests/test_muzero_mcts_core.py
pytest tests/test_batched_search.py
```

> **Note:** Older implementations are available in `search/deprecated/`. Use the main `search/` module for new development.
