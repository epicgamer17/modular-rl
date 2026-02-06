# Hyperparameter Optimization

Tools for automated hyperparameter tuning of RL agents.

## Usage

```python
from hyperparameter_optimization.hyperopt import HyperparameterOptimizer

# Define search space
search_space = {
    'learning_rate': (1e-5, 1e-2, 'log-uniform'),
    'batch_size': [32, 64, 128, 256],
    'gamma': (0.95, 0.999, 'uniform'),
    'buffer_size': [50000, 100000, 500000]
}

# Create optimizer
optimizer = HyperparameterOptimizer(
    agent_class=RainbowDQN,
    game_config=CartPoleConfig(),
    search_space=search_space,
    max_evals=100
)

# Run optimization
best_config = optimizer.optimize()
```

## Search Algorithms

Supported optimization methods:
- Random search
- Bayesian optimization (Tree-structured Parzen Estimator)
- Population-based training

## Parallel Evaluation

Speed up tuning with parallel trials:

```python
optimizer = HyperparameterOptimizer(
    agent_class=RainbowDQN,
    game_config=game_config,
    search_space=search_space,
    max_evals=100,
    n_workers=4  # Parallel workers
)
```

## Result Storage

Results saved to `hyperopt_results/`:
- `trials.json` - All trial configurations and results
- `best_config.json` - Best found configuration
- `study.pkl` - Full optimization state

## Integration with Experiments

Use optimized configs in experiment runs:

```python
from hyperparameter_optimization.hyperopt import load_best_config

config = load_best_config('experiments/rainbow/hyperopt_results/')
agent = RainbowDQN(config, game_config)
```
