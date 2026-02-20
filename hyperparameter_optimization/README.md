# Hyperparameter Optimization

Tools for automated hyperparameter tuning of RL agents.

## Installation

```bash
pip install -e .
```

## Supported Algorithms

- Random search ✅
- Bayesian optimization (Tree-structured Parzen Estimator) ✅
- Population-based training ✅

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
