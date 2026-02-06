# Experiments

Training runs, checkpoints, and results for different algorithm experiments.

## Directory Structure

```
experiments/
├── rainbow/              # Rainbow DQN experiments
│   ├── rainbow_and_ape-x/
│   └── rainbow_hyperopt/ # Hyperparameter optimization
├── rainbow-nfsp/         # Rainbow + NFSP experiments
│   ├── checkpoints/
│   └── nfsp_testing.ipynb
└── rainbowzero/          # Rainbow + MuZero experiments
    ├── paper_outline.tex # Research paper draft
    ├── cartpole/         # CartPole results
    ├── catan/            # Catan game results
    ├── slippery_grid/    # Grid world experiments
    └── tictactoe/        # Tic-Tac-Toe experiments
        └── checkpoints/  # Training checkpoints
```

## Experiment Organization

Each experiment directory typically contains:

- `config.yaml` - Experiment configuration
- `checkpoints/` - Model checkpoints
- `graphs/` - Training curves and metrics
- `videos/` - Agent gameplay recordings
- `logs/` - Training logs

## Running Experiments

### Rainbow DQN
```bash
cd experiments/rainbow/rainbow_hyperopt
python train.py --config config.yaml
```

### MuZero (RainbowZero)
```bash
cd experiments/rainbowzero/tictactoe
python train.py
```

### NFSP
```bash
cd experiments/rainbow-nfsp
python train_nfsp.py
```

## Checkpoints

Checkpoints are saved in subdirectories by experiment:
- Standard format: `checkpoints/{experiment_name}/{timestamp}/`
- Contains: model weights, optimizer state, config, training stats

## Results Analysis

Use the Jupyter notebooks in each experiment directory to:
- Plot training curves
- Compare different runs
- Evaluate agent performance

## Paper Drafts

`rainbowzero/paper_outline.tex` contains LaTeX source for research paper drafts.
