# ELO Rating System

ELO rating implementation for evaluating agent strength in competitive games.

## Installation

```bash
pip install -e .
```

## Usage

```python
from elo.elo import EloRating

# Create rating system
elo = EloRating(k_factor=32, initial_rating=1500)

# Record match outcome
new_winner, new_loser = elo.update_ratings(winner_rating, loser_rating)

# Or with draw
new_r1, new_r2 = elo.update_ratings(r1, r2, result=0.5)
```

## Parameters

- `k_factor` - Maximum rating change per game (default: 32)
- `initial_rating` - Starting rating for new players (default: 1500)

## Expected Score

Calculate win probability given ratings:

```python
expected = elo.expected_score(1600, 1500)  # ~0.64
```

## Tournament Evaluation

Run round-robin tournaments between agents:

```python
results = elo.run_tournament(agents, env, n_games=100)
ratings = elo.compute_ratings(results)
```

## Notebook

See `elo_test.ipynb` for interactive testing and visualization.
