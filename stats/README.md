# Statistics and Visualization

Tools for tracking training metrics and visualizing learned representations.

## Installation

Statistics tools are included in the main package:

```bash
pip install -e .
```

## Components

### Statistics Tracking (`stats.py`)

Track and log training metrics ✅

```python
from stats.stats import Statistics

stats = Statistics(
    log_dir='logs/experiment_1',
    metrics=['reward', 'loss', 'value_est', 'q_values']
)

# Log metrics
stats.log('reward', episode_reward)
stats.log('loss', loss_value)

# Save to disk
stats.save()
```

### Latent Space Visualization

Visualize learned representations using dimensionality reduction:

#### PCA (`latent_pca.py`)
```python
from stats.latent_pca import LatentPCA

pca = LatentPCA(n_components=2)
reduced = pca.fit_transform(latent_states)
pca.plot(reduced, labels=actions)
```

#### t-SNE (`latent_tsne.py`)
```python
from stats.latent_tsne import LatentTSNE

tsne = LatentTSNE(perplexity=30, n_iter=1000)
reduced = tsne.fit_transform(latent_states)
tsne.plot(reduced, color_by='episode')
```

#### UMAP (`latent_umap.py`)
```python
from stats.latent_umap import LatentUMAP

umap = LatentUMAP(n_neighbors=15, min_dist=0.1)
reduced = umap.fit_transform(latent_states)
umap.plot(reduced, labels=episode_rewards)
```

## Usage in Agents

Most agents automatically track statistics:

```python
# Statistics automatically updated during training
agent.train(episodes=1000)

# Access statistics
print(agent.stats.get_mean('reward'))
agent.stats.plot_training_curves()
```

## Visualization Examples

- Training curves: Reward, loss, value estimates over time
- Latent space: How representations cluster by state/action
- Action distributions: Policy entropy and exploration

## Testing

```bash
pytest tests/stats/test_stats_tracker_append.py
pytest tests/stats/test_stats_latent_visualizations.py
```
