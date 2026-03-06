import pytest

pytestmark = pytest.mark.integration

import numpy as np
import torch

from stats.latent_pca import LatentPCAVisualizer
from stats.latent_tsne import LatentTSNEVisualizer

try:
    from stats.latent_umap import LatentUMAPVisualizer

    HAS_UMAP = True
    UMAP_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - dependency-gated path
    LatentUMAPVisualizer = None
    HAS_UMAP = False
    UMAP_IMPORT_ERROR = str(exc)



def _visualizer_configs(include_umap=True):
    cfg = {
        "pca": (LatentPCAVisualizer, {"n_components": 2}, 50),
        "tsne": (
            LatentTSNEVisualizer,
            {"n_components": 2, "perplexity": 5.0, "n_iter": 250},
            50,
        ),
    }
    if include_umap and HAS_UMAP:
        cfg["umap"] = (LatentUMAPVisualizer, {"n_components": 2, "n_neighbors": 10}, 50)
    return cfg



def test_latent_visualizers_fit_transform_vectors():
    for name, (visualizer_cls, kwargs, n_samples) in _visualizer_configs().items():
        latents = torch.randn(n_samples, 32)
        visualizer = visualizer_cls(**kwargs)

        points = visualizer.fit_transform(latents)

        assert isinstance(points, np.ndarray), f"{name} should return a NumPy array"
        assert points.shape == (n_samples, 2)



def test_latent_visualizers_fit_transform_images():
    for name, (visualizer_cls, kwargs, n_samples) in _visualizer_configs().items():
        latents = torch.randn(n_samples, 3, 16, 16)
        visualizer = visualizer_cls(**kwargs)

        points = visualizer.fit_transform(latents)

        assert points.shape == (n_samples, 2), f"{name} image transform shape mismatch"



def test_latent_visualizers_plotting(tmp_path):
    labels = np.random.randint(0, 3, size=20)

    for name, (visualizer_cls, kwargs, _) in _visualizer_configs().items():
        latents = torch.randn(20, 16)
        visualizer = visualizer_cls(**kwargs)
        save_path = tmp_path / f"latent_{name}.png"

        visualizer.plot(latents, labels=labels, save_path=str(save_path), show=False)

        assert save_path.exists(), f"{name} did not write plot output"



@pytest.mark.skipif(not HAS_UMAP, reason="UMAP dependency unavailable")
def test_umap_transform_after_fit():
    visualizer = LatentUMAPVisualizer(n_components=2, n_neighbors=10)
    latents = torch.randn(40, 16)
    visualizer.fit_transform(latents)

    new_latents = torch.randn(5, 16)
    new_points = visualizer.transform(new_latents)

    assert new_points.shape == (5, 2)



def test_umap_dependency_status_message_is_string():
    if HAS_UMAP:
        assert UMAP_IMPORT_ERROR is None
    else:
        assert isinstance(UMAP_IMPORT_ERROR, str)
