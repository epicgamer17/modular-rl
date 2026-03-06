import pytest
pytestmark = pytest.mark.integration

import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from stats.stats import StatTracker


def _prepare_output_dir(tmp_path):
    output_dir = tmp_path / "test_output_stats_latent"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Prevent plots from showing up during tests
    plt.switch_backend("Agg")
    return output_dir


def test_pca_integration(tmp_path):
    output_dir = _prepare_output_dir(tmp_path)
    tracker = StatTracker(name="test_model")

    # Add latent data
    latents = torch.randn(50, 32)
    labels = torch.randint(0, 3, (50,))

    tracker.add_latent_visualization("latent_space", latents, labels, method="pca")

    # This will plot graphs and should create the file
    tracker.plot_graphs(dir=str(output_dir))

    expected_file = output_dir / "test_model_latent_space_pca.png"
    assert expected_file.exists()


def test_tsne_integration(tmp_path):
    output_dir = _prepare_output_dir(tmp_path)
    tracker = StatTracker(name="test_model_tsne")

    # Add latent data
    # Use small N for speed
    latents = torch.randn(20, 16)
    labels = torch.randint(0, 2, (20,))

    tracker.add_latent_visualization(
        "z_rep", latents, labels, method="tsne", perplexity=5.0, n_iter=250
    )

    tracker.plot_graphs(dir=str(output_dir))

    expected_file = output_dir / "test_model_tsne_z_rep_tsne.png"
    assert expected_file.exists()


def test_umap_integration(tmp_path):
    output_dir = _prepare_output_dir(tmp_path)
    tracker = StatTracker(name="test_model_umap")

    latents = torch.randn(20, 16)

    tracker.add_latent_visualization("z_umap", latents, method="umap", n_neighbors=5)

    tracker.plot_graphs(dir=str(output_dir))

    expected_file = output_dir / "test_model_umap_z_umap_umap.png"
    assert expected_file.exists()
