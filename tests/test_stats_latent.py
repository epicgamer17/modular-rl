import unittest
import torch
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from stats.stats import StatTracker

class TestStatsLatent(unittest.TestCase):
    def setUp(self):
        self.output_dir = "test_output_stats_latent"
        os.makedirs(self.output_dir, exist_ok=True)
        # Prevent plots from showing up during tests
        plt.switch_backend('Agg')

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_pca_integration(self):
        tracker = StatTracker(name="test_model")
        
        # Add latent data
        latents = torch.randn(50, 32)
        labels = torch.randint(0, 3, (50,))
        
        tracker.add_latent_visualization("latent_space", latents, labels, method='pca')
        
        # This will plot graphs and should create the file
        tracker.plot_graphs(dir=self.output_dir)
        
        expected_file = os.path.join(self.output_dir, "test_model_latent_space_pca.png")
        self.assertTrue(os.path.exists(expected_file))

    def test_tsne_integration(self):
        tracker = StatTracker(name="test_model_tsne")
        
        # Add latent data
        # Use small N for speed
        latents = torch.randn(20, 16)
        labels = torch.randint(0, 2, (20,))
        
        tracker.add_latent_visualization("z_rep", latents, labels, method='tsne', perplexity=5.0, n_iter=250)
        
        tracker.plot_graphs(dir=self.output_dir)
        
        expected_file = os.path.join(self.output_dir, "test_model_tsne_z_rep_tsne.png")
        self.assertTrue(os.path.exists(expected_file))

    def test_umap_integration(self):
        tracker = StatTracker(name="test_model_umap")
        
        latents = torch.randn(20, 16)
        
        tracker.add_latent_visualization("z_umap", latents, method='umap', n_neighbors=5)
        
        tracker.plot_graphs(dir=self.output_dir)
        
        expected_file = os.path.join(self.output_dir, "test_model_umap_z_umap_umap.png")
        self.assertTrue(os.path.exists(expected_file))

if __name__ == '__main__':
    unittest.main()
