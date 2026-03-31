import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import Union, Optional, List


class LatentPCAVisualizer:
    def __init__(self, n_components: int = 2):
        self.pca = PCA(n_components=n_components)
        self.n_components = n_components

    def _process_latents(self, latents: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Converts latents to numpy and flattens them if necessary.
        Expects input shape (N, ...). Output shape (N, flattened_dim).
        """
        if isinstance(latents, torch.Tensor):
            latents = latents.detach().cpu().numpy()

        if latents.ndim > 2:
            # Flatten all dimensions except the batch dimension
            return latents.reshape(latents.shape[0], -1)
        return latents

    def fit(self, latents: Union[torch.Tensor, np.ndarray]):
        """Fits the PCA model to the provided latents."""
        processed = self._process_latents(latents)
        self.pca.fit(processed)

    def transform(self, latents: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Transforms the latents into the PCA space."""
        processed = self._process_latents(latents)
        return self.pca.transform(processed)

    def fit_transform(self, latents: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Fits and transforms the latents."""
        processed = self._process_latents(latents)
        return self.pca.fit_transform(processed)

    def plot(
        self,
        latents: Union[torch.Tensor, np.ndarray],
        labels: Optional[Union[List, np.ndarray]] = None,
        save_path: Optional[str] = None,
        title: str = "Latent Space PCA",
        show: bool = True,
    ):
        """
        Plots the 2D PCA projection of the latents.

        Args:
            latents: The latent representations to plot.
            labels: Optional labels for each point (used for coloring).
            save_path: If provided, saves the plot to this path.
            title: Title of the plot.
            show: Whether to display the plot (plt.show()).
        """
        if self.n_components != 2:
            raise ValueError("Plotting is only supported for n_components=2")

        # fit_transform if not already fitted, or just transform?
        # For simplicity in this helper, we'll re-fit on the data provided
        # to ensure the best view of *this* batch.
        # If the user wants to use a pre-fitted PCA, they should call transform separately and plot manually,
        # or we could add a flag. Let's assume we fit on this data for visualization.
        points = self.fit_transform(latents)

        plt.figure(figsize=(10, 8))

        if labels is not None:
            # If labels are provided, use them for coloring
            unique_labels = np.unique(labels)
            for label in unique_labels:
                mask = np.array(labels) == label
                plt.scatter(
                    points[mask, 0], points[mask, 1], label=str(label), alpha=0.7
                )
            plt.legend()
        else:
            plt.scatter(points[:, 0], points[:, 1], alpha=0.7)

        plt.title(title)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)

        if show:
            plt.show()
        else:
            plt.close()
