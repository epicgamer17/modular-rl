import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Union, Optional, List

try:
    import umap
except ImportError:
    raise ImportError(
        "umap-learn is required. Please install it with `pip install umap-learn`."
    )


class LatentUMAPVisualizer:
    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "euclidean",
    ):
        self.reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
        )
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
        """Fits the UMAP model to the provided latents."""
        processed = self._process_latents(latents)
        self.reducer.fit(processed)

    def transform(self, latents: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Transforms the latents into the UMAP space."""
        # UMAP can transform new data relative to the fitted manifold
        processed = self._process_latents(latents)
        return self.reducer.transform(processed)

    def fit_transform(self, latents: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Fits and transforms the latents."""
        processed = self._process_latents(latents)
        return self.reducer.fit_transform(processed)

    def plot(
        self,
        latents: Union[torch.Tensor, np.ndarray],
        labels: Optional[Union[List, np.ndarray]] = None,
        save_path: Optional[str] = None,
        title: str = "Latent Space UMAP",
        show: bool = True,
    ):
        """
        Plots the 2D UMAP projection of the latents.

        Args:
            latents: The latent representations to plot.
            labels: Optional labels for each point (used for coloring).
            save_path: If provided, saves the plot to this path.
            title: Title of the plot.
            show: Whether to display the plot (plt.show()).
        """
        if self.n_components != 2:
            raise ValueError("Plotting is only supported for n_components=2")

        # For visualization of a batch, usually fit_transform is best
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
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)

        if show:
            plt.show()
        else:
            plt.close()
