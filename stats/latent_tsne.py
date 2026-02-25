import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from typing import Union, Optional, List


class LatentTSNEVisualizer:
    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        learning_rate: Union[float, str] = "auto",
        n_iter: int = 1000,
    ):
        self.tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            n_iter=n_iter,
            init="pca",  # Better initialization often helps stability
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

    def fit_transform(self, latents: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Fits and transforms the latents using t-SNE."""
        processed = self._process_latents(latents)
        return self.tsne.fit_transform(processed)

    # Standard sklearn t-SNE doesn't support 'transform' for unseen data
    # (it is transductive), so we omit 'transform' and 'fit' methods to avoid confusion.

    def plot(
        self,
        latents: Union[torch.Tensor, np.ndarray],
        labels: Optional[Union[List, np.ndarray]] = None,
        save_path: Optional[str] = None,
        title: str = "Latent Space t-SNE",
        show: bool = True,
    ):
        """
        Plots the 2D t-SNE projection of the latents.

        Args:
            latents: The latent representations to plot.
            labels: Optional labels for each point (used for coloring).
            save_path: If provided, saves the plot to this path.
            title: Title of the plot.
            show: Whether to display the plot (plt.show()).
        """
        if self.n_components != 2:
            raise ValueError("Plotting is only supported for n_components=2")

        # t-SNE needs to be re-run on the specific batch we want to visualize
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
        # t-SNE axes don't have intrinsic meaning like PCA components
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)

        if show:
            plt.show()
        else:
            plt.close()
