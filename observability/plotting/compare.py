import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional
from .style import plot_theme


def compare_metrics(
    run_data: Dict[str, List[float]],
    name: str = "Metric",
    title: Optional[str] = None,
    theme: str = "matplotx:pitaya_smoothie",
    save_path: Optional[str] = None,
):
    """Overlay multiple runs for comparison. Each run gets the next slot in the active theme's color cycle."""
    with plot_theme(theme):
        fig, ax = plt.subplots(figsize=(12, 7))

        for i, (run_name, values) in enumerate(run_data.items()):
            color = f"C{i}"
            x = np.arange(len(values))

            window = 50
            if len(values) > window:
                smoothed = np.convolve(values, np.ones(window) / window, mode='valid')
                x_smoothed = np.arange(window - 1, len(values))
                ax.plot(x_smoothed, smoothed, label=run_name, color=color, linewidth=2.5)
                ax.plot(x, values, color=color, alpha=0.1, linewidth=0.5)
            else:
                ax.plot(x, values, label=run_name, color=color, linewidth=2)

        ax.set_title(title or f"Comparison: {name}", fontsize=20, fontweight="bold", pad=25)
        ax.set_xlabel("Steps", fontsize=14)
        ax.set_ylabel("Value", fontsize=14)
        ax.legend(fontsize=12, frameon=True, shadow=True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()
