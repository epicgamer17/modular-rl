import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Dict, Optional

def plot_regression_results(
    name: str,
    train_scores: List[float],
    test_scores: Optional[List[float]] = None,
    losses: Optional[Dict[str, List[float]]] = None,
    save_dir: str = "artifacts/plots"
) -> str:
    """
    Plots training results and saves the figure.
    
    Args:
        name: Name of the test/experiment.
        train_scores: List of training scores per episode.
        test_scores: Optional list of test scores.
        losses: Optional dictionary mapping loss names to lists of loss values.
        save_dir: Directory to save the plots.
        
    Returns:
        Path to the saved plot.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    # Calculate number of subplots
    plot_items = []
    if train_scores and len(train_scores) > 0:
        plot_items.append(("Training Scores", train_scores))
    if test_scores and len(test_scores) > 0:
        plot_items.append(("Test Scores", test_scores))
    if losses and len(losses) > 0:
        # Check if any loss list actually has data
        has_loss = False
        for l in losses.values():
            if len(l) > 0:
                has_loss = True
                break
        if has_loss:
            plot_items.append(("Losses", losses))
        
    num_subplots = len(plot_items)
    if num_subplots == 0:
        print("No data to plot.")
        return ""
        
    fig, axes = plt.subplots(num_subplots, 1, figsize=(10, 5 * num_subplots))
    if num_subplots == 1:
        axes = [axes]
    
    for i, (title, data) in enumerate(plot_items):
        ax = axes[i]
        if title == "Training Scores":
            ax.plot(data, label="Score", alpha=0.3, color='blue')
            if len(data) >= 10:
                # Plot moving average
                window = min(100, len(data) // 10 + 1)
                ma = np.convolve(data, np.ones(window)/window, mode='valid')
                ax.plot(np.arange(window-1, len(data)), ma, label=f"MA({window})", color='blue', linewidth=2)
            ax.set_ylabel("Score")
        elif title == "Test Scores":
            ax.plot(data, marker='o', linestyle='-', label="Test Score", color='green')
            ax.set_ylabel("Score")
        elif title == "Losses":
            for loss_name, loss_values in data.items():
                if len(loss_values) > 0:
                    ax.plot(loss_values, label=loss_name)
            ax.set_ylabel("Loss Value")
            ax.set_yscale('log')
            
        ax.set_title(f"{name} - {title}")
        ax.set_xlabel("Steps/Episodes" if title != "Test Scores" else "Evaluation Index")
        ax.legend()
        ax.grid(True, which="both", ls="-", alpha=0.5)
        
    plt.tight_layout()
    filename = f"{name.lower().replace(' ', '_')}_regression.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to {save_path}")
    return save_path
