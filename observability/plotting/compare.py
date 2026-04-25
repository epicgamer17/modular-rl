import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional, Any
from .style import apply_scientific_style, get_color, get_alpha

def compare_metrics(
    run_data: Dict[str, List[float]], # run_name -> values
    name: str = "Metric",
    title: Optional[str] = None,
    mode: str = "dark",
    save_path: Optional[str] = None
):
    """Overlay multiple runs for comparison."""
    apply_scientific_style(mode=mode)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = [
        get_color("primary"), 
        get_color("secondary"), 
        get_color("accent"), 
        get_color("critical"), 
        get_color("neutral")
    ]
    
    for i, (run_name, values) in enumerate(run_data.items()):
        color = colors[i % len(colors)]
        x = np.arange(len(values))
        
        # Smoothed line
        window = 50
        if len(values) > window:
            smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
            x_smoothed = np.arange(window-1, len(values))
            ax.plot(x_smoothed, smoothed, label=run_name, color=color, linewidth=2.5)
            # Faint raw line
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
