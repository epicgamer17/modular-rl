import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional

# Muted Scientific Palette (based on common paper styles like Nature/Science or standard high-quality libraries)
# Avoids "neon" look, focuses on readability and aesthetics.
SCIENTIFIC_COLORS = {
    "primary": "#4A69BD",      # Muted Blue
    "secondary": "#E58E26",    # Muted Orange
    "accent": "#78E08F",       # Muted Green
    "critical": "#EB2F06",     # Muted Red
    "neutral": "#60A3BC",      # Muted Cyan
    "dark_bg": "#1E272E",      # Deep Slate for Dark Mode
    "light_bg": "#FFFFFF",     # Pure White for Light Mode
    "text_dark": "#D2DAE2",    # Light Grey text for Dark Mode
    "text_light": "#2F3640",   # Dark Slate text for Light Mode
    "grid": "#485460",         # Grid color
}

def apply_scientific_style(mode: str = "dark"):
    """Apply premium scientific plotting styles."""
    is_dark = mode == "dark"
    bg_color = SCIENTIFIC_COLORS["dark_bg"] if is_dark else SCIENTIFIC_COLORS["light_bg"]
    text_color = SCIENTIFIC_COLORS["text_dark"] if is_dark else SCIENTIFIC_COLORS["text_light"]
    grid_color = SCIENTIFIC_COLORS["grid"] if is_dark else "#DCDDE1"

    # Matplotlib Global RC Params
    plt.rcParams.update({
        "figure.facecolor": bg_color,
        "axes.facecolor": bg_color,
        "axes.edgecolor": text_color,
        "axes.labelcolor": text_color,
        "axes.grid": True,
        "grid.color": grid_color,
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
        "xtick.color": text_color,
        "ytick.color": text_color,
        "text.color": text_color,
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Roboto", "Arial", "DejaVu Sans"],
        "legend.frameon": True,
        "legend.facecolor": bg_color,
        "legend.edgecolor": text_color,
        "savefig.facecolor": bg_color,
        "savefig.edgecolor": bg_color,
    })

    # Seaborn Theme
    sns.set_theme(
        style="whitegrid" if not is_dark else "darkgrid",
        palette=sns.color_palette([
            SCIENTIFIC_COLORS["primary"],
            SCIENTIFIC_COLORS["secondary"],
            SCIENTIFIC_COLORS["accent"],
            SCIENTIFIC_COLORS["critical"],
            SCIENTIFIC_COLORS["neutral"],
        ]),
    )

def get_color(key: str) -> str:
    return SCIENTIFIC_COLORS.get(key, "#000000")

def get_alpha(uncertainty: bool = False) -> float:
    """Standardized transparency rules."""
    return 0.2 if uncertainty else 0.8
