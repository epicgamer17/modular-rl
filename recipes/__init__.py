"""Algorithm registries for assembling RL system components.

Each registry provides factory functions that wire together the network,
buffer, loss pipeline, target builder, and action selector for a given
algorithm family.  Callers supply hyperparameters and environment metadata;
the registry returns ready-to-use components.
"""

from recipes.muzero import build_muzero_components
from recipes.ppo import build_ppo_components
from recipes.rainbow import build_rainbow_components

__all__ = [
    "build_muzero_components",
    "build_ppo_components",
    "build_rainbow_components",
]
