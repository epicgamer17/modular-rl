from __future__ import annotations
import importlib
import os
from typing import Any
import torch


class SearchBackendFactory:
    """Unified factory for creating search engines regardless of the backend."""

    @staticmethod
    def create(
        config: Any, device: torch.device = None, num_actions: int = None
    ) -> Any:
        """
        Creates a search engine based on the provided configuration.

        Args:
            config: Full agent configuration with a ``search_backend`` attribute.
            device: Torch device for the search engine (defaults to CPU if not in config).
            num_actions: Number of actions for the environment.

        Returns:
            An instance of ModularSearch from the selected backend.
        """
        # 1. Extract common parameters if not provided
        if device is None:
            device = getattr(config, "device", torch.device("cpu"))

        if num_actions is None:
            game_cfg = getattr(config, "game", None)
            if game_cfg:
                num_actions = getattr(game_cfg, "num_actions", None)

        # 2. Use old_muzero's own search module (not the new search/ package)
        backend = getattr(config, "search_backend", "python")
        if backend == "python":
            from search.search_py.modular_search import ModularSearch
        elif backend == "aos":
            from search.aos_search.search_algorithm import ModularSearch
        else:
            raise ValueError(f"Unsupported search backend for old_muzero: {backend}")

        return ModularSearch(config, device, num_actions)
