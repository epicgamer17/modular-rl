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

        # 2. Dispatch to the correct backend module directly
        backend: str = getattr(config, "search_backend", "python").lower()

        if backend == "aos":
            from search.aos_search.search_algorithm import ModularSearch
        elif backend == "cpp":
            _cpp = importlib.import_module(
                os.getenv("MCTS_CPP_MODULE", "mcts_cpp_backend")
            )
            ModularSearch = _cpp.ModularSearch
        else:
            from search.search_py.modular_search import ModularSearch

        return ModularSearch(config, device, num_actions)
