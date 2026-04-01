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
        # 3. Create Backend
        backend = getattr(config, "search_backend", "python")
        
        # Extract common parameters for all backends
        gumbel = getattr(config, "gumbel", False)
        bootstrap_method = getattr(config, "bootstrap_method", "parent_value")
        policy_extraction = getattr(config, "policy_extraction", "visit")
        backprop_method = getattr(config, "backprop_method", "average")
        num_simulations = getattr(config, "num_simulations", 50)
        discount_factor = getattr(config, "discount_factor", 1.0)
        pb_c_init = getattr(config, "pb_c_init", 1.25)
        pb_c_base = getattr(config, "pb_c_base", 19652)
        stochastic = getattr(config, "stochastic", False)
        known_bounds = getattr(config, "known_bounds", None)
        min_max_epsilon = getattr(config, "min_max_epsilon", 1e-8)
        search_batch_size = getattr(config, "search_batch_size", 0)
        use_virtual_mean = getattr(config, "use_virtual_mean", False)
        virtual_loss = getattr(config, "virtual_loss", 0.0)
        num_players = getattr(config, "num_players", 1)
        if game_cfg := getattr(config, "game", None):
            num_players = getattr(game_cfg, "num_players", num_players)

        if backend == "python":
            from search.search_py.modular_search import ModularSearch

            return ModularSearch(
                device=device,
                num_actions=num_actions,
                gumbel=gumbel,
                bootstrap_method=bootstrap_method,
                policy_extraction=policy_extraction,
                backprop_method=backprop_method,
                num_simulations=num_simulations,
                discount_factor=discount_factor,
                pb_c_init=pb_c_init,
                pb_c_base=pb_c_base,
                stochastic=stochastic,
                known_bounds=known_bounds,
                min_max_epsilon=min_max_epsilon,
                search_batch_size=search_batch_size,
                use_virtual_mean=use_virtual_mean,
                virtual_loss=virtual_loss,
                num_players=num_players,
                # Python specific
                gumbel_m=getattr(config, "gumbel_m", 8),
                gumbel_cvisit=getattr(config, "gumbel_cvisit", 50),
                gumbel_cscale=getattr(config, "gumbel_cscale", 1.0),
                max_search_depth=getattr(config, "max_search_depth", None),
                support_range=getattr(config, "support_range", None),
                use_dirichlet=getattr(config, "use_dirichlet", False),
                dirichlet_alpha=getattr(config, "dirichlet_alpha", 0.3),
                dirichlet_fraction=getattr(config, "dirichlet_fraction", 0.25),
                injection_frac=getattr(config, "injection_frac", 0.0),
            )
        elif backend == "aos":
            from search.aos_search.search_algorithm import ModularSearch
            
            compile_cfg = getattr(config, "compilation", None)
            compile_enabled = getattr(compile_cfg, "enabled", False) if compile_cfg else False
            compile_fullgraph = getattr(compile_cfg, "fullgraph", False) if compile_cfg else False

            return ModularSearch(
                device=device,
                num_actions=num_actions,
                num_simulations=num_simulations,
                discount_factor=discount_factor,
                pb_c_init=pb_c_init,
                pb_c_base=pb_c_base,
                gumbel=gumbel,
                bootstrap_method=bootstrap_method,
                backprop_method=backprop_method,
                stochastic=stochastic,
                known_bounds=known_bounds,
                min_max_epsilon=min_max_epsilon,
                search_batch_size=search_batch_size,
                use_virtual_mean=use_virtual_mean,
                virtual_loss=virtual_loss,
                num_players=num_players,
                # AOS specific
                max_search_depth=getattr(config, "max_search_depth", 5),
                max_nodes=getattr(config, "max_nodes", 512),
                num_codes=getattr(config, "num_codes", 0),
                use_dirichlet=getattr(config, "use_dirichlet", False),
                dirichlet_alpha=getattr(config, "dirichlet_alpha", 0.3),
                dirichlet_fraction=getattr(config, "dirichlet_fraction", 0.25),
                gumbel_cvisit=getattr(config, "gumbel_cvisit", 50.0),
                gumbel_cscale=getattr(config, "gumbel_cscale", 1.0),
                use_sequential_halving=getattr(config, "use_sequential_halving", False),
                gumbel_m=getattr(config, "gumbel_m", 8),
                internal_decision_modifier=getattr(config, "internal_decision_modifier", "none"),
                internal_chance_modifier=getattr(config, "internal_chance_modifier", "none"),
                compile_enabled=compile_enabled,
                compile_fullgraph=compile_fullgraph,
            )
        elif backend == "cpp":
            from search import set_backend, ModularSearch as CppModularSearch

            set_backend("cpp")
            return CppModularSearch(
                device=device,
                num_actions=num_actions,
                num_simulations=num_simulations,
                discount_factor=discount_factor,
                pb_c_init=pb_c_init,
                pb_c_base=pb_c_base,
                gumbel=gumbel,
                bootstrap_method=bootstrap_method,
                backprop_method=backprop_method,
                stochastic=stochastic,
                known_bounds=known_bounds,
                min_max_epsilon=min_max_epsilon,
                num_players=num_players,
            )
        else:
            raise ValueError(f"Unsupported search backend: {backend}")
