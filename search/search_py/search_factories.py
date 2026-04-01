from search.search_py.modular_search import ModularSearch


def create_mcts(config, device, num_actions) -> ModularSearch:
    """Create a Python-backend ModularSearch from config.

    All strategy objects (scoring, backprop, prior injectors, etc.) are
    instantiated via explicit parameters passed to ``ModularSearch.__init__``.
    """
    return ModularSearch(
        device=device,
        num_actions=num_actions,
        gumbel=getattr(config, "gumbel", False),
        bootstrap_method=getattr(config, "bootstrap_method", "parent_value"),
        policy_extraction=getattr(config, "policy_extraction", "visit"),
        backprop_method=getattr(config, "backprop_method", "average"),
        gumbel_m=getattr(config, "gumbel_m", 8),
        known_bounds=getattr(config, "known_bounds", None),
        min_max_epsilon=getattr(config, "min_max_epsilon", 1e-8),
        search_batch_size=getattr(config, "search_batch_size", 0),
        num_simulations=getattr(config, "num_simulations", 50),
        gumbel_cvisit=getattr(config, "gumbel_cvisit", 50),
        gumbel_cscale=getattr(config, "gumbel_cscale", 1.0),
        discount_factor=getattr(config, "discount_factor", 1.0),
        pb_c_init=getattr(config, "pb_c_init", 1.25),
        pb_c_base=getattr(config, "pb_c_base", 19652),
        stochastic=getattr(config, "stochastic", False),
        max_search_depth=getattr(config, "max_search_depth", None),
        use_virtual_mean=getattr(config, "use_virtual_mean", False),
        virtual_loss=getattr(config, "virtual_loss", 0.0),
        support_range=getattr(config, "support_range", None),
        use_dirichlet=getattr(config, "use_dirichlet", False),
        dirichlet_alpha=getattr(config, "dirichlet_alpha", 0.3),
        dirichlet_fraction=getattr(config, "dirichlet_fraction", 0.25),
        injection_frac=getattr(config, "injection_frac", 0.0),
        num_players=getattr(config, "num_players", 1),
    )
