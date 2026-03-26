from search.search_py.modular_search import ModularSearch


def create_mcts(config, device, num_actions) -> ModularSearch:
    """Create a Python-backend ModularSearch from config.

    All strategy objects (scoring, backprop, prior injectors, etc.) are
    derived automatically from ``config`` inside ``ModularSearch.__init__``.
    """
    return ModularSearch(config, device, num_actions)
