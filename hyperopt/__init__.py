import pickle
from .config import (
    MarlHyperoptConfig,
    SarlHyperoptConfig,
    set_sarl_config,
    set_marl_config,
    get_active_config
)
from .evaluation import (
    test_score_evaluation,
    elo_evaluation,
    best_agent_elo_evaluation,
    test_agents_elo_evaluation
)
from .training import marl_run_training, sarl_run_training
from .objectives import marl_objective, sarl_objective
from .analysis import (
    analyze_trial_stats,
    analyze_hyperparameter_importance,
    predict_best_config,
    plot_general_trends,
    simulate_elo_math
)

def save_search_space(search_space, initial_best_config=None):
    if initial_best_config is None: initial_best_config = [{}]
    with open("search_space.pkl", "wb") as f: pickle.dump(search_space, f)
    with open("best_config.pkl", "wb") as f: pickle.dump(initial_best_config, f)
    return search_space, initial_best_config

__all__ = [
    "MarlHyperoptConfig",
    "SarlHyperoptConfig",
    "set_sarl_config",
    "set_marl_config",
    "get_active_config",
    "test_score_evaluation",
    "elo_evaluation",
    "best_agent_elo_evaluation",
    "test_agents_elo_evaluation",
    "marl_run_training",
    "sarl_run_training",
    "marl_objective",
    "sarl_objective",
    "analyze_trial_stats",
    "analyze_hyperparameter_importance",
    "predict_best_config",
    "plot_general_trends",
    "simulate_elo_math",
    "save_search_space"
]
