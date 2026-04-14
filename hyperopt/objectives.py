import os
import gc
import pickle
from typing import Dict, Union
from hyperopt import STATUS_OK, STATUS_FAIL, space_eval
from .config import get_active_config, MarlHyperoptConfig, SarlHyperoptConfig
from .training import marl_run_training, sarl_run_training

def _determine_trial_name(config: Union[MarlHyperoptConfig, SarlHyperoptConfig], params: Dict) -> str:
    trials_path = f"./{config.file_name}_trials.p"
    if os.path.exists(trials_path):
        with open(trials_path, "rb") as f:
            trials = pickle.load(f)
        return f"{config.file_name}_{len(trials.trials) + 1}"
    try:
        with open("best_config.pkl", "rb") as f: initial_best_configs = pickle.load(f)
        with open("search_space.pkl", "rb") as f: search_space = pickle.load(f)
    except FileNotFoundError: return f"{config.file_name}_1"
    cur_agent_config = config.agent_config(config.prep_params(params.copy()), config.game_config(env_factory=config.env_factory))
    for i, raw_cfg in enumerate(initial_best_configs, 1):
        target_agent_config = config.agent_config(config.prep_params(space_eval(search_space, raw_cfg)), config.game_config(env_factory=config.env_factory))
        if cur_agent_config == target_agent_config: return f"{config.file_name}_best_{i}"
    return f"{config.file_name}_1"

def _check_params_validity(params: Dict) -> None:
    if "min_minibatch" in params and "min_replay" in params:
        assert params["min_replay"] >= params["min_minibatch"], "Replay min must be >= minibatch"
    if "replay_size" in params and "min_replay" in params:
        assert params["replay_size"] > params["min_replay"], "Replay size must be > min replay"

def marl_objective(params):
    gc.collect()
    config = get_active_config()
    name = _determine_trial_name(config, params)
    try:
        _check_params_validity(params)
        return {"status": STATUS_OK, "loss": marl_run_training(params, name)}
    except Exception as e:
        print(f"MARL Training failed: {e}")
        return {"status": STATUS_FAIL, "loss": 0}

def sarl_objective(params):
    gc.collect()
    config = get_active_config()
    name = _determine_trial_name(config, params)
    try:
        _check_params_validity(params)
        return {"status": STATUS_OK, "loss": sarl_run_training(params, name)}
    except Exception as e:
        print(f"SARL Training failed: {e}")
        return {"status": STATUS_FAIL, "loss": 0}
