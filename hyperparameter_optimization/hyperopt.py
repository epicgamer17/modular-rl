import dill as pickle
import os
import gc
import glob
import sys
import inspect
from pathlib import Path
from collections import defaultdict
from pprint import pprint
from dataclasses import dataclass, field
from typing import Any, Callable, List, Literal, Optional, Dict, Union

import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib

# Hyperopt and ML Imports
from hyperopt import STATUS_OK, STATUS_FAIL, space_eval
from hyperopt.pyll import as_apply, stochastic
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder

# --- Mocks and Local Imports ---
# These are placeholders to ensure the script parses even if your local environment
# is missing the specific agents/elo packages.
try:
    from actors.baselines.random_agent import RandomAgent
    from elo.elo import StandingsTable
except ImportError:
    print("Warning: Local 'agents' or 'elo' modules not found. Using mocks.")
    RandomAgent = lambda: None

    class MockStandingsTable:
        def __init__(self, players, start_elo=1400):
            self.players = players
            self.start_elo = start_elo
            self.results = []

        def add_player(self, player):
            self.players.append(player)

        def play_matches(self, *args, **kwargs):
            pass

        def add_result(self, p1, p2, result):
            pass

        def bayes_elo(self, return_params=False):
            # Return a dummy DF structure expected by the code
            data = {"Elo": [1400] * len(self.players)}
            idx = [p.name if hasattr(p, "name") else str(p) for p in self.players]
            return {"Elo table": pd.DataFrame(data, index=idx)}

    StandingsTable = MockStandingsTable


# --- Configuration Classes ---


@dataclass
class MarlHyperoptConfig:
    file_name: str
    eval_method: str  # "elo", "best_agent_elo", "test_agents_elo"
    best_agent: Any
    env_factory: Callable
    prep_params: Callable
    agent_class: Any
    agent_config: Callable
    game_config: Callable
    games_per_pair: int
    num_opps: int
    table: Any
    play_sequence: Callable
    checkpoint_interval: int = 100
    test_interval: int = 100
    test_trials: int = 10
    test_agents: List = field(default_factory=lambda: [RandomAgent()])
    test_agent_weights: List[float] = field(default_factory=lambda: [1.0])
    device: str = "cpu"


@dataclass
class SarlHyperoptConfig:
    file_name: str
    eval_method: Literal[
        "final_score", "rolling_average", "final_score_rolling_average"
    ]
    env_factory: Callable
    prep_params: Callable
    agent_class: Any
    agent_config: Callable
    game_config: Callable
    checkpoint_interval: int = 100
    test_interval: int = 100
    test_trials: int = 10
    last_n_rolling_avg: int = 10
    device: str = "cpu"


# --- Global State Management ---

_MARL_CONFIG: Optional[MarlHyperoptConfig] = None
_SARL_CONFIG: Optional[SarlHyperoptConfig] = None


def set_sarl_config(config: SarlHyperoptConfig):
    """Set the global SARL config from another file before running hyperopt."""
    global _SARL_CONFIG
    _SARL_CONFIG = config


def set_marl_config(config: MarlHyperoptConfig):
    """Set the global MARL config from another file before running hyperopt."""
    global _MARL_CONFIG
    _MARL_CONFIG = config


def get_active_config():
    """Helper to retrieve the active global configuration."""
    if "_MARL_CONFIG" in globals() and _MARL_CONFIG is not None:
        return _MARL_CONFIG
    if "_SARL_CONFIG" in globals() and _SARL_CONFIG is not None:
        return _SARL_CONFIG
    return None


def save_search_space(search_space, initial_best_config=None):
    if initial_best_config is None:
        initial_best_config = [{}]

    with open("search_space.pkl", "wb") as f:
        pickle.dump(search_space, f)
    with open("best_config.pkl", "wb") as f:
        pickle.dump(initial_best_config, f)

    return search_space, initial_best_config


# --- Training & Trial Helper Functions ---


def _determine_trial_name(
    config: Union[MarlHyperoptConfig, SarlHyperoptConfig], params: Dict
) -> str:
    """
    Determines the unique name for the trial based on previous trials
    or initial best configuration matches.
    """
    trials_path = f"./{config.file_name}_trials.p"

    # 1. Check if we are resuming an existing trials file
    if os.path.exists(trials_path):
        with open(trials_path, "rb") as f:
            trials = pickle.load(f)
        return "{}_{}".format(config.file_name, len(trials.trials) + 1)

    # 2. Check if this param set matches one of the 'initial_best_configs'
    try:
        with open("best_config.pkl", "rb") as f:
            initial_best_configs = pickle.load(f)
        with open("search_space.pkl", "rb") as f:
            search_space = pickle.load(f)
    except FileNotFoundError:
        # If files don't exist, assume this is simply the first trial
        print("No search_space or best_config found. Starting trial 1.")
        return "{}_1".format(config.file_name)

    current_agent_config = config.agent_config(
        config.prep_params(params.copy()),
        config.game_config(env_factory=config.env_factory),
    )

    for i, best_config_raw in enumerate(initial_best_configs, 1):
        # Evaluate the search space to get actual values
        best_config_resolved = space_eval(search_space, best_config_raw)

        target_agent_config = config.agent_config(
            config.prep_params(best_config_resolved),
            config.game_config(env_factory=config.env_factory),
        )

        if current_agent_config == target_agent_config:
            print(f"Using initial best config #{i}")
            return "{}_best_{}".format(config.file_name, i)

    print("No initial best config matched, first trial")
    return "{}_1".format(config.file_name)


def _check_params_validity(params: Dict) -> None:
    """Asserts constraints on hyperparameters."""
    if "min_replay_buffer_size" in params and "minibatch_size" in params:
        assert (
            params["min_replay_buffer_size"] >= params["minibatch_size"]
        ), "Replay buffer min size must be >= minibatch size"

    if "replay_buffer_size" in params and "min_replay_buffer_size" in params:
        assert (
            params["replay_buffer_size"] > params["min_replay_buffer_size"]
        ), "Replay buffer size must be > min replay buffer size"


def _env_factory_safe(env_factory_fn):
    """Attempts to create env with rgb_array, falls back to default."""
    try:
        return env_factory_fn(render_mode="rgb_array")
    except TypeError:
        return env_factory_fn()


# --- Evaluation Logic ---


def test_score_evaluation(agent, eval_method, num_trials=10, last_n=10):
    """
    Evaluates SARL agents.
    Returns: A NEGATIVE score (because Hyperopt minimizes loss).
    """
    final_score = agent.test(num_trials=num_trials, dir="./checkpoints/")["score"]

    # Get history of scores
    score_history = [
        stat_dict["score"] for stat_dict in agent.stats.stats["test_score"]["score"]
    ]

    if eval_method == "final_score":
        return -final_score

    elif eval_method == "rolling_average":
        recent_scores = score_history[-last_n:]
        if not recent_scores:
            return 0.0
        return -np.around(np.mean(recent_scores), 1)

    elif eval_method == "final_score_rolling_average":
        recent_scores = score_history[-last_n:]
        rolling_avg = np.mean(recent_scores) if recent_scores else 0
        combined = (final_score + rolling_avg) / 2
        return -combined

    return 0.0


def elo_evaluation(agent):
    assert _MARL_CONFIG is not None, "Config not set."
    config = _MARL_CONFIG

    opponents_indices = np.random.choice(
        range(len(config.table.players)),
        size=min(config.num_opps, len(config.table.players)),
        replace=False,
    )

    opp_names = [config.table.players[o].name for o in opponents_indices]
    print(f"Testing against opponents: {opp_names}")

    config.table.add_player(agent)

    # Save table state before playing
    with open("hyperopt_elo_table.pkl", "wb") as f:
        pickle.dump(config.table, f)

    if len(opponents_indices) == 0:
        return 0

    config.table.play_matches(
        play_sequence=config.play_sequence,
        player_index=len(config.table.players) - 1,
        opponent_indices=opponents_indices,
        games_per_pair=config.games_per_pair,
    )

    # Save table state after playing
    with open("hyperopt_elo_table.pkl", "wb") as f:
        pickle.dump(config.table, f)

    bayes_elo = config.table.bayes_elo()["Elo table"]
    elo = bayes_elo.iloc[-1]["Elo"]

    print(bayes_elo)
    print(f"Elo: {elo}")
    return -elo


def best_agent_elo_evaluation(agent):
    assert _MARL_CONFIG is not None, "Config not set."
    config = _MARL_CONFIG

    # Create a temporary table with just the best agent
    table = StandingsTable([config.best_agent], start_elo=1400)
    table.add_player(agent)

    table.play_matches(
        play_sequence=config.play_sequence,
        player_index=1,
        opponent_indices=[0],
        games_per_pair=config.games_per_pair,
    )

    bayes_elo = table.bayes_elo()["Elo table"]

    # Calculate ELO difference relative to 1400 baseline
    agent_elo = bayes_elo.iloc[-1]["Elo"]
    best_agent_baseline_elo = bayes_elo.iloc[0]["Elo"]

    diff = best_agent_baseline_elo - 1400
    normalized_elo = agent_elo - diff

    print(bayes_elo)
    print(f"Normalized Elo: {normalized_elo}")
    return -normalized_elo


def test_agents_elo_evaluation(agent):
    assert _MARL_CONFIG is not None, "Config not set."
    config = _MARL_CONFIG
    total_weighted_loss = 0

    for test_agent, weight in zip(config.test_agents, config.test_agent_weights):
        table = StandingsTable([test_agent], start_elo=1400)
        table.add_player(agent)

        table.play_matches(
            play_sequence=config.play_sequence,
            player_index=1,
            opponent_indices=[0],
            games_per_pair=config.games_per_pair,
        )

        bayes_elo = table.bayes_elo()["Elo table"]

        agent_elo = bayes_elo.iloc[-1]["Elo"]
        opponent_elo = bayes_elo.iloc[0]["Elo"]

        # Normalize
        diff = opponent_elo - 1400
        normalized_elo = agent_elo - diff

        weighted_elo = normalized_elo * weight
        total_weighted_loss -= weighted_elo  # accumulating negative elo (loss)

        print(bayes_elo)
        print(f"Elo vs {test_agent}: {normalized_elo} (Weighted: {weighted_elo})")

    return total_weighted_loss


# --- Training Loops ---


def marl_run_training(params, agent_name):
    assert _MARL_CONFIG is not None, "Config not set."
    config = _MARL_CONFIG

    params = config.prep_params(params)
    env = _env_factory_safe(config.env_factory)

    agent = config.agent_class(
        env=env,
        config=config.agent_config(
            config_dict=params,
            game_config=config.game_config(env_factory=config.env_factory),
        ),
        name=agent_name,
        device=config.device,
        test_agents=config.test_agents,
    )

    agent.checkpoint_interval = config.checkpoint_interval
    agent.test_interval = config.test_interval
    agent.test_trials = config.test_trials

    agent.train()

    if config.eval_method == "elo":
        return elo_evaluation(agent)
    elif config.eval_method == "best_agent_elo":
        return best_agent_elo_evaluation(agent)
    elif config.eval_method == "test_agents_elo":
        return test_agents_elo_evaluation(agent)
    else:
        raise NotImplementedError(f"Unknown eval method: {config.eval_method}")


def sarl_run_training(params, agent_name):
    assert _SARL_CONFIG is not None, "SARL Config not set."
    config = _SARL_CONFIG

    params = config.prep_params(params)
    env = _env_factory_safe(config.env_factory)

    agent = config.agent_class(
        env=env,
        config=config.agent_config(
            config_dict=params,
            game_config=config.game_config(env_factory=config.env_factory),
        ),
        name=agent_name,
        device=config.device,
    )

    agent.checkpoint_interval = config.checkpoint_interval
    agent.test_interval = config.test_interval
    agent.test_trials = config.test_trials

    print(f"Starting training for SARL agent: {agent_name}")
    agent.train()
    print("Training complete. Starting test score evaluation.")

    return test_score_evaluation(
        agent,
        eval_method=config.eval_method,
        num_trials=config.test_trials,
        last_n=config.last_n_rolling_avg,
    )


# --- Objective Functions ---


def marl_objective(params):
    gc.collect()
    assert _MARL_CONFIG is not None, "Config not set."
    config = _MARL_CONFIG

    print("Params: ", params)
    name = _determine_trial_name(config, params)

    status = STATUS_OK
    try:
        _check_params_validity(params)
        score = marl_run_training(params, name)
    except AssertionError as e:
        status = STATUS_FAIL
        print(f"Exited due to invalid hyperparameter combination: {e}")
        return {"status": status, "loss": 0}
    except Exception as e:
        status = STATUS_FAIL
        print(f"MARL Training failed with error: {e}")
        return {"status": status, "loss": 0}

    print("Trial done")
    return {"status": status, "loss": score}


def sarl_objective(params):
    gc.collect()
    assert _SARL_CONFIG is not None, "SARL Config not set."
    config = _SARL_CONFIG

    print("Params: ", params)
    name = _determine_trial_name(config, params)

    status = STATUS_OK
    try:
        _check_params_validity(params)
        score = sarl_run_training(params, name)
    except AssertionError as e:
        status = STATUS_FAIL
        print(f"Exited due to invalid hyperparameter combination: {e}")
        return {"status": status, "loss": 0}
    except Exception as e:
        status = STATUS_FAIL
        print(f"SARL Training failed with error: {e}")
        return {"status": status, "loss": 0}

    print("Trial done")
    return {"status": status, "loss": score}


# --- Analysis Helper Functions ---


def flatten_dict(d, parent_key="", sep="."):
    """Recursively flattens a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def safe_value(v):
    """Safely converts parameters to string or primitive types for analysis."""
    if inspect.isclass(v):
        return v.__name__
    if callable(v):
        return v.__name__ if hasattr(v, "__name__") else str(v)
    if isinstance(v, (int, float, str, bool, type(None))):
        return v
    if isinstance(v, (list, tuple)) and len(v) == 1:
        return v[0]
    return str(v)


def find_stats_file(trial_folder):
    """
    Recursively searches for 'stats.pkl' in the trial folder.
    Returns the path to the one in the directory with the highest step count.
    """
    stats_files = list(Path(trial_folder).rglob("stats.pkl"))

    if not stats_files:
        return None

    def get_step_count(path):
        parent = path.parent.name
        if parent.startswith("step_"):
            try:
                return int(parent.split("_")[1])
            except ValueError:
                return 0
        return 0

    return sorted(stats_files, key=get_step_count)[-1]


# --- Main Analysis Functions ---


def analyze_trial_stats(
    trials_path,
    stat_map,
    checkpoints_folder="./checkpoints",
    config=None,
    trials_to_skip=None,
):
    """
    General analysis of trial stats.
    stat_map: {'Display Name': ('category', 'key')}
    """
    if config is None:
        config = get_active_config()
        if config is None:
            print("Error: No configuration found.")
            return

    if trials_to_skip is None:
        trials_to_skip = []

    max_stats = {key: [float("-inf"), 0] for key in stat_map.keys()}
    max_final_stats = {key: [float("-inf"), 0] for key in stat_map.keys()}

    try:
        with open(trials_path, "rb") as f:
            trials = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: '{trials_path}' not found.")
        return

    print(f"--- Analyzing Trials for {config.file_name} ---")

    for i, trial in enumerate(trials.trials):
        trial_num = i + 1
        if trial_num in trials_to_skip:
            continue

        if trial["result"]["status"] == "ok":
            try:
                trial_folder_name = f"{config.file_name}_{trial_num}"
                trial_folder_path = Path(checkpoints_folder) / trial_folder_name

                if not trial_folder_path.exists():
                    trial_folder_name = f"{config.file_name}_best_{trial_num}"
                    trial_folder_path = Path(checkpoints_folder) / trial_folder_name

                if not trial_folder_path.exists():
                    matches = glob.glob(
                        f"{checkpoints_folder}/{config.file_name}*{trial_num}"
                    )
                    if matches:
                        trial_folder_path = Path(matches[0])
                    else:
                        continue

                stats_path = find_stats_file(trial_folder_path)
                if not stats_path:
                    continue

                with open(stats_path, "rb") as f:
                    stats = pickle.load(f)

                for display_name, (category, metric_key) in stat_map.items():
                    if category in stats and metric_key in stats[category]:
                        raw_data = stats[category][metric_key]

                        # Fix: Handle list of dicts vs list of values
                        if raw_data and isinstance(raw_data[0], dict):
                            # assume value is stored in 'value' or just take the first value found
                            if "value" in raw_data[0]:
                                data_list = [d["value"] for d in raw_data]
                            else:
                                data_list = [list(d.values())[0] for d in raw_data]
                        else:
                            data_list = raw_data

                        if not data_list:
                            continue

                        curr_max = max(data_list)
                        if curr_max > max_stats[display_name][0]:
                            max_stats[display_name][0] = curr_max
                            max_stats[display_name][1] = trial_num

                        curr_final = data_list[-1]
                        if curr_final > max_final_stats[display_name][0]:
                            max_final_stats[display_name][0] = curr_final
                            max_final_stats[display_name][1] = trial_num

            except Exception as e:
                print(f"Error processing Trial {trial_num}: {e}")

    def print_table(title, data_dict):
        print(f"\n--- {title} ---")
        print(f"| {'Statistic':<30} | {'Value':<10} | {'Trial #':<8} |")
        print(f"|{'-'*32}|{'-'*12}|{'-'*10}|")
        for key in stat_map.keys():
            val, idx = data_dict[key]
            if val == float("-inf"):
                val_str = "N/A"
            else:
                val_str = f"{val:<10.4f}"
            print(f"| **{key:<30}** | {val_str} | {idx:<8} |")

    print_table("Max Stats (Entire Run)", max_stats)
    print_table("Max Stats (Final Result)", max_final_stats)


def analyze_hyperparameter_importance(trials_path, search_space=None):
    """
    Calculates hyperparameter importance and trains a Regressor for prediction.
    """
    try:
        with open(trials_path, "rb") as f:
            trials = pickle.load(f)
    except FileNotFoundError:
        print("Trials file not found.")
        return None, None, None, None

    if search_space is None:
        try:
            with open("search_space.pkl", "rb") as f:
                search_space = pickle.load(f)
        except FileNotFoundError:
            print("Error: search_space.pkl not found and not provided.")
            return None, None, None, None

    params = []
    losses = []

    for trial in trials.trials:
        vals = {
            k: val[0] if len(val) > 0 else None
            for k, val in trial["misc"]["vals"].items()
        }
        try:
            param_dict = space_eval(search_space, vals)
            loss = -trial["result"]["loss"]
            if not np.isnan(loss) and loss != 0:
                losses.append(loss)
                params.append(param_dict)
        except Exception:
            continue

    if not params:
        print("No valid data.")
        return None, None, None, None

    flat_params = [flatten_dict(p) for p in params]
    for fp in flat_params:
        for k in fp:
            fp[k] = safe_value(fp[k])

    df = pd.DataFrame(flat_params)
    df["loss"] = losses

    correlations = {}
    hyperparameters = set(df.columns) - {"loss"}

    for hp in hyperparameters:
        num_col = pd.to_numeric(df[hp], errors="coerce")
        if not num_col.isnull().all():
            clean = pd.DataFrame({"p": num_col, "l": df["loss"]}).dropna()
            if len(clean) > 1:
                pr, _ = scipy.stats.pearsonr(clean["p"], clean["l"])
                sr, _ = scipy.stats.spearmanr(clean["p"], clean["l"])
                correlations[hp] = {"P": abs(pr), "S": abs(sr)}

    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df[cat_cols] = encoder.fit_transform(df[cat_cols])
    df = df.fillna(-1)

    X, y = df.drop("loss", axis=1), df["loss"]
    model = RandomForestRegressor(n_estimators=100, n_jobs=-1)
    model.fit(X, y)

    rf_imp = dict(zip(X.columns, model.feature_importances_))
    final_data = {}
    for k in set(correlations).union(rf_imp):
        final_data[k] = {
            "Pearson": correlations.get(k, {}).get("P", 0),
            "Spearman": correlations.get(k, {}).get("S", 0),
            "RF": rf_imp.get(k, 0),
        }

    res_df = pd.DataFrame.from_dict(final_data, orient="index")
    for c in res_df.columns:
        res_df[c] /= res_df[c].max() if res_df[c].max() > 0 else 1

    res_df["Score"] = res_df.mean(axis=1)
    res_df = res_df.sort_values("Score")

    plt.figure(figsize=(10, max(6, len(res_df) * 0.5)))
    res_df.drop("Score", axis=1).plot(kind="barh", ax=plt.gca())
    plt.title("Generalized Hyperparameter Importance")
    plt.tight_layout()
    plt.show()

    return model, encoder, X.columns, cat_cols


def predict_best_config(
    search_space, model, encoder, feature_names, categorical_cols, n_candidates=5000
):
    """
    Uses the trained RF model to predict the best configuration from random samples.
    """
    print(f"Generating {n_candidates} raw candidates for prediction...")

    search_space_expr = as_apply(search_space)
    raw_configs = [stochastic.sample(search_space_expr) for _ in range(n_candidates)]

    flat_params = [flatten_dict(p) for p in raw_configs]
    for fp in flat_params:
        for k in list(fp.keys()):
            fp[k] = safe_value(fp[k])

    candidates_df = pd.DataFrame(flat_params)

    # Apply encoding from training phase
    for col in categorical_cols:
        if col in candidates_df.columns:
            candidates_df[col] = encoder.transform(candidates_df[[col]])

    candidates_df = candidates_df.replace([np.inf, -np.inf], np.nan).fillna(-1)

    # Align columns
    for col in feature_names:
        if col not in candidates_df.columns:
            candidates_df[col] = -1
    candidates_df = candidates_df[feature_names]

    predicted_losses = model.predict(candidates_df)

    max_idx = np.argmax(predicted_losses)
    best_loss = predicted_losses[max_idx]
    best_config_raw = raw_configs[max_idx]

    print(f"Maximum Predicted Loss (Acquisition Score): {best_loss:.4f}")
    print("Raw Best Hyperparameters:")
    pprint(best_config_raw)

    plt.figure(figsize=(10, 6))
    plt.hist(predicted_losses, bins=100, edgecolor="black", alpha=0.7)
    plt.title("Distribution of Predicted Losses for Candidates")
    plt.xlabel("Predicted Loss")
    plt.ylabel("Frequency")
    plt.show()


def plot_general_trends(trials_path, search_space=None, min_loss=-float("inf")):
    """
    Plots trends for all hyperparameters found in the trials file.
    """
    try:
        with open(trials_path, "rb") as f:
            trials = pickle.load(f)
    except:
        print("Could not load trials.")
        return

    if search_space is None:
        try:
            with open("search_space.pkl", "rb") as f:
                search_space = pickle.load(f)
        except:
            print("Could not load search space.")
            return

    params, losses = [], []
    for trial in trials.trials:
        vals = {
            k: val[0] if len(val) > 0 else None
            for k, val in trial["misc"]["vals"].items()
        }
        param_dict = space_eval(search_space, vals)
        loss = -trial["result"]["loss"]
        losses.append(loss)
        params.append(param_dict)

    plt.figure(figsize=(15, 5))
    plt.plot(losses, label="ELO/Score")
    if len(losses) > 1:
        z = np.polyfit(range(len(losses)), losses, 1)
        plt.plot(np.poly1d(z)(range(len(losses))), "r--", label="Trend")
    plt.title("Score over Trials (All)")
    plt.legend()
    plt.show()

    filtered = [(l, p) for l, p in zip(losses, params) if l > min_loss]
    if not filtered:
        print(f"No trials above min_loss {min_loss}.")
        return
    losses, params = zip(*filtered)

    flat_params = [flatten_dict(p) for p in params]
    for fp in flat_params:
        for k in list(fp.keys()):
            fp[k] = safe_value(fp[k])

    hyperparameters = sorted(set().union(*(p.keys() for p in flat_params)))
    max_hp_vals = {}

    for hp_name in hyperparameters:
        values = [p.get(hp_name) for p in flat_params]
        valid_pairs = [(v, l) for v, l in zip(values, losses) if v is not None]

        if len(set(v for v, _ in valid_pairs)) <= 1:
            continue

        data_map = defaultdict(lambda: {"losses": []})
        for v, l in valid_pairs:
            data_map[v]["losses"].append(l)

        summary = {}
        for v, d in data_map.items():
            summary[v] = {
                "mean": np.mean(d["losses"]),
                "min": np.min(d["losses"]),
                "max": np.max(d["losses"]),
                "count": len(d["losses"]),
            }

        means = {v: d["mean"] for v, d in summary.items()}
        max_hp_vals[hp_name] = max(means, key=means.get)

        keys_raw = list(summary.keys())
        try:
            keys_sorted = sorted(keys_raw, key=lambda x: float(x))
        except:
            keys_sorted = sorted(keys_raw, key=str)

        means_sorted = [summary[k]["mean"] for k in keys_sorted]
        mins_sorted = [summary[k]["min"] for k in keys_sorted]
        maxs_sorted = [summary[k]["max"] for k in keys_sorted]
        counts_sorted = [summary[k]["count"] for k in keys_sorted]

        yerr = [
            np.array(means_sorted) - np.array(mins_sorted),
            np.array(maxs_sorted) - np.array(means_sorted),
        ]

        plt.figure(figsize=(15, 6))
        x_pos = range(len(keys_sorted))
        bars = plt.bar(
            x_pos,
            means_sorted,
            yerr=yerr,
            capsize=5,
            color="skyblue",
            edgecolor="black",
        )

        def fmt(x):
            return f"{x:.3g}" if isinstance(x, float) else str(x)[:20]

        plt.xticks(x_pos, [fmt(k) for k in keys_sorted], rotation=45, ha="right")

        y_max = max(maxs_sorted) if maxs_sorted else 1
        plt.ylim(top=y_max * 1.1)

        for bar, count, mean_val, k in zip(
            bars, counts_sorted, means_sorted, keys_sorted
        ):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                summary[k]["max"] + (y_max * 0.02),
                f"N={count}",
                ha="center",
                fontsize=8,
                weight="bold",
            )
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{mean_val:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="darkred",
            )

        plt.title(f"Mean Score for {hp_name} (with Min/Max range)")
        plt.tight_layout()
        plt.show()

    print("\nBest Values found per Hyperparameter:")
    pprint(max_hp_vals)


def simulate_elo_math(start_elo=1400, total_games=100):
    """
    Simulates Elo curve logic using the StandingsTable.
    """
    try:
        from elo.elo import StandingsTable
    except ImportError:
        print("Cannot simulate ELO math without 'elo.elo' module.")
        return

    class DummyPlayer:
        def __init__(self, name):
            self.name = name

        def __str__(self):
            return self.name

    p1 = DummyPlayer("Hero")
    p2 = DummyPlayer("Opponent")

    win_percentages = []
    p1_elos = []

    print("Simulating ELO curve...")
    for p1_wins in range(total_games + 1):
        p2_wins = total_games - p1_wins

        table = StandingsTable([p1, p2], start_elo=start_elo)

        # Batch add results
        for _ in range(p1_wins):
            table.add_result(p1, p2, result=1)
        for _ in range(p2_wins):
            table.add_result(p1, p2, result=-1)

        elo_df = table.bayes_elo(return_params=True)["Elo table"]

        # We need to find the specific rows for our dummy players
        # The index usually matches the name
        try:
            p2_curr = elo_df.loc["Opponent", "Elo"]
            p1_val_raw = elo_df.loc["Hero", "Elo"]
        except KeyError:
            # Fallback if index is integer-based
            p2_curr = elo_df.iloc[1]["Elo"]
            p1_val_raw = elo_df.iloc[0]["Elo"]

        # Normalize: Keep P2 fixed at start_elo relative to P1
        diff = p2_curr - start_elo
        p1_val = p1_val_raw - diff

        win_percentages.append(p1_wins / total_games)
        p1_elos.append(p1_val)

    plt.figure(figsize=(10, 6))
    plt.plot(win_percentages, p1_elos, "b-o")
    plt.title(f"Elo Rating vs. Win Percentage ({total_games} games)")
    plt.xlabel("Win Percentage")
    plt.ylabel("Renormalized Elo")
    plt.grid(True, linestyle="--", alpha=0.7)

    mid = len(win_percentages) // 2
    plt.axvline(0.5, color="r", linestyle=":")
    plt.annotate(
        f"50% Win = {p1_elos[mid]:.0f} Elo",
        (0.5, p1_elos[mid]),
        xytext=(10, 0),
        textcoords="offset points",
    )
    plt.show()
