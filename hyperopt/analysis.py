import pickle
import glob
import inspect
from pathlib import Path
from collections import defaultdict
from pprint import pprint
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
from hyperopt import space_eval
from hyperopt.pyll import as_apply, stochastic
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from .config import get_active_config

def flatten_dict(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict): items.extend(flatten_dict(v, new_key, sep=sep).items())
        else: items.append((new_key, v))
    return dict(items)

def safe_value(v):
    if inspect.isclass(v) or callable(v): return v.__name__ if hasattr(v, "__name__") else str(v)
    if isinstance(v, (int, float, str, bool, type(None))): return v
    if isinstance(v, (list, tuple)) and len(v) == 1: return v[0]
    return str(v)

def find_stats_file(trial_folder):
    stats_files = list(Path(trial_folder).rglob("stats.pkl"))
    if not stats_files: return None
    def get_step_count(path):
        parent = path.parent.name
        return int(parent.split("_")[1]) if parent.startswith("step_") else 0
    return sorted(stats_files, key=get_step_count)[-1]

def analyze_trial_stats(trials_path, stat_map, checkpoints_folder="./checkpoints", config=None, trials_to_skip=None):
    if config is None:
        config = get_active_config()
        if config is None: return print("No config.")
    if trials_to_skip is None: trials_to_skip = []
    max_stats = {key: [float("-inf"), 0] for key in stat_map.keys()}
    max_final_stats = {key: [float("-inf"), 0] for key in stat_map.keys()}
    try:
        with open(trials_path, "rb") as f: trials = pickle.load(f)
    except FileNotFoundError: return print(f"'{trials_path}' not found.")
    for i, trial in enumerate(trials.trials):
        trial_num = i + 1
        if trial_num in trials_to_skip or trial["result"]["status"] != "ok": continue
        trial_folder_path = Path(checkpoints_folder) / f"{config.file_name}_{trial_num}"
        if not trial_folder_path.exists():
            trial_folder_path = Path(checkpoints_folder) / f"{config.file_name}_best_{trial_num}"
        if not trial_folder_path.exists():
            match = glob.glob(f"{checkpoints_folder}/{config.file_name}*{trial_num}")
            if match: trial_folder_path = Path(match[0])
            else: continue
        stats_path = find_stats_file(trial_folder_path)
        if not stats_path: continue
        with open(stats_path, "rb") as f: stats = pickle.load(f)
        for d_name, (cat, m_key) in stat_map.items():
            if cat in stats and m_key in stats[cat]:
                raw = stats[cat][m_key]
                data_list = [d["value"] if isinstance(d, dict) and "value" in d else (list(d.values())[0] if isinstance(d, dict) else d) for d in raw]
                if not data_list: continue
                if max(data_list) > max_stats[d_name][0]: max_stats[d_name] = [max(data_list), trial_num]
                if data_list[-1] > max_final_stats[d_name][0]: max_final_stats[d_name] = [data_list[-1], trial_num]
    def print_table(title, data_dict):
        print(f"\n--- {title} ---")
        print(f"| {'Statistic':<30} | {'Value':<10} | {'Trial #':<8} |")
        for key in stat_map.keys():
            val, idx = data_dict[key]
            val_str = f"{val:<10.4f}" if val != float("-inf") else "N/A"
            print(f"| **{key:<30}** | {val_str} | {idx:<8} |")
    print_table("Max Stats (Entire Run)", max_stats)
    print_table("Max Stats (Final Result)", max_final_stats)

def analyze_hyperparameter_importance(trials_path, search_space=None):
    try:
        with open(trials_path, "rb") as f: trials = pickle.load(f)
    except FileNotFoundError: return print("Trials file not found.")
    if search_space is None:
        try:
            with open("search_space.pkl", "rb") as f: search_space = pickle.load(f)
        except FileNotFoundError: return print("No search space.")
    params, losses = [], []
    for trial in trials.trials:
        vals = {k: v[0] if len(v) > 0 else None for k, v in trial["misc"]["vals"].items()}
        try:
            l = -trial["result"]["loss"]
            if not np.isnan(l) and l != 0:
                losses.append(l)
                params.append(space_eval(search_space, vals))
        except: continue
    if not params: return print("No valid data.")
    flat_params = [flatten_dict(p) for p in params]
    for fp in flat_params:
        for k in fp: fp[k] = safe_value(fp[k])
    df = pd.DataFrame(flat_params)
    df["loss"] = losses
    correlations, hyperparameters = {}, set(df.columns) - {"loss"}
    for hp in hyperparameters:
        num_col = pd.to_numeric(df[hp], errors="coerce")
        if not num_col.isnull().all():
            clean = pd.DataFrame({"p": num_col, "l": df["loss"]}).dropna()
            if len(clean) > 1: correlations[hp] = {"P": abs(scipy.stats.pearsonr(clean["p"], clean["l"])[0]), "S": abs(scipy.stats.spearmanr(clean["p"], clean["l"])[0])}
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df[cat_cols] = encoder.fit_transform(df[cat_cols])
    df = df.fillna(-1)
    X, y = df.drop("loss", axis=1), df["loss"]
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    rf_imp = dict(zip(X.columns, model.feature_importances_))
    final_data = {k: {"Pearson": correlations.get(k,{}).get("P",0), "Spearman": correlations.get(k,{}).get("S",0), "RF": rf_imp.get(k,0)} for k in set(correlations).union(rf_imp)}
    res_df = pd.DataFrame.from_dict(final_data, orient="index")
    for c in res_df.columns: res_df[c] /= res_df[c].max() if res_df[c].max() > 0 else 1
    res_df["Score"] = res_df.mean(axis=1)
    res_df = res_df.sort_values("Score")
    plt.figure(figsize=(10, max(6, len(res_df) * 0.5)))
    res_df.drop("Score", axis=1).plot(kind="barh", ax=plt.gca())
    plt.tight_layout(); plt.show()
    return model, encoder, X.columns, cat_cols

def predict_best_config(search_space, model, encoder, feature_names, cat_cols, n_candidates=5000):
    raw_configs = [stochastic.sample(as_apply(search_space)) for _ in range(n_candidates)]
    flat_params = [flatten_dict(p) for p in raw_configs]
    for fp in flat_params:
        for k in list(fp.keys()): fp[k] = safe_value(fp[k])
    candidates_df = pd.DataFrame(flat_params)
    for col in cat_cols:
        if col in candidates_df.columns: candidates_df[col] = encoder.transform(candidates_df[[col]])
    candidates_df = candidates_df.replace([np.inf, -np.inf], np.nan).fillna(-1)
    for col in feature_names:
        if col not in candidates_df.columns: candidates_df[col] = -1
    candidates_df = candidates_df[feature_names]
    predicted_losses = model.predict(candidates_df)
    max_idx = np.argmax(predicted_losses)
    pprint(raw_configs[max_idx])
    plt.hist(predicted_losses, bins=100); plt.show()

def plot_general_trends(trials_path, search_space=None, min_loss=-float("inf")):
    try:
        with open(trials_path, "rb") as f: trials = pickle.load(f)
        if search_space is None:
            with open("search_space.pkl", "rb") as f: search_space = pickle.load(f)
    except: return print("Could not load trials/space.")
    params, losses = [], []
    for trial in trials.trials:
        vals = {k: v[0] if len(v) > 0 else None for k, v in trial["misc"]["vals"].items()}
        losses.append(-trial["result"]["loss"]); params.append(space_eval(search_space, vals))
    plt.plot(losses); plt.show()
    filtered = [(l, p) for l, p in zip(losses, params) if l > min_loss]
    if not filtered: return
    losses, params = zip(*filtered)
    flat_params = [flatten_dict(p) for p in params]
    for fp in flat_params:
        for k in list(fp.keys()): fp[k] = safe_value(fp[k])
    hyperparameters = sorted(set().union(*(p.keys() for p in flat_params)))
    for hp_name in hyperparameters:
        values = [p.get(hp_name) for p in flat_params]
        valid_pairs = [(v, l) for v, l in zip(values, losses) if v is not None]
        if len(set(v for v, _ in valid_pairs)) <= 1: continue
        data_map = defaultdict(lambda: {"losses": []})
        for v, l in valid_pairs: data_map[v]["losses"].append(l)
        summary = {v: {"mean": np.mean(d["losses"]), "min": np.min(d["losses"]), "max": np.max(d["losses"]), "count": len(d["losses"])} for v, d in data_map.items()}
        keys_sorted = sorted(summary.keys(), key=lambda x: (float(x) if isinstance(x, (int, float)) or (isinstance(x, str) and x.replace('.','',1).isdigit()) else str(x)))
        means_sorted = [summary[k]["mean"] for k in keys_sorted]
        plt.figure(figsize=(15, 6))
        plt.bar(range(len(keys_sorted)), means_sorted); plt.xticks(range(len(keys_sorted)), keys_sorted, rotation=45); plt.show()

def simulate_elo_math(start_elo=1400, total_games=100):
    try: from elo.elo import StandingsTable
    except ImportError: return print("No elo.elo module.")
    class DummyPlayer:
        def __init__(self, name): self.name = name
        def __str__(self): return self.name
    p1, p2 = DummyPlayer("Hero"), DummyPlayer("Opponent")
    win_pcts, p1_elos = [], []
    for p1_wins in range(total_games + 1):
        p2_wins = total_games - p1_wins
        table = StandingsTable([p1, p2], start_elo=start_elo)
        for _ in range(p1_wins): table.add_result(p1, p2, result=1)
        for _ in range(p2_wins): table.add_result(p1, p2, result=-1)
        elo_df = table.bayes_elo(return_params=True)["Elo table"]
        p2_curr = elo_df.loc["Opponent", "Elo"] if "Opponent" in elo_df.index else elo_df.iloc[1]["Elo"]
        p1_val_raw = elo_df.loc["Hero", "Elo"] if "Hero" in elo_df.index else elo_df.iloc[0]["Elo"]
        p1_val = p1_val_raw - (p2_curr - start_elo)
        win_pcts.append(p1_wins / total_games); p1_elos.append(p1_val)
    plt.plot(win_pcts, p1_elos); plt.show()
