import pickle
import numpy as np
import pandas as pd
from typing import Any, List
from .config import get_active_config

try:
    from elo.elo import StandingsTable
except ImportError:
    class StandingsTable:
        def __init__(self, players, start_elo=1400):
            self.players, self.start_elo = players, start_elo
        def add_player(self, player): self.players.append(player)
        def play_matches(self, *args, **kwargs): pass
        def bayes_elo(self, return_params=False):
            return {"Elo table": pd.DataFrame({"Elo": [1400] * len(self.players)}, index=[str(p) for p in self.players])}

def test_score_evaluation(agent, eval_method, num_trials=10, last_n=10):
    final_score = agent.test(num_trials=num_trials, dir="./checkpoints/")["score"]
    score_history = [s["score"] for s in agent.stats.stats["test_score"]["score"]]
    if eval_method == "final_score": return -final_score
    elif eval_method == "rolling_average":
        recent = score_history[-last_n:]
        return -np.around(np.mean(recent), 1) if recent else 0.0
    elif eval_method == "final_score_rolling_average":
        recent = score_history[-last_n:]
        rolling_avg = np.mean(recent) if recent else 0
        return -(final_score + rolling_avg) / 2
    return 0.0

def elo_evaluation(agent):
    config = get_active_config()
    opp_indices = np.random.choice(range(len(config.table.players)), size=min(config.num_opps, len(config.table.players)), replace=False)
    config.table.add_player(agent)
    with open("hyperopt_elo_table.pkl", "wb") as f: pickle.dump(config.table, f)
    if len(opp_indices) == 0: return 0
    config.table.play_matches(play_sequence=config.play_sequence, player_index=len(config.table.players) - 1, opponent_indices=opp_indices, games_per_pair=config.games_per_pair)
    with open("hyperopt_elo_table.pkl", "wb") as f: pickle.dump(config.table, f)
    bayes_elo = config.table.bayes_elo()["Elo table"]
    return -bayes_elo.iloc[-1]["Elo"]

def best_agent_elo_evaluation(agent):
    config = get_active_config()
    table = StandingsTable([config.best_agent], start_elo=1400)
    table.add_player(agent)
    table.play_matches(play_sequence=config.play_sequence, player_index=1, opponent_indices=[0], games_per_pair=config.games_per_pair)
    bayes_elo = table.bayes_elo()["Elo table"]
    return -(bayes_elo.iloc[-1]["Elo"] - (bayes_elo.iloc[0]["Elo"] - 1400))

def test_agents_elo_evaluation(agent):
    config = get_active_config()
    total_loss = 0
    for test_agent, weight in zip(config.test_agents, config.test_agent_weights):
        table = StandingsTable([test_agent], start_elo=1400)
        table.add_player(agent)
        table.play_matches(play_sequence=config.play_sequence, player_index=1, opponent_indices=[0], games_per_pair=config.games_per_pair)
        bayes_elo = table.bayes_elo()["Elo table"]
        total_loss -= (bayes_elo.iloc[-1]["Elo"] - (bayes_elo.iloc[0]["Elo"] - 1400)) * weight
    return total_loss
