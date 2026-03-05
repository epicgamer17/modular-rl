import torch
from search.search_py.scoring_methods import UCBScoring

original_get_scores = UCBScoring.get_scores


def patched_get_scores(self, node, min_max_stats):
    scores = original_get_scores(self, node, min_max_stats)
    print(f"PY Scores at visits {node.visit_count}: {scores.tolist()}")
    return scores


UCBScoring.get_scores = patched_get_scores

from search.aos_search.scoring import ucb_score_fn

original_ucb_score_fn = ucb_score_fn


def patched_ucb_score_fn(tree, node_indices, pb_c_init, pb_c_base, min_max_stats):
    scores = original_ucb_score_fn(
        tree, node_indices, pb_c_init, pb_c_base, min_max_stats
    )
    print(f"AOS Scores: {scores.tolist()}")
    return scores


import search.aos_search.batched_mcts

search.aos_search.batched_mcts.ucb_score_fn = patched_ucb_score_fn

from tests.test_search_parity_standalone import test_full_search_ucb_parity, get_config

config = get_config()
test_full_search_ucb_parity(config)
