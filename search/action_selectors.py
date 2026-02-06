from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import torch
from search.nodes import ChanceNode, DecisionNode
from search.scoring_methods import PriorScoring, ScoringMethod
import torch.nn.functional as F


class SelectionStrategy(ABC):
    @abstractmethod
    def select_child(self, node, min_max_stats, pruned_actionset=None):
        pass


class TopScoreSelection(SelectionStrategy):
    """
    Selects the child with the highest score according to the scoring_method.
    Supports a tiebreak_scoring_method for cases where scores are identical.
    """

    def __init__(
        self,
        scoring_method: ScoringMethod,
        tiebreak_scoring_method: Optional[ScoringMethod] = None,
    ):
        self.scoring_method = scoring_method
        self.tiebreak_scoring_method = tiebreak_scoring_method

    def select_child(self, node, min_max_stats, pruned_actionset=None):
        assert isinstance(node, DecisionNode)
        assert node.expanded(), "node must be expanded to select a child"

        actions = list(node.children.keys())
        if pruned_actionset is not None:
            # assert node.parent is None
            actions = [a for a in pruned_actionset if a in node.children]

        if len(actions) == 1:
            return actions[0], node.children[actions[0]]

        assert len(actions) > 0
        # print(actions)

        # Compute Primary Scores
        scores_dict = self.scoring_method.get_scores(node, min_max_stats)

        # Filter scores for allowed actions
        relevant_scores = {
            a: (s.detach().item() if torch.is_tensor(s) else float(s))
            for a, s in scores_dict.items()
            if a in actions
        }

        # Find max score
        max_score = max(relevant_scores.values())

        # Identify ties
        tied_actions = [
            a for a, s in relevant_scores.items() if np.isclose(s, max_score)
        ]

        # Break ties
        if len(tied_actions) == 1:
            action = tied_actions[0]
        else:
            if self.tiebreak_scoring_method:
                # Use tiebreak scoring method to resolve ties
                # We calculate secondary scores only for the tied actions (or all if simpler)
                all_sec_scores = self.tiebreak_scoring_method.get_scores(
                    node, min_max_stats
                )
                sec_scores_filtered = {a: all_sec_scores[a] for a in tied_actions}

                max_sec = max(sec_scores_filtered.values())
                tied_actions = [
                    a for a, s in sec_scores_filtered.items() if np.isclose(s, max_sec)
                ]

            # Default random tiebreak
            action = np.random.choice(tied_actions)

        assert isinstance(action, int) or isinstance(action, np.integer)
        return action, node.children[action]


class SamplingSelection(SelectionStrategy):
    """
    Selects a child by sampling.
    For DecisionNodes: Samples based on scores from scoring_method (softmax or direct if prob).
    For ChanceNodes: Samples codes based on probabilities.
    """

    def __init__(
        self, scoring_method: Optional[ScoringMethod] = None, temperature: float = 1.0
    ):
        # TODO BETTER WAY OF ENFORCING THE SCORING METHOD IS A PROBABILITY, FOR EXAMPLE SAMPLING FROM IMPROVED GUMBEL POLICY
        assert isinstance(scoring_method, PriorScoring)
        self.scoring_method = scoring_method
        self.temperature = temperature

    def select_child(self, node, min_max_stats, pruned_actionset=None):
        if isinstance(node, DecisionNode):
            actions = list(node.children.keys())
            if pruned_actionset is not None:
                # assert node.parent is None
                actions = [a for a in pruned_actionset if a in node.children]

            assert len(actions) > 0

            # Use scores to determine probabilities
            scores_dict = self.scoring_method.get_scores(node, min_max_stats)
            scores = np.array([scores_dict[a] for a in actions])

            if self.temperature == 0:
                action_idx = np.argmax(scores)
            else:
                probs = scores / scores.sum()
                action_idx = np.random.choice(len(actions), p=probs)
            action = actions[action_idx]
            return action, node.children[action]

        elif isinstance(node, ChanceNode):
            # TODO MAKE THIS USE SCORING METHODS TOO
            codes = list(node.code_probs.keys())
            probs = list(node.code_probs.values())
            code = node._sample_code(codes, probs)
            selected_code = F.one_hot(torch.tensor(code), num_classes=len(codes))
            child_node = node.children[code]
            return selected_code, child_node
