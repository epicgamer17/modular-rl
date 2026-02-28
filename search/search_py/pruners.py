from abc import ABC, abstractmethod
import math
from typing import Any, List, Optional, Tuple, Dict
from search.min_max_stats import MinMaxStats


class PruningMethod(ABC):
    @abstractmethod
    def initialize(self, root: Any, config: Any) -> Any:
        """
        Initializes the pruning state at the beginning of a search.
        Returns the initial state object.
        """
        pass

    @abstractmethod
    def step(
        self,
        root: Any,
        state: Any,
        config: Any,
        min_max_stats: MinMaxStats,
        current_sim_idx: int,
    ) -> Tuple[Optional[List[int]], Any]:
        """
        Called before each simulation step.
        Returns (allowed_actions, next_state).
        If allowed_actions is None, all actions are allowed.
        """
        pass

    @property
    def mask_target_policy(self) -> bool:
        return False


class NoPruning(PruningMethod):
    """
    Does not prune any children; allows all actions at every step.
    """

    def initialize(self, root: Any, config: Any) -> Any:
        return None

    def step(
        self,
        root: Any,
        state: Any,
        config: Any,
        min_max_stats: MinMaxStats,
        current_sim_idx: int,
    ) -> Tuple[Optional[List[int]], Any]:
        return None, None


class SequentialHalvingPruning(PruningMethod):
    """
    Sequential Halving Pruning.
    Eliminates half of the candidates after specific budget rounds.
    Only allows the 'surviving' candidates to be selected during simulations.
    """

    def initialize(self, node: Any, config: Any) -> Any:
        candidates = list(node.children.keys())
        m = len(candidates)
        # We need to track survivors, current phase budget, etc.
        # State: {survivors, m_initial, sims_left_in_round}

        # Calculate first round budget
        sims_this_round = self._calc_sims_this_round(
            m, len(candidates), config.num_simulations, 0
        )

        return {
            "survivors": candidates,
            "m_initial": m,
            "sims_left_in_round": sims_this_round,
        }

    def step(
        self,
        node: Any,
        state: Any,
        config: Any,
        min_max_stats: MinMaxStats,
        current_sim_idx: int,
    ) -> Tuple[Optional[List[int]], Any]:

        survivors = state["survivors"]
        m_initial = state["m_initial"]
        sims_left = state["sims_left_in_round"]

        # If budget for the previous round is exhausted, we prune
        if sims_left <= 0:
            # Phase complete, eliminate
            survivors = self._eliminate(node, survivors, config, min_max_stats)

            # Calculate next budget
            sims_used_so_far = (
                current_sim_idx  # we are about to run this simulation index
            )
            sims_this_round = self._calc_sims_this_round(
                m_initial, len(survivors), config.num_simulations, sims_used_so_far
            )
            sims_left = sims_this_round

        # Return survivors for this step and update budget state
        # (decrement happens effectively by consuming this step)
        next_state = {
            "survivors": survivors,
            "m_initial": m_initial,
            "sims_left_in_round": sims_left - 1,
        }
        # print(survivors)
        return survivors, next_state

    def _calc_sims_this_round(self, m_initial, n_survivors, total_budget, sims_used):
        if n_survivors <= 1:
            return total_budget - sims_used  # Dump remaining if any

        if n_survivors > 2:
            # Original formula:
            # max(1, floor(total / (log2(m) * n_survivors))) * n_survivors
            sims_per_survivor = max(
                1, math.floor(total_budget / (math.log2(m_initial) * n_survivors))
            )
            sims_this_round = sims_per_survivor * n_survivors
        else:
            sims_this_round = total_budget - sims_used

        # Cap at total budget
        if sims_used + sims_this_round > total_budget:
            sims_this_round = total_budget - sims_used

        return int(sims_this_round)

    def _eliminate(self, node, survivors, config, min_max_stats):
        # Sort and pick top half
        def sort_by_score(action):
            child = node.children[action]
            max_visits = 0
            if node.children:
                max_visits = max(ch.visits for ch in node.children.values())

            q_value = min_max_stats.normalize(node.get_child_q_from_parent(child))

            # Replicate sigma logic
            # sigma = (cvisit + max_visits) * cscale * q
            sigma = (config.gumbel_cvisit + max_visits) * config.gumbel_cscale * q_value

            # score = log(prior) + sigma
            # Ensure prior > 0
            prior = child.prior if child.prior > 0 else 1e-12
            return math.log(prior) + sigma

        survivors_sorted = sorted(survivors, key=sort_by_score, reverse=True)

        num_to_eliminate = math.ceil(len(survivors) / 2.0)

        # Ensure we keep at least 2 if possible (from original logic)
        # "if len(survivors) - num_to_eliminate < 2: survivors = survivors[:2]"
        if len(survivors) - num_to_eliminate < 2:
            if len(survivors) > 2:
                new_survivors = survivors_sorted[:2]
            else:
                new_survivors = survivors_sorted[:2]  # Keep top 2
        else:
            new_survivors = survivors_sorted[:-num_to_eliminate]

        return new_survivors

    @property
    def mask_target_policy(self) -> bool:
        return True


class AlphaBetaPruning(PruningMethod):
    """
    Alpha-Beta Pruning.
    Maintains alpha/beta bounds for each node.
    Prunes branches where the current value estimate exceeds beta.
    Assumes Negamax value structure (values flip sign between layers).
    """

    def initialize(self, root: Any, config: Any) -> Any:
        # Initial bounds: (-inf, +inf)
        return {"alpha": -float("inf"), "beta": float("inf")}

    def step(
        self,
        node: Any,
        state: Any,
        config: Any,
        min_max_stats: MinMaxStats,
        current_sim_idx: int,
    ) -> Tuple[Optional[List[int]], Any]:

        alpha = state["alpha"]
        beta = state["beta"]

        # Update alpha based on current children values
        # We only look at expanded/visited children to tighten the bound
        # (Heuristic: assumes current value estimates are reliable enough to prune)
        current_best = -float("inf")
        if node.children:
            for action, child in node.children.items():
                if child.expanded() or child.visits > 0:
                    # Negamax: value to us is get_child_q_from_parent
                    q = node.get_child_q_from_parent(child)
                    if q > current_best:
                        current_best = q

        if current_best > alpha:
            alpha = current_best

        # Pruning Check
        if alpha >= beta:
            # Beta Cutoff: This node is too good for the opponent to allow us to reach here
            # (or we found a move that refutes the parent's expectation)
            # Returning empty list means ALL actions are pruned.
            return [], state

        # Prepare next state for children (Negamax flip)
        # next_alpha = -beta, next_beta = -alpha
        next_state = {"alpha": -beta, "beta": -alpha}

        # Return None to indicate "all actions allowed" (unless pruned)
        return None, next_state

    @property
    def mask_target_policy(self) -> bool:
        return False
