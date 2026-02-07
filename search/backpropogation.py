from abc import ABC, abstractmethod
from typing import List, Any
import torch
from search.nodes import DecisionNode, ChanceNode


class Backpropagator(ABC):
    @abstractmethod
    def backpropagate(
        self, search_path, leaf_value, leaf_to_play, min_max_stats, config
    ):
        """
        Backpropagates the leaf value up the search path to update node values.
        """
        pass


class AverageDiscountedReturnBackpropagator(Backpropagator):
    def backpropagate(
        self, search_path, leaf_value, leaf_to_play, min_max_stats, config
    ):
        n = len(search_path)
        if n == 0:
            return []

        # --- 1) Build per-player accumulator array acc[p] = Acc_p(i) for current i (starting from i = n-1) ---
        # Acc_p(i) definition: discounted return from node i for a node whose player is p:
        # Acc_p(i) = sum_{j=i+1..n-1} discount^{j-i-1} * sign(p, j) * reward_j
        #            + discount^{n-1-i} * sign(p, leaf) * leaf_value
        # Where sign(p, j) = +1 if acting_player_at_j (which is search_path[j-1].to_play) == p else -1.
        #
        # We compute Acc_p(n-1) = sign(p, leaf) * leaf_value as base, then iterate backward:
        # Acc_p(i-1) = s(p, i) * reward_i + discount * Acc_p(i)

        # Initialize acc for i = n-1 (base: discounted exponent 0 for leaf value)
        # acc is a Python list of floats length num_players
        acc = [0.0] * config.game.num_players
        for p in range(config.game.num_players):
            acc[p] = leaf_value if leaf_to_play == p else -leaf_value

        # totals[i] will hold Acc_{node_player}(i)
        totals = [0.0] * n
        # Iterate from i = n-1 down to 0
        for i in range(n - 1, -1, -1):
            node = search_path[i]
            node_player = node.to_play
            # totals for this node = acc[node_player] (current Acc_p(i))
            # print(totals[i])
            totals[i] = acc[node_player]

            node.value_sum += totals[i]
            node.visits += 1

            # Prepare acc for i-1 (if any)
            if i > 0:
                # reward at index i belongs to acting_player = search_path[i-1].to_play
                acting_player = search_path[i - 1].to_play
                if isinstance(search_path[i], DecisionNode):
                    r_i = search_path[i - 1].child_reward(search_path[i])

                    # # Update per-player accumulators in O(num_players)
                    # # Acc_p(i-1) = sign(p, i) * r_i + discount * Acc_p(i)
                    # # sign(p, i) = +1 if acting_player == p else -1
                    # # We overwrite acc[p] in-place to be Acc_p(i-1)
                    for p in range(config.game.num_players):
                        sign = 1.0 if acting_player == p else -1.0
                        acc[p] = sign * r_i + config.discount_factor * acc[p]
                elif isinstance(search_path[i], ChanceNode):
                    for p in range(config.game.num_players):
                        # sign = 1.0 if acting_player == p else -1.0
                        # acc[p] = sign * r_i + config.discount_factor * acc[p]
                        # chance nodes can be thought to have 0 reward, and no discounting (as its like the roll after the action, or another way of thinking of it is that only on decision nodes do we discount expected reward, a chance node is not a decision point)
                        acc[p] = acc[p]
                child_q = search_path[i - 1].get_child_q_from_parent(search_path[i])
                min_max_stats.update(child_q)
            else:
                min_max_stats.update(search_path[i].value())


class MinimaxBackpropagator(Backpropagator):
    """
    Alpha-Beta Backpropagation (Minimax Value).
    Updates node.value_sum such that node.value() returns the Minimax value.
    """

    def backpropagate(
        self, search_path, leaf_value, leaf_to_play, min_max_stats, config
    ):
        n = len(search_path)
        if n == 0:
            return

        # 1. Handle Leaf
        leaf_node = search_path[-1]
        leaf_node.visits += 1

        # Calculate value relative to the node's player
        val = leaf_value if leaf_node.to_play == leaf_to_play else -leaf_value

        # Force value() to return val: value_sum = val * visits
        leaf_node.value_sum = val * leaf_node.visits
        min_max_stats.update(val)

        # 2. Propagate Upwards
        for i in range(n - 2, -1, -1):
            node = search_path[i]
            node.visits += 1

            if isinstance(node, DecisionNode):
                # Maximize Q-value of children
                # Because we are processing bottom-up, children are already updated.
                best_val = -float("inf")

                for action_or_code, child in node.children.items():
                    # get_child_q_from_parent uses child.value(), which is now the minimax value
                    q_val = node.get_child_q_from_parent(child)
                    if q_val > best_val:
                        best_val = q_val

                # If no children (shouldn't happen in backprop path), keep current val
                if len(node.children) == 0:
                    best_val = node.value()

                node.value_sum = best_val * node.visits

            elif isinstance(node, ChanceNode):
                # Expectimax: Weighted sum of children Q-values
                val_sum = 0.0
                total_prob = 0.0

                for code, child in node.children.items():
                    p = float(node.code_probs[code])
                    q_val = node.get_child_q_from_parent(child)

                    val_sum += p * q_val
                    total_prob += p

                avg_val = val_sum / total_prob if total_prob > 0 else 0.0
                node.value_sum = avg_val * node.visits

            # Update global stats with the new minimax value
            min_max_stats.update(node.value())
