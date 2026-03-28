from abc import ABC, abstractmethod
from typing import List, Any
import torch
from search.search_py.nodes import DecisionNode, ChanceNode


class Backpropagator(ABC):
    @abstractmethod
    def backpropagate(
        self,
        search_path,
        action_path,
        leaf_value,
        leaf_to_play,
        min_max_stats,
        config,
    ):
        """
        Backpropagates the leaf value up the search path to update node values.
        """
        pass  # pragma: no cover


class AverageDiscountedReturnBackpropagator(Backpropagator):
    def backpropagate(
        self,
        search_path,
        action_path,
        leaf_value,
        leaf_to_play,
        min_max_stats,
        config,
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
                parent = search_path[i - 1]
                # reward at index i belongs to acting_player = search_path[i-1].to_play
                acting_player = parent.to_play

                # Retrieve action used to reach node
                action = action_path[i - 1]

                # Wait, parent is expanding to node.
                # If parent is ChanceNode, node is DecisionNode.
                # If parent is DecisionNode, node is ChanceNode (or Decision if deterministic).

                if isinstance(node, DecisionNode):
                    r_i = parent.child_reward(node)

                    # Update per-player accumulators in O(num_players)
                    # Acc_p(i-1) = sign(p, i) * r_i + discount * Acc_p(i)
                    # sign(p, i) = +1 if acting_player == p else -1
                    # We overwrite acc[p] in-place to be Acc_p(i-1)
                    for p in range(config.game.num_players):
                        sign = 1.0 if acting_player == p else -1.0
                        acc[p] = sign * r_i + config.discount_factor * acc[p]
                elif isinstance(node, ChanceNode):
                    for p in range(config.game.num_players):
                        # chance nodes can be thought to have 0 reward, and no discounting (as its like the roll after the action, or another way of thinking of it is that only on decision nodes do we discount expected reward, a chance node is not a decision point)
                        acc[p] = acc[p]

                # --- VECTORIZED UPDATE ---
                # Update parent's tensor stats for the action taken
                # We need the return relative to PARENT's player.
                # acc now holds values for i-1 (parent).

                target_q = (
                    acc[acting_player] if hasattr(parent, "to_play") else acc[0]
                )  # ChanceNodes share parent to_play

                # Correct access for ChanceNode parent?
                # ChanceNode.to_play == parent.to_play.
                # So acc[acting_player] is correct.

                # Update visits
                parent.child_visits[action] += 1

                # Invalidate v_mix cache since children stats changed
                parent._v_mix = None

                # Incremental Mean Update
                # v_new = v_old + (target - v_old) / n
                current_val = parent.child_values[action]
                n_visits = parent.child_visits[action]

                # Note: target_q here is the Monte Carlo return G.
                # Does child_values store G or Q?
                # Q(s,a) = E[G]. So yes.

                parent.child_values[action] += (target_q - current_val) / n_visits
                min_max_stats.update(parent.child_values[action])

            else:
                min_max_stats.update(search_path[i].value())


class MinimaxBackpropagator(Backpropagator):
    """
    Alpha-Beta Backpropagation (Minimax Value).
    Updates node.value_sum such that node.value() returns the Minimax value.
    """

    def backpropagate(
        self,
        search_path,
        action_path,
        leaf_value,
        leaf_to_play,
        min_max_stats,
        config,
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

            # Need to update PARENT stats?
            # Iteration is: node is parent, we look at children.
            # search_path[i] is the node we are updating.
            # We determine node's value from its children.
            # Wait, Minimax logic computes node.value from node.children.

            # BUT we also need to update node's tensors!
            # node.child_values[action] needs to be updated.
            # The child responsible for the update is in the path?
            # No, Minimax scans ALL children.

            # Note: `DecisionNode` stores `child_values`.
            # If we update `node.children`'s values recursively,
            # do we need to reflect that in `node.child_values` tensor?
            # YES.

            # But iterating all children is slow!
            # However, Minimax is used for strict tree search (usually smaller).
            # If efficient updates are needed, we can't iterate all children.
            # But Minimax inherently looks at all children.

            # Optimization:
            # We only updated one child (the one in search_path[i+1]).
            # So `node.child_values[action_path[i]]` needs update.
            # The other children didn't change (in this simulation).

            # Get the child that was just updated
            child = search_path[i + 1]
            action = action_path[i]

            # Update tensor for that child
            # child.value() is the new minimax value of the child.
            q_val = node.get_child_q_from_parent(child)
            node.child_values[action] = q_val
            node.child_visits[action] += 1  # Is this right?
            # Minimax doesn't average, but visits track usage.

            # Invalidate v_mix cache
            node._v_mix = None

            # Now recompute node's value from TENSORS
            if isinstance(node, DecisionNode):
                # Maximize Q
                # Only iterate expanded (visited) children?
                # child_values contains valid Qs for visited children.
                # unvisited children have 0 or bootstrap.
                # Minimax usually assumes full expansion or heuristic.
                # Using visits mask
                mask = node.child_visits > 0
                if mask.any():
                    best_val = node.child_values[mask].max().item()
                else:
                    best_val = node.value()  # bootstrap

                node.value_sum = best_val * node.visits

            elif isinstance(node, ChanceNode):
                # Expectimax: sum(prob * val)
                # Use tensors
                vals = node.child_values
                probs = node.child_priors
                # Normalize probs? ChanceNode child_priors are probs.

                # If we assume we visited all children or use priors for unvisited?
                # Usually ChanceNode expands fully?
                # With lazy expansion, we might have unvisited.
                # Expected value over ALL outcomes.
                # Default unvisited to bootstrap or 0?
                # Let's assume child_values matches child.value() which handles bootstrap.
                # But child_values initialized to 0.
                # We need to compute expectation properly.

                # For now, minimal update:
                avg_val = (vals * probs).sum().item()
                # Check normalized probs?

                node.value_sum = avg_val * node.visits

            # Update global stats with the new minimax value
            min_max_stats.update(node.value())
