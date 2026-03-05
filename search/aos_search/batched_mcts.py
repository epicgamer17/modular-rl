"""Fully vectorized batched MCTS operating on a :class:`FlatTree` SoA.

Replaces ``search_py/modular_search.py``'s ``_run_batched_simulations`` and
``_run_batched_vectorized_simulations`` with a single ``batched_mcts_step``
that performs Selection → Expansion → Backpropagation using only tensor
operations — **no per-batch Python loops**.
"""

from __future__ import annotations

import torch

from search.aos_search.tree import FlatTree
from search.aos_search.scoring import ScoringFn, ucb_score_fn
from search.aos_search.backpropogation import BackpropFn, average_discounted_backprop
from search.aos_search.min_max_stats import VectorizedMinMaxStats


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
UNEXPANDED_SENTINEL: int = -1
"""Value in ``children_index`` meaning "this edge has not been expanded"."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def batched_mcts_step(
    tree: FlatTree,
    agent_network,
    max_depth: int,
    pb_c_init: float,
    pb_c_base: float,
    discount: float,
    backprop_fn: BackpropFn = average_discounted_backprop,
    min_max_stats: VectorizedMinMaxStats | None = None,
    root_scoring_fn: ScoringFn | None = None,
    root_scoring_kwargs: dict | None = None,
    interior_scoring_fn: ScoringFn | None = None,
    interior_scoring_kwargs: dict | None = None,
    use_value_prefix: bool = False,
    decision_modifier_fn=None,
    chance_modifier_fn=None,
) -> None:
    """Execute one full MCTS simulation across every batch element.

    The function modifies *tree* **in-place**.

    Args:
        tree: Pre-allocated :class:`FlatTree` (mutated in-place).
        agent_network: Network exposing
            ``hidden_state_inference(parent_indices, actions)``.
            Must return an object with ``.value``, ``.reward``, and
            ``.policy`` (with a ``.logits`` attribute) fields.
        max_depth: Maximum tree depth for the selection phase.
        pb_c_init: PUCT additive exploration constant.
        pb_c_base: PUCT log-base exploration constant.
        discount: MDP discount factor γ.
        backprop_fn: Pluggable single-depth backpropagation function.  Must
            conform to the :data:`~search.aos_search.backpropogation.BackpropFn`
            signature.  Defaults to
            :func:`~search.aos_search.backpropogation.average_discounted_backprop`.
        min_max_stats: Optional :class:`VectorizedMinMaxStats` passed to
            scoring functions so Q-values are normalised through the global
            running bounds.  Pass ``None`` to use local normalisation.
        root_scoring_fn: Scoring function used at **depth 0** (the root).
            When ``None``, defaults to ``ucb_score_fn``.  Use
            ``gumbel_score_fn`` here for Gumbel MuZero.
        root_scoring_kwargs: Extra kwargs forwarded to ``root_scoring_fn``
            (e.g. ``gumbel_cvisit``, ``gumbel_cscale``).
        interior_scoring_fn: Scoring function used at **depth > 0**.
            When ``None``, inherits ``root_scoring_fn``.
        interior_scoring_kwargs: Extra kwargs forwarded to
            ``interior_scoring_fn``.  When ``None``, inherits
            ``root_scoring_kwargs``.
        use_value_prefix: If ``True``, the edge reward stored in
            ``children_rewards`` is treated as a cumulative (absolute) reward
            rather than an instantaneous one.  During backpropagation the
            step reward is computed as ``child_reward - parent_reward``.
        decision_modifier_fn: Optional callable ``(logits: [B, A]) -> [B, A]``
            applied to new Decision-node logits during expansion (e.g.
            Top-K masking, invalid-action masking).
        chance_modifier_fn: Optional callable ``(logits: [B, C]) -> [B, C]``
            applied to new Chance-node code logits during expansion.
    """
    B = tree.node_visits.shape[0]
    device = tree.node_visits.device

    # ------------------------------------------------------------------
    # Phase 1 — Selection
    # ------------------------------------------------------------------
    # Build default root scoring fn + kwargs
    ucb_defaults = {
        "pb_c_init": pb_c_init,
        "pb_c_base": pb_c_base,
        "min_max_stats": min_max_stats,
    }
    if root_scoring_fn is None:
        root_scoring_fn = ucb_score_fn
        root_scoring_kwargs = ucb_defaults
    else:
        root_scoring_kwargs = root_scoring_kwargs or {}
        if "min_max_stats" not in root_scoring_kwargs:
            root_scoring_kwargs["min_max_stats"] = min_max_stats

    # Interior scoring inherits root by default
    if interior_scoring_fn is None:
        interior_scoring_fn = root_scoring_fn
        interior_scoring_kwargs = root_scoring_kwargs
    else:
        interior_scoring_kwargs = interior_scoring_kwargs or {}
        if "min_max_stats" not in interior_scoring_kwargs:
            interior_scoring_kwargs["min_max_stats"] = min_max_stats

    path_nodes, path_actions, depths = _select(
        tree,
        B,
        max_depth,
        device,
        root_scoring_fn=root_scoring_fn,
        root_scoring_kwargs=root_scoring_kwargs,
        interior_scoring_fn=interior_scoring_fn,
        interior_scoring_kwargs=interior_scoring_kwargs,
    )
    print(f"AOS Sim Depths: {depths.tolist()}")

    # ------------------------------------------------------------------
    # Phase 2 — Expansion
    # ------------------------------------------------------------------
    leaf_values = _expand(
        tree,
        agent_network,
        path_nodes,
        path_actions,
        depths,
        B,
        device,
        decision_modifier_fn=decision_modifier_fn,
        chance_modifier_fn=chance_modifier_fn,
    )

    # ------------------------------------------------------------------
    # Phase 3 — Backpropagation
    # ------------------------------------------------------------------
    _backpropagate(
        tree,
        path_nodes,
        path_actions,
        depths,
        leaf_values,
        discount,
        B,
        device,
        backprop_fn=backprop_fn,
        use_value_prefix=use_value_prefix,
        min_max_stats=min_max_stats,
    )


# ---------------------------------------------------------------------------
# Phase 1 — Selection
# ---------------------------------------------------------------------------


def _select(
    tree: FlatTree,
    B: int,
    max_depth: int,
    device: torch.device,
    root_scoring_fn: ScoringFn = ucb_score_fn,
    root_scoring_kwargs: dict | None = None,
    interior_scoring_fn: ScoringFn | None = None,
    interior_scoring_kwargs: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Descend the tree using UCB scores until an unexpanded edge is hit.

    Returns:
        path_nodes:   ``int32 [B, max_depth + 1]`` — visited node indices
                      (column 0 is always the root = 0).
        path_actions: ``int32 [B, max_depth]``     — actions taken at each
                      depth step.
        depths:       ``int32 [B]``                — actual depth reached per
                      batch element (number of *actions* taken).
    """
    # path_nodes[:, 0] = root (index 0 for every batch element)
    # [B, max_depth + 1]
    path_nodes = torch.zeros(
        (B, max_depth + 1),
        dtype=torch.int32,
        device=device,
    )
    # [B, max_depth]
    path_actions = torch.full(
        (B, max_depth),
        UNEXPANDED_SENTINEL,
        dtype=torch.int32,
        device=device,
    )
    # [B]
    depths = torch.zeros(B, dtype=torch.int32, device=device)

    # Current position of each batch element in the tree
    # [B]
    current_nodes = torch.zeros(B, dtype=torch.int32, device=device)
    # [B] — tracks which batch items are still actively descending
    is_active = torch.ones(B, dtype=torch.bool, device=device)

    batch_idx = torch.arange(B, device=device)

    # Resolve interior fn (defaults to root when not provided)
    _interior_fn = (
        interior_scoring_fn if interior_scoring_fn is not None else root_scoring_fn
    )
    _interior_kw = interior_scoring_kwargs or root_scoring_kwargs or {}
    _root_kw = root_scoring_kwargs or {}

    for depth in range(max_depth):
        # --- Select scoring function based on depth ---
        if depth == 0:
            active_scoring_fn = root_scoring_fn
            active_kw = _root_kw
        else:
            active_scoring_fn = _interior_fn
            active_kw = _interior_kw

        # --- Determine node types for active batch items ---
        # 0 = Decision, 1 = Chance
        current_types = tree.node_types[batch_idx, current_nodes.long()]  # [B] int8
        is_decision = current_types == 0  # [B] bool
        is_chance = current_types == 1  # [B] bool

        # --- Decision Node scoring (UCB / Gumbel) ---
        # [B, max_edges]
        decision_scores = active_scoring_fn(tree, current_nodes, **active_kw)
        # Micro-noise for uniform tie-breaking when scores are exactly equal
        # (e.g. all-unvisited flat-prior root). Scale is far below any real
        # score difference so it never changes the winner when scores differ.
        decision_scores = decision_scores + torch.rand_like(decision_scores) * 1e-6
        decision_actions = decision_scores.argmax(dim=-1).to(torch.int32)  # [B]

        # --- Chance Node quasi-sampling ---
        # Minimise the gap between prior probabilities and empirical visit
        # fractions: select argmax(prior - visits / parent_visits).
        # [B]
        parent_visits = tree.node_visits[batch_idx, current_nodes.long()].float()  # [B]
        # [B, max_edges]
        child_visits_f = tree.children_visits[batch_idx, current_nodes.long()].float()
        child_logits = tree.children_prior_logits[batch_idx, current_nodes.long()]
        priors = torch.softmax(child_logits, dim=-1)  # [B, max_edges]
        visit_fractions = child_visits_f / (
            parent_visits.unsqueeze(-1) + 1e-8
        )  # [B, max_edges]
        chance_scores = priors - visit_fractions  # [B, max_edges]

        # Mask out invalid edges for chance nodes (action mask)
        chance_mask = tree.children_action_mask[
            batch_idx, current_nodes.long()
        ]  # [B, max_edges]
        chance_scores = torch.where(
            chance_mask,
            chance_scores,
            torch.tensor(-float("inf"), dtype=chance_scores.dtype, device=device),
        )
        # Micro-noise for uniform tie-breaking on chance nodes
        chance_scores = chance_scores + torch.rand_like(chance_scores) * 1e-6
        chance_actions = chance_scores.argmax(dim=-1).to(torch.int32)  # [B]

        # --- Merge: Decision uses argmax(UCB), Chance uses argmax(gap) ---
        # [B]
        actions = torch.where(is_decision, decision_actions, chance_actions)

        # --- Look up children ---
        # children_index: [B, N, max_edges]
        # next_nodes: [B] int32
        next_nodes = tree.children_index[
            batch_idx, current_nodes.long(), actions.long()
        ]  # [B]

        # --- Determine which batch items hit an unexpanded edge ---
        # [B] bool
        hit_leaf = next_nodes == UNEXPANDED_SENTINEL

        # Batch items that were active but just hit a leaf
        newly_inactive = is_active & hit_leaf
        # Items that successfully descended
        is_active = is_active & ~hit_leaf

        # Record the action for items that took an action this step
        # (both those that hit a leaf AND those that descended further)
        took_action = newly_inactive | is_active
        path_actions[batch_idx, depth] = torch.where(
            took_action,
            actions,
            path_actions[batch_idx, depth],
        )

        # For items that successfully descended, update current node
        current_nodes = torch.where(is_active, next_nodes, current_nodes)
        path_nodes[batch_idx, depth + 1] = current_nodes

        # Update depths for items that took an action this step
        depths = torch.where(
            took_action,
            torch.tensor(depth + 1, dtype=torch.int32, device=device),
            depths,
        )

        if not is_active.any():
            break

    return path_nodes, path_actions, depths


# ---------------------------------------------------------------------------
# Phase 2 — Expansion
# ---------------------------------------------------------------------------


def _expand(
    tree: FlatTree,
    agent_network,
    path_nodes: torch.Tensor,
    path_actions: torch.Tensor,
    depths: torch.Tensor,
    B: int,
    device: torch.device,
    decision_modifier_fn=None,
    chance_modifier_fn=None,
) -> torch.Tensor:
    """Expand leaf nodes via a single batched network call.

    Allocates new node slots in the FlatTree, writes network outputs
    (value, reward, policy logits) into those slots, and links parents
    to the new children.

    Returns:
        leaf_values: ``float32 [B]`` — value estimates for the expanded
        leaves.
    """
    batch_idx = torch.arange(B, device=device)

    # --- Identify parent nodes and the actions that led to leaves ---
    # depth d means action indices 0..d-1 were taken; the expanding
    # action is at index d-1.
    # For depth==0 (root already a leaf—unlikely but handled gracefully)
    # we clamp to 0.
    safe_depth = (depths - 1).clamp(min=0).long()  # [B]

    parent_indices = path_nodes[batch_idx, safe_depth]  # [B] int32
    leaf_actions = path_actions[batch_idx, safe_depth]  # [B] int32

    # --- Batched network call ---
    # The caller stores hidden states in a [B, N, H] buffer alongside
    # the tree. We pass parent_indices + leaf_actions and rely on
    # agent_network to look up the correct hidden states.
    outputs = agent_network.hidden_state_inference(
        parent_indices,
        leaf_actions.long(),
    )

    # Expected output fields:
    #   .value  — [B] float32
    #   .reward — [B] float32
    #   .policy.logits — [B, num_actions] float32
    leaf_values = outputs.value.detach().float()  # [B]
    rewards = outputs.reward.detach().float()  # [B]
    policy_logits = outputs.policy.logits.detach()  # [B, num_actions]

    # --- Allocate new node indices ---
    # [B] int32
    new_node_indices = tree.next_alloc_index.clone()
    tree.next_alloc_index += 1

    # --- Write node-level stats for the new nodes ---
    new_idx_long = new_node_indices.long()
    tree.node_visits[batch_idx, new_idx_long] = 0
    tree.node_values[batch_idx, new_idx_long] = leaf_values
    # Permanently store the raw network value estimate (v̂_π).
    # node_values is updated by backpropagation; raw_network_values never is.
    tree.raw_network_values[batch_idx, new_idx_long] = leaf_values
    # 0 = Decision node
    tree.node_types[batch_idx, new_idx_long] = 0

    # --- Apply functional modifiers to logits BEFORE writing to tree ---
    # node_types currently written (line above): 0 = Decision for all.
    # If you later set some to 1 (Chance), split by node type here.
    node_type = tree.node_types[batch_idx, new_idx_long]  # [B] int8
    is_decision = node_type == 0  # [B]
    is_chance = node_type == 1  # [B]

    if decision_modifier_fn is not None and is_decision.any():
        # Apply modifier to all rows; rows that aren't Decision will be
        # overwritten below by chance_modifier_fn if applicable.
        policy_logits = decision_modifier_fn(policy_logits)

    if chance_modifier_fn is not None and is_chance.any():
        modified_chance = chance_modifier_fn(policy_logits)
        # Only overwrite rows that are actually Chance nodes
        chance_expanded = is_chance.unsqueeze(-1).expand_as(policy_logits)
        policy_logits = torch.where(chance_expanded, modified_chance, policy_logits)

    # --- Write edge-level stats (policy logits) for new nodes ---
    num_policy_actions = policy_logits.shape[-1]
    # children_prior_logits: [B, N, max_edges]
    tree.children_prior_logits[batch_idx, new_idx_long, :num_policy_actions] = (
        policy_logits
    )

    # --- Derive action mask from logits (-inf → masked out) ---
    tree.children_action_mask[batch_idx, new_idx_long, :num_policy_actions] = (
        policy_logits > -float("inf")
    )

    # --- Link parent → child via children_index ---
    tree.children_index[batch_idx, parent_indices.long(), leaf_actions.long()] = (
        new_node_indices
    )

    # --- Store edge reward from parent to new child ---
    tree.children_rewards[batch_idx, parent_indices.long(), leaf_actions.long()] = (
        rewards
    )

    # --- Store cumulative reward in node_rewards ---
    # node_rewards[child] = node_rewards[parent] + edge_reward
    # This allows value-prefix differential: step_r = child_r - parent_r
    parent_cumulative = tree.node_rewards[batch_idx, parent_indices.long()]  # [B]
    tree.node_rewards[batch_idx, new_idx_long] = parent_cumulative + rewards

    return leaf_values


# ---------------------------------------------------------------------------
# Phase 3 — Backpropagation
# ---------------------------------------------------------------------------


def _backpropagate(
    tree: FlatTree,
    path_nodes: torch.Tensor,
    path_actions: torch.Tensor,
    depths: torch.Tensor,
    leaf_values: torch.Tensor,
    discount: float,
    B: int,
    device: torch.device,
    backprop_fn: BackpropFn = average_discounted_backprop,
    use_value_prefix: bool = False,
    min_max_stats: VectorizedMinMaxStats | None = None,
) -> None:
    """Walk backwards along recorded paths, updating visit counts and values.

    Uses a boolean validity mask so that every batch element is handled by
    the same tensor operations regardless of its actual depth — **no
    per-batch Python loops**.  The mathematical update at each depth is
    fully delegated to ``backprop_fn``.

    Args:
        tree: FlatTree to update in-place.
        path_nodes: ``[B, max_depth + 1]`` node indices along each path.
        path_actions: ``[B, max_depth]`` actions taken at each depth.
        depths: ``[B]`` actual depth of each path.
        leaf_values: ``[B]`` value estimates at the expanded leaves.
        discount: MDP discount factor γ.
        B: Batch size.
        device: Torch device.
        backprop_fn: Pluggable single-depth update function conforming to
            :data:`~search.aos_search.backpropogation.BackpropFn`.
        use_value_prefix: If ``True``, ``children_rewards`` holds cumulative
            (absolute) rewards.  The step reward is computed as
            ``child_reward - parent_reward`` where parent_reward comes from
            ``node_rewards``.
        min_max_stats: Optional :class:`VectorizedMinMaxStats`.  When provided,
            bounds are expanded after every depth step using the freshly
            updated edge Q-values, ensuring the scoring functions always see
            current bounds throughout the simulation.
    """
    max_depth = path_actions.shape[1]
    batch_idx = torch.arange(B, device=device)

    # Running value being propagated upward, starts at leaf value
    # [B]
    running_value = leaf_values.clone()

    # Iterate from the deepest possible depth back to root
    for d in range(max_depth - 1, -1, -1):
        # --- validity mask: only update paths that reached this depth ---
        # depths > d means action d was taken
        # [B] bool
        valid = depths > d

        if not valid.any():
            continue

        # --- Parent node and action at depth d ---
        nodes_at_d = path_nodes[batch_idx, d]  # [B] int32
        actions_at_d = path_actions[batch_idx, d]  # [B] int32

        # --- Incorporate edge reward into the discounted return ---
        parent_long = nodes_at_d.long()
        action_long = actions_at_d.long()

        if use_value_prefix:
            # Value prefix: reward is cumulative → step_reward = child_r - parent_r
            # child is one depth deeper in path_nodes
            child_node = path_nodes[batch_idx, d + 1]  # [B] int32
            child_cumulative = tree.node_rewards[batch_idx, child_node.long()]  # [B]
            parent_cumulative = tree.node_rewards[batch_idx, parent_long]  # [B]
            step_reward = child_cumulative - parent_cumulative  # [B]
        else:
            # Standard: children_rewards holds instantaneous edge rewards
            step_reward = tree.children_rewards[
                batch_idx, parent_long, action_long
            ]  # [B]

        discounted_return = step_reward + discount * running_value  # [B]

        # --- Delegate Q/V updates to the pluggable backprop function ---
        # The function returns the value to continue propagating upward.
        running_value = backprop_fn(
            tree,
            batch_idx,
            nodes_at_d,
            actions_at_d,
            discounted_return,
            discount,
            valid,
        )

        # --- Update global min-max bounds with the newly written Q-values ---
        # Read back what backprop_fn just wrote for this depth's edges so the
        # scoring functions always see up-to-date bounds.
        if min_max_stats is not None:
            fresh_q = tree.children_values[batch_idx, parent_long, action_long]  # [B]
            # Expand to [B, 1] so VectorizedMinMaxStats.update accepts [B, A]
            min_max_stats.update(fresh_q.unsqueeze(-1), valid_mask=valid.unsqueeze(-1))
