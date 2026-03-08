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
    search_batch_size: int = 1,
    virtual_loss_visits: float = 1.0,
    virtual_loss_value: float = -1.0,
    penalty_type: str = "virtual_mean",
    backprop_fn: BackpropFn = average_discounted_backprop,
    min_max_stats: VectorizedMinMaxStats | None = None,
    root_scoring_fn: ScoringFn | None = None,
    root_scoring_kwargs: dict | None = None,
    interior_scoring_fn: ScoringFn | None = None,
    interior_scoring_kwargs: dict | None = None,
    decision_modifier_fn=None,
    chance_modifier_fn=None,
    trajectory_actions: torch.Tensor | None = None,
    num_players: int = 2,
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
        search_batch_size: Number of simulations to run in a single step via
            Virtual Loss.  Requires *agent_network* to support batched
            inference of size ``[B * search_batch_size]``.
        virtual_loss_visits: Visit penalty applied to nodes in the current path.
        virtual_loss_value: Value penalty applied to nodes in the current path.
        penalty_type: "virtual_mean" or "virtual_loss".
        backprop_fn: Pluggable single-depth backpropagation function.
        min_max_stats: Optional :class:`VectorizedMinMaxStats` global bounds.
        root_scoring_fn: Scoring function used at **depth 0** (the root).
        root_scoring_kwargs: Extra kwargs forwarded to ``root_scoring_fn``.
        interior_scoring_fn: Scoring function used at **depth > 0**.
        interior_scoring_kwargs: Extra kwargs forwarded to
            ``interior_scoring_fn``.
        decision_modifier_fn: Optional callable for Decision-node logits.
        chance_modifier_fn: Optional callable for Chance-node logits.
        trajectory_actions: Optional [B, H] forced actions for root.
    """
    B = tree.node_visits.shape[0]
    device = tree.node_visits.device

    # --- Setup scoring functions and kwargs ---
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

    if interior_scoring_fn is None:
        interior_scoring_fn = root_scoring_fn
        interior_scoring_kwargs = root_scoring_kwargs
    else:
        interior_scoring_kwargs = interior_scoring_kwargs or {}
        if "min_max_stats" not in interior_scoring_kwargs:
            interior_scoring_kwargs["min_max_stats"] = min_max_stats

    # ------------------------------------------------------------------
    # Phase 1 — Selection
    # ------------------------------------------------------------------
    paths = []

    for k in range(search_batch_size):
        path_nodes, path_actions, depths = _select(
            tree,
            B,
            max_depth,
            device,
            root_scoring_fn=root_scoring_fn,
            root_scoring_kwargs=root_scoring_kwargs,
            interior_scoring_fn=interior_scoring_fn,
            interior_scoring_kwargs=interior_scoring_kwargs,
            trajectory_actions=trajectory_actions if k == 0 else None,
        )
        paths.append((path_nodes, path_actions, depths))

        if search_batch_size > 1:
            _apply_virtual_penalty(
                tree,
                path_nodes,
                path_actions,
                depths,
                virtual_loss_visits,
                virtual_loss_value,
                penalty_type,
                B,
                device,
                sign=1.0,
            )

    # ------------------------------------------------------------------
    # Phase 2 — Batched Inference
    # ------------------------------------------------------------------
    batch_idx = torch.arange(B, device=device)
    all_parents = []
    all_actions = []

    for path_nodes, path_actions, depths in paths:
        safe_d = (depths - 1).clamp(min=0).long()
        all_parents.append(path_nodes[batch_idx, safe_d])
        all_actions.append(path_actions[batch_idx, safe_d])

    flat_parents_t = torch.cat(all_parents, dim=0)
    flat_actions_t = torch.cat(all_actions, dim=0)

    # Resolve flat_batch_idx for tree lookups
    flat_batch_idx = batch_idx.repeat(search_batch_size)

    # Determine node types (0 = Decision, 1 = Chance)
    parent_types = tree.node_types[flat_batch_idx, flat_parents_t.long()]
    is_decision = parent_types == 0
    is_chance = parent_types == 1

    B_total = flat_parents_t.shape[0]

    outputs_decision = None
    if is_decision.any():
        outputs_decision = agent_network.hidden_state_inference(
            flat_parents_t[is_decision], flat_actions_t[is_decision].long()
        )

    outputs_chance = None
    if is_chance.any():
        outputs_chance = agent_network.afterstate_inference(
            flat_parents_t[is_chance], flat_actions_t[is_chance].long()
        )

    if outputs_decision is not None:
        val_shape = outputs_decision.value.shape[1:]
        rew_shape = outputs_decision.reward.shape[1:]
        play_shape = outputs_decision.to_play.shape[1:]
        play_dtype = outputs_decision.to_play.dtype
    else:
        val_shape = outputs_chance.value.shape[1:]
        rew_shape = val_shape
        play_shape = val_shape
        play_dtype = torch.long

    # Determine max_edges from the tree structure to handle varying logit counts
    max_edges = tree.children_prior_logits.shape[-1]

    # Pre-allocate empty tensors
    leaf_values_t = torch.zeros(
        (B_total, *val_shape), dtype=torch.float32, device=device
    )
    rewards_t = torch.zeros((B_total, *rew_shape), dtype=torch.float32, device=device)
    to_plays_t = torch.zeros((B_total, *play_shape), dtype=play_dtype, device=device)

    # Pre-allocate policy logits with -inf padding to handle Decision vs Chance sizing
    policy_logits_t = torch.full(
        (B_total, max_edges),
        fill_value=-float("inf"),
        dtype=torch.float32,
        device=device,
    )

    # Scatter decision node results
    if outputs_decision is not None:
        leaf_values_t[is_decision] = outputs_decision.value
        rewards_t[is_decision] = outputs_decision.reward
        to_plays_t[is_decision] = outputs_decision.to_play

        num_act = outputs_decision.policy.logits.shape[-1]
        policy_logits_t[is_decision, :num_act] = outputs_decision.policy.logits

    # Scatter chance node results
    if outputs_chance is not None:
        leaf_values_t[is_chance] = outputs_chance.value
        # .reward is inherently 0 due to torch.zeros

        num_codes = outputs_chance.policy.logits.shape[-1]
        policy_logits_t[is_chance, :num_codes] = outputs_chance.policy.logits

        # Inherit parent's to_play from the tree
        inherited_to_play = tree.to_play[
            flat_batch_idx[is_chance], flat_parents_t[is_chance].long()
        ]

        if len(play_shape) > 0 and inherited_to_play.dim() == 1:
            inherited_to_play = inherited_to_play.unsqueeze(-1)

        to_plays_t[is_chance] = inherited_to_play.to(play_dtype)

    leaf_values_chunks = leaf_values_t.chunk(search_batch_size, dim=0)
    rewards_chunks = rewards_t.chunk(search_batch_size, dim=0)
    policy_logits_chunks = policy_logits_t.chunk(search_batch_size, dim=0)
    to_plays_chunks = to_plays_t.chunk(search_batch_size, dim=0)

    # ------------------------------------------------------------------
    # Phase 3 — Backpropagation
    # ------------------------------------------------------------------
    # Phase 3A: Global Reversion — clean the tree of all virtual distortions
    if search_batch_size > 1:
        for k in range(search_batch_size):
            path_nodes, path_actions, depths = paths[k]
            _apply_virtual_penalty(
                tree,
                path_nodes,
                path_actions,
                depths,
                virtual_loss_visits,
                virtual_loss_value,
                penalty_type,
                B,
                device,
                sign=-1.0,
            )

    # Phase 3B: Expansion & Backpropagation — commit numeric updates to clean tree
    for k in range(search_batch_size):
        path_nodes, path_actions, depths = paths[k]

        leaf_values = _expand_write(
            tree,
            path_nodes,
            path_actions,
            depths,
            leaf_values_chunks[k].detach().float(),
            rewards_chunks[k].detach().float(),
            policy_logits_chunks[k].detach(),
            to_plays_chunks[k].detach().to(torch.int8),
            B,
            device,
            decision_modifier_fn=decision_modifier_fn,
            chance_modifier_fn=chance_modifier_fn,
        )

        _backpropagate(
            tree,
            path_nodes,
            path_actions,
            depths,
            leaf_values,
            to_plays_chunks[k].detach().to(torch.long),
            discount,
            B,
            device,
            backprop_fn=backprop_fn,
            min_max_stats=min_max_stats,
            num_players=num_players,
        )


# ---------------------------------------------------------------------------
# Phase 1 — Selection
# ---------------------------------------------------------------------------


def _apply_virtual_penalty(
    tree: FlatTree,
    path_nodes: torch.Tensor,
    path_actions: torch.Tensor,
    depths: torch.Tensor,
    virtual_loss_visits: float,
    virtual_loss_value: float,
    penalty_type: str,
    B: int,
    device: torch.device,
    sign: float = 1.0,
):
    batch_idx = torch.arange(B, device=device)
    max_depth = path_actions.shape[1]

    vl_visits = torch.tensor(
        virtual_loss_visits * sign, dtype=torch.float32, device=device
    )
    val_tensor = torch.tensor(virtual_loss_value, dtype=torch.float32, device=device)

    for d in range(max_depth):
        valid = depths > d
        if not valid.any():
            continue

        nodes_at_d = path_nodes[batch_idx, d].long()
        actions_at_d = path_actions[batch_idx, d].long()

        # Read old visits
        old_node_v = tree.node_visits[batch_idx, nodes_at_d].float()
        old_child_v = tree.children_visits[batch_idx, nodes_at_d, actions_at_d].float()

        # Calculate new visits
        new_node_v = old_node_v + torch.where(valid, vl_visits, 0.0)
        new_child_v = old_child_v + torch.where(valid, vl_visits, 0.0)

        if penalty_type == "virtual_loss":
            # Read old Q-values
            old_node_q = tree.node_values[batch_idx, nodes_at_d]
            old_child_q = tree.children_values[batch_idx, nodes_at_d, actions_at_d]

            if sign > 0:
                # Applying penalty
                new_node_q = (
                    old_node_q * old_node_v + val_tensor * vl_visits
                ) / new_node_v.clamp(min=1e-8)
                new_child_q = (
                    old_child_q * old_child_v + val_tensor * vl_visits
                ) / new_child_v.clamp(min=1e-8)
            else:
                # Reversing penalty
                new_node_q = (
                    old_node_q * old_node_v - val_tensor * (-vl_visits)
                ) / new_node_v.clamp(min=1e-8)
                new_child_q = (
                    old_child_q * old_child_v - val_tensor * (-vl_visits)
                ) / new_child_v.clamp(min=1e-8)

                # Prevent drift
                new_node_q = torch.where(new_node_v <= 0, 0.0, new_node_q)
                new_child_q = torch.where(new_child_v <= 0, 0.0, new_child_q)

            tree.node_values[batch_idx, nodes_at_d] = torch.where(
                valid, new_node_q, old_node_q
            )
            tree.children_values[batch_idx, nodes_at_d, actions_at_d] = torch.where(
                valid, new_child_q, old_child_q
            )

        # Write visits back
        tree.node_visits[batch_idx, nodes_at_d] = torch.where(
            valid, new_node_v.to(torch.int32), old_node_v.to(torch.int32)
        )
        tree.children_visits[batch_idx, nodes_at_d, actions_at_d] = torch.where(
            valid, new_child_v.to(torch.int32), old_child_v.to(torch.int32)
        )


def _select(
    tree: FlatTree,
    B: int,
    max_depth: int,
    device: torch.device,
    root_scoring_fn: ScoringFn = ucb_score_fn,
    root_scoring_kwargs: dict | None = None,
    interior_scoring_fn: ScoringFn | None = None,
    interior_scoring_kwargs: dict | None = None,
    trajectory_actions: torch.Tensor | None = None,
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

        chance_mask = tree.children_action_mask[batch_idx, current_nodes.long()]
        masked_chance_logits = torch.where(
            chance_mask, child_logits, torch.full_like(child_logits, -float("inf"))
        )
        priors = torch.softmax(masked_chance_logits, dim=-1)  # [B, max_edges]

        chance_scores = priors / (child_visits_f + 1.0)  # [B, max_edges]

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

        if depth == 0 and trajectory_actions is not None:
            traj_mask = trajectory_actions >= 0
            actions = torch.where(
                traj_mask, trajectory_actions.to(torch.int32), actions
            )

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


def _expand_write(
    tree: FlatTree,
    path_nodes: torch.Tensor,
    path_actions: torch.Tensor,
    depths: torch.Tensor,
    leaf_values: torch.Tensor,
    rewards: torch.Tensor,
    policy_logits: torch.Tensor,
    to_plays: torch.Tensor,
    B: int,
    device: torch.device,
    decision_modifier_fn=None,
    chance_modifier_fn=None,
) -> torch.Tensor:
    """Expand leaf nodes while preventing overwrites if the edge was already expanded.

    Allocates new node slots in the FlatTree ONLY if they don't already exist.
    Updates visit counts and incremental value means for the surviving node.
    """
    batch_idx = torch.arange(B, device=device)
    safe_depth = (depths - 1).clamp(min=0).long()

    parent_indices = path_nodes[batch_idx, safe_depth]
    leaf_actions = path_actions[batch_idx, safe_depth]

    # --- 1. Detect if edge was already expanded by an earlier simulation ---
    existing_nodes = tree.children_index[
        batch_idx, parent_indices.long(), leaf_actions.long()
    ]

    # We only allocate if the edge is UNEXPANDED and the path is valid
    needs_allocation = (existing_nodes == UNEXPANDED_SENTINEL) & (depths > 0)

    # --- 2. Conditionally allocate ---
    # Using tree.next_alloc_index for new entries, or reusing existing_nodes
    new_node_indices = torch.where(
        needs_allocation, tree.next_alloc_index, existing_nodes
    )
    # Increment global allocation counter only by the number of actually new nodes
    tree.next_alloc_index += needs_allocation.to(torch.int32)
    new_idx_long = new_node_indices.long()

    # --- 3. Accumulate Visits and Values for the true surviving node ---
    valid = depths > 0
    tree.node_visits[batch_idx, new_idx_long] += valid.to(torch.int32)

    # Maintain an incremental mean of network values for reused nodes
    old_v = tree.node_values[batch_idx, new_idx_long]
    # Use float for mean calculation; clamp visit count to avoid div-by-zero
    new_v_count = tree.node_visits[batch_idx, new_idx_long].float().clamp(min=1.0)
    new_v = old_v + valid.float() * (leaf_values - old_v) / new_v_count

    tree.node_values[batch_idx, new_idx_long] = new_v
    tree.raw_network_values[batch_idx, new_idx_long] = new_v

    # --- 4. Apply functional modifiers ---
    # Assuming standard Decision nodes for the new allocations
    if decision_modifier_fn is not None:
        policy_logits = decision_modifier_fn(policy_logits)

    # --- 5. Write structural data ONLY for newly allocated nodes ---
    # We use boolean indexing to efficiently update only the newly created nodes
    # without overwriting the priors of nodes that were already established.
    alloc_mask = needs_allocation
    if alloc_mask.any():
        alloc_idx = batch_idx[alloc_mask]
        alloc_nodes = new_idx_long[alloc_mask]
        alloc_parents = parent_indices.long()[alloc_mask]
        alloc_actions = leaf_actions.long()[alloc_mask]

        # to_plays handling
        if to_plays.dim() > 1:
            tree.to_play[alloc_idx, alloc_nodes] = to_plays[alloc_mask].squeeze(-1)
        else:
            tree.to_play[alloc_idx, alloc_nodes] = to_plays[alloc_mask]

        tree.node_types[alloc_idx, alloc_nodes] = 0

        # Link parent -> child
        tree.children_index[alloc_idx, alloc_parents, alloc_actions] = new_node_indices[
            alloc_mask
        ]

        # Rewards
        tree.children_rewards[alloc_idx, alloc_parents, alloc_actions] = rewards[
            alloc_mask
        ]
        parent_cumulative = tree.node_rewards[alloc_idx, alloc_parents]
        tree.node_rewards[alloc_idx, alloc_nodes] = (
            parent_cumulative + rewards[alloc_mask]
        )

        # Logits and Masks
        num_act = policy_logits.shape[-1]

        # Reset action mask row for safety
        tree.children_action_mask[alloc_idx, alloc_nodes] = False

        tree.children_prior_logits[alloc_idx, alloc_nodes, :num_act] = policy_logits[
            alloc_mask
        ]
        tree.children_action_mask[alloc_idx, alloc_nodes, :num_act] = policy_logits[
            alloc_mask
        ] > -float("inf")

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
    leaf_to_play: torch.Tensor,
    discount: float,
    B: int,
    device: torch.device,
    backprop_fn: BackpropFn = average_discounted_backprop,
    min_max_stats: VectorizedMinMaxStats | None = None,
    num_players: int = 2,
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
        min_max_stats: Optional :class:`VectorizedMinMaxStats`.  When provided,
            bounds are expanded after every depth step using the freshly
            updated edge Q-values, ensuring the scoring functions always see
            current bounds throughout the simulation.
    """
    max_depth = path_actions.shape[1]
    batch_idx = torch.arange(B, device=device)

    # Running value being propagated upward, starts at leaf value
    # [B, num_players]
    running_values = torch.zeros((B, num_players), dtype=torch.float32, device=device)
    for p in range(num_players):
        is_p = leaf_to_play.squeeze() == p
        running_values[:, p] = torch.where(is_p, leaf_values, -leaf_values)

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
        is_decision = tree.node_types[batch_idx, parent_long] == 0  # [B] bool
        effective_discount = torch.where(
            is_decision,
            torch.tensor(discount, device=device),
            torch.tensor(1.0, device=device),
        )
        action_long = actions_at_d.long()
        child_long = path_nodes[batch_idx, d + 1].long()
        acting_player = tree.to_play[batch_idx, parent_long].long()

        # Standard: children_rewards holds instantaneous edge rewards
        step_reward = tree.children_rewards[batch_idx, parent_long, action_long]  # [B]

        # [B, num_players]
        is_acting = torch.arange(num_players, device=device).unsqueeze(
            0
        ) == acting_player.unsqueeze(1)
        reward_sign = torch.where(is_acting, 1.0, -1.0)

        new_running_values = (
            reward_sign * step_reward.unsqueeze(1)
            + effective_discount.unsqueeze(1) * running_values
        )
        running_values = torch.where(
            valid.unsqueeze(1), new_running_values, running_values
        )
        target_q = running_values[batch_idx, acting_player]  # [B]

        # --- Delegate Q/V updates to the pluggable backprop function ---
        # The function returns the value to continue propagating upward.
        # Although backprop_fn updates tree nodes, finding value propagates through running_values
        updated_q = backprop_fn(
            tree,
            batch_idx,
            nodes_at_d,
            actions_at_d,
            target_q,
            discount,
            valid,
        )
        running_values[batch_idx, acting_player] = updated_q

        # --- Update global min-max bounds with the newly written Q-values ---
        # Read back what backprop_fn just wrote for this depth's edges so the
        # scoring functions always see up-to-date bounds.
        if min_max_stats is not None:
            fresh_q = tree.children_values[batch_idx, parent_long, action_long]  # [B]
            # Expand to [B, 1] so VectorizedMinMaxStats.update accepts [B, A]
            min_max_stats.update(fresh_q.unsqueeze(-1), valid_mask=valid.unsqueeze(-1))
