"""Functional search pipeline factory for the vectorized FlatTree MCTS.

Replaces the OOP ``create_mcts`` factory (which wired up ``SearchAlgorithm``
full of strategy-class instances) with ``build_search_pipeline``, which
selects pure functions at construction time and returns a single
``run_mcts`` callable that owns the complete search lifecycle.

Usage example::

    run_mcts = build_search_pipeline(config, device, num_actions)
    result = run_mcts(batched_obs, batched_info, agent_network)
"""

from __future__ import annotations

import functools
from typing import Callable, Dict, Optional, Any

import torch

from search.aos_search.tree import FlatTree
from search.aos_search.batched_mcts import batched_mcts_step
from search.aos_search.min_max_stats import VectorizedMinMaxStats
from search.aos_search.scoring import ScoringFn, ucb_score_fn, gumbel_score_fn
from search.aos_search.functional_modifiers import (
    apply_dirichlet_noise,
)
from search.aos_search.dynamic_masking import apply_sequential_halving
from search.aos_search.backpropogation import (
    BackpropFn,
    average_discounted_backprop,
    minimax_backprop,
)
from search.aos_search.search_output import (
    SearchOutput,
    visit_count_policy,
    gumbel_max_q_policy,
    minimax_policy,
)


# ---------------------------------------------------------------------------
# Registry maps (config string → pure function)
# ---------------------------------------------------------------------------

_BACKPROP_REGISTRY: Dict[str, BackpropFn] = {
    "average": average_discounted_backprop,
    "minimax": minimax_backprop,
}

_POLICY_REGISTRY: Dict[str, str] = {
    "visit_count": "visit_count",
    "gumbel": "gumbel",
    "minimax": "minimax",
    "best_action": "minimax",  # alias
    "completed_q": "gumbel",  # alias
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_valid_mask(
    info: Dict[str, Any], B: int, num_actions: int, device: torch.device
) -> Optional[torch.Tensor]:
    """Convert a legal-moves info dict to a ``[B, num_actions]`` bool mask.

    Returns ``None`` if no legal-move information is present (all actions
    valid).
    """
    legal_moves = info.get("legal_moves", None) if isinstance(info, dict) else None
    if legal_moves is None:
        return None

    # legal_moves may be a list-of-lists (one per batch item) or a flat list
    # (shared across the batch).
    if isinstance(legal_moves[0], int):
        # Flat list — replicate across batch
        legal_moves = [legal_moves] * B

    mask = torch.zeros(B, num_actions, dtype=torch.bool, device=device)
    for b, moves in enumerate(legal_moves):
        for a in moves:
            mask[b, a] = True
    return mask


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_search_pipeline(
    config,
    device: torch.device,
    num_actions: int,
) -> Callable:
    """Build a functional MCTS pipeline from the given config.

    Selects pure functions for:
      * **Modifier** — Dirichlet noise (``config.use_dirichlet``).
      * **Backprop** — via ``config.backprop_method`` (``"average"`` /
        ``"minimax"``; defaults to ``"average"``).
      * **Policy extraction** — via ``config.policy_extraction`` (``"visit_count"``,
        ``"gumbel" / "completed_q"``, ``"minimax" / "best_action"``; defaults
        to ``"visit_count"``).

    No OOP strategy class is instantiated.

    Args:
        config: Agent/search configuration object.  Expected attributes:

            * ``num_simulations`` — int
            * ``max_search_depth`` — int (tree depth per simulation)
            * ``max_nodes`` — int (FlatTree pre-allocation budget)
            * ``pb_c_init`` — float
            * ``pb_c_base`` — float
            * ``discount_factor`` — float (γ)
            * ``use_dirichlet`` — bool
            * ``dirichlet_alpha`` — float
            * ``dirichlet_fraction`` — float
            * ``backprop_method`` — str (optional, default ``"average"``)
            * ``policy_extraction`` — str (optional, default ``"visit_count"``)
            * ``policy_temperature`` — float (optional, default ``1.0``)
            * ``gumbel_cvisit`` — float (needed for gumbel policy)
            * ``gumbel_cscale`` — float (needed for gumbel policy)
            * ``num_codes`` — int (stochastic MuZero; default ``1``)

        device: Torch device to allocate tensors on.
        num_actions: Number of discrete actions (``max_edges`` in the tree).

    Returns:
        A callable ``run_mcts(batched_obs, batched_info, agent_network)
        -> SearchOutput`` that handles the full MCTS lifecycle.
    """
    # ---- Select backprop function ------------------------------------------
    backprop_key = config.backprop_method.lower()
    assert backprop_key in _BACKPROP_REGISTRY, (
        f"Unknown backprop_method '{backprop_key}'. "
        f"Valid options: {list(_BACKPROP_REGISTRY)}"
    )
    backprop_fn: BackpropFn = _BACKPROP_REGISTRY[backprop_key]

    # ---- Select policy extraction function ---------------------------------
    policy_key = _POLICY_REGISTRY.get(config.policy_extraction.lower(), "visit_count")

    # ---- Freeze config-driven hyperparameters into the pipeline ------------
    num_simulations: int = config.num_simulations
    max_depth: int = config.max_search_depth
    max_nodes: int = config.max_nodes
    num_codes: int = config.num_codes
    pb_c_init: float = config.pb_c_init
    pb_c_base: float = config.pb_c_base
    discount: float = config.discount_factor

    # Dirichlet args
    use_dirichlet: bool = config.use_dirichlet
    dirichlet_alpha: float = config.dirichlet_alpha
    dirichlet_fraction: float = config.dirichlet_fraction

    # Gumbel args
    gumbel_cvisit: float = config.gumbel_cvisit
    gumbel_cscale: float = config.gumbel_cscale

    # Sequential Halving
    use_sequential_halving: bool = config.use_sequential_halving
    gumbel_m: int = config.gumbel_m

    # ---- Select scoring function -------------------------------------------
    scoring_key: str = config.scoring_method.lower()
    ucb_kwargs: dict = {"pb_c_init": pb_c_init, "pb_c_base": pb_c_base}
    if scoring_key == "gumbel":
        # Gumbel at the root, UCB in the interior (standard Gumbel MuZero setup)
        root_scoring_fn: ScoringFn = gumbel_score_fn
        root_scoring_kwargs: dict = {
            "gumbel_cvisit": gumbel_cvisit,
            "gumbel_cscale": gumbel_cscale,
        }
        interior_scoring_fn: ScoringFn = ucb_score_fn
        interior_scoring_kwargs: dict = ucb_kwargs
    else:
        # Default: UCB everywhere
        root_scoring_fn = ucb_score_fn
        root_scoring_kwargs = ucb_kwargs
        interior_scoring_fn = ucb_score_fn
        interior_scoring_kwargs = ucb_kwargs

    # Value prefix
    use_value_prefix: bool = config.use_value_prefix

    # ---- Build the closure -------------------------------------------------
    def run_mcts(
        batched_obs: Any,
        batched_info: Dict[str, Any],
        agent_network,
    ) -> SearchOutput:
        """Execute a full vectorized MCTS search and return the root policy.

        Args:
            batched_obs: Batched observations fed to
                ``agent_network.obs_inference``.
            batched_info: Dict containing optional ``"legal_moves"`` for
                action masking.
            agent_network: Network with ``obs_inference`` and
                ``hidden_state_inference`` methods.

        Returns:
            :class:`~search.aos_search.search_output.SearchOutput` namedtuple
            with ``target_policy [B, A]``, ``exploratory_policy [B, A]``,
            and ``best_actions [B]``.
        """
        # ------------------------------------------------------------------
        # 1. Initial inference
        # ------------------------------------------------------------------
        with torch.inference_mode():
            outputs = agent_network.obs_inference(batched_obs)

        # outputs.value : [B]
        # outputs.policy.logits : [B, num_actions]
        B: int = outputs.value.shape[0]

        root_logits = outputs.policy.logits  # [B, num_actions]

        # ------------------------------------------------------------------
        # 2. Apply functional modifiers to root logits
        # ------------------------------------------------------------------
        # 2a. Dirichlet exploration noise (modifies the logits themselves)
        if use_dirichlet:
            root_logits = apply_dirichlet_noise(
                root_logits, dirichlet_alpha, dirichlet_fraction
            )

        # ------------------------------------------------------------------
        # 3. Allocate FlatTree and write root node
        # ------------------------------------------------------------------
        tree = FlatTree.allocate(
            batch_size=B,
            max_nodes=max_nodes,
            num_actions=num_actions,
            num_codes=num_codes,
            device=device,
        )

        # Root is node 0.  Write initial network value and PRISTINE policy priors.
        root_v = outputs.value.float()  # [B]
        tree.node_values[:, 0] = root_v
        # Root raw network value — never overwritten by backprop
        tree.raw_network_values[:, 0] = root_v
        tree.node_visits[:, 0] = 1  # count the initial "visit"

        # children_prior_logits[:, 0, :num_actions] = pristine (noised) logits
        tree.children_prior_logits[:, 0, :num_actions] = root_logits.float()

        # 3b. Mask invalid actions via the boolean mask (logits stay pristine)
        valid_mask = _build_valid_mask(batched_info, B, num_actions, device)
        if valid_mask is not None:
            tree.children_action_mask[:, 0, :num_actions] = valid_mask

        # 3c. Sample Gumbel noise for this search (stored for policy extraction).
        # Use PyTorch's native C++ sampler strictly to prevent -inf / NaN blowups.
        if policy_key == "gumbel":
            gumbel_dist = torch.distributions.Gumbel(0.0, 1.0)
            gumbel_noise_sample = gumbel_dist.sample((B, num_actions)).to(device)
        else:
            gumbel_noise_sample = None

        # ------------------------------------------------------------------
        # 4. Initialise global min-max stats
        # ------------------------------------------------------------------
        known_bounds = config.known_bounds
        min_max_stats = VectorizedMinMaxStats.allocate(
            B, device, known_bounds=known_bounds
        )

        # ------------------------------------------------------------------
        # 5. Main simulation loop
        # ------------------------------------------------------------------
        with torch.inference_mode():
            for sim_idx in range(num_simulations):
                # --- Dynamic pruning (Sequential Halving) ---
                if use_sequential_halving:
                    apply_sequential_halving(
                        tree=tree,
                        current_sim_idx=sim_idx,
                        total_simulations=num_simulations,
                        base_m=gumbel_m,
                        gumbel_cvisit=gumbel_cvisit,
                        gumbel_cscale=gumbel_cscale,
                        gumbel_noise=gumbel_noise_sample,
                        min_max_stats=min_max_stats,
                        num_actions=num_actions,
                    )

                batched_mcts_step(
                    tree=tree,
                    agent_network=agent_network,
                    max_depth=max_depth,
                    pb_c_init=pb_c_init,
                    pb_c_base=pb_c_base,
                    discount=discount,
                    backprop_fn=backprop_fn,
                    min_max_stats=min_max_stats,
                    root_scoring_fn=root_scoring_fn,
                    root_scoring_kwargs=root_scoring_kwargs,
                    interior_scoring_fn=interior_scoring_fn,
                    interior_scoring_kwargs=interior_scoring_kwargs,
                    use_value_prefix=use_value_prefix,
                )
                # Update global bounds using the root's child Q-values
                root_q = tree.children_values[:, 0, :num_actions].float()
                root_visited = tree.children_visits[:, 0, :num_actions] > 0
                min_max_stats.update(root_q, valid_mask=root_visited)

        # ------------------------------------------------------------------
        # 6. Extract final policies from root node (Temp is handled downstream by decorators)
        # ------------------------------------------------------------------
        if policy_key == "gumbel":
            policies = gumbel_max_q_policy(
                tree,
                gumbel_cvisit=gumbel_cvisit,
                gumbel_cscale=gumbel_cscale,
                gumbel_noise=gumbel_noise_sample,
                min_max_stats=min_max_stats,
                num_actions=num_actions,
            )
        elif policy_key == "minimax":
            policies = minimax_policy(tree, num_actions=num_actions)
        else:
            # Default: visit_count
            policies = visit_count_policy(tree, num_actions=num_actions)

        return policies

    return run_mcts
