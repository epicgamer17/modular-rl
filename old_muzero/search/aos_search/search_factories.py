"""Functional search pipeline factory for the vectorized FlatTree MCTS.

Replaces the OOP ``create_mcts`` factory (which wired up ``ModularSearch``
full of strategy-class instances) with ``build_search_pipeline``, which
selects pure functions at construction time and returns a single
``run_mcts`` callable that owns the complete search lifecycle.

Usage example::

    run_mcts = build_search_pipeline(config, device, num_actions)
    result = run_mcts(batched_obs, batched_info, agent_network)
"""

from __future__ import annotations

import functools
import math
from typing import Callable, Dict, Optional, Any

import torch
import torch.utils._pytree as pytree

from old_muzero.search.aos_search.tree import FlatTree
from old_muzero.search.aos_search.batched_mcts import batched_mcts_step
from old_muzero.search.aos_search.min_max_stats import VectorizedMinMaxStats
from old_muzero.search.aos_search.scoring import ScoringFn, ucb_score_fn, gumbel_score_fn
from old_muzero.search.aos_search.functional_modifiers import (
    apply_dirichlet_noise,
)
from old_muzero.search.aos_search.dynamic_masking import apply_sequential_halving
from old_muzero.search.aos_search.backpropogation import (
    BackpropFn,
    average_discounted_backprop,
    minimax_backprop,
)
from old_muzero.search.aos_search.search_output import (
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

    Accepts three input formats for ``info["legal_moves"]``:
    - ``[B, num_actions]`` bool tensor — returned directly (actors' format)
    - ``[num_actions]`` bool tensor — broadcast to ``[B, num_actions]``
    - list of ints (flat) — replicated across batch items
    - list of lists of ints (per-item) — mapped element-wise
    """
    legal_moves = info.get("legal_moves", None) if isinstance(info, dict) else None
    if legal_moves is None:
        return None

    # Fast path: actors pass pre-vectorized bool tensors — handle directly.
    if torch.is_tensor(legal_moves) and legal_moves.dtype == torch.bool:
        if legal_moves.dim() == 2:  # [B, num_actions] — already correct
            return legal_moves.to(device=device)
        elif legal_moves.dim() == 1:  # [num_actions] — broadcast across batch
            return legal_moves.unsqueeze(0).expand(B, -1).to(device=device)

    # Fallback: list-of-ints or list-of-lists (e.g. from parity tests / Python backend).
    if isinstance(legal_moves[0], int):
        # Flat list — replicate across batch
        legal_moves = [legal_moves] * B

    mask = torch.zeros(B, num_actions, dtype=torch.bool, device=device)
    for b, moves in enumerate(legal_moves):
        for a in moves:
            mask[b, a] = True
    return mask


class MCTSPipeline:
    """A pickleable MCTS pipeline that replaces the local ``run_mcts`` closure.

    This class encapsulates all the search hyperparameters and selected strategy
    functions (backprop, scoring, policy extraction) into a single callable
    object that can be serialized by ``torch.multiprocessing``.
    """

    def __init__(
        self,
        config: Any,
        device: torch.device,
        num_actions: int,
        backprop_fn: BackpropFn,
        policy_key: str,
        scoring_key: str,
        pb_c_init: float,
        pb_c_base: float,
        discount: float,
        num_simulations: int,
        max_depth: int,
        max_nodes: int,
        num_codes: int,
        use_dirichlet: bool,
        dirichlet_alpha: float,
        dirichlet_fraction: float,
        gumbel_cvisit: float,
        gumbel_cscale: float,
        search_batch_size: int,
        virtual_loss_visits: float,
        virtual_loss_value: float,
        penalty_type: str,
        bootstrap_method: str,
        num_players: int,
        use_sequential_halving: bool,
        gumbel_m: int,
    ):
        self.config = config
        self.device = device
        self.num_actions = num_actions
        self.backprop_fn = backprop_fn
        self.policy_key = policy_key
        self.scoring_key = scoring_key
        self.pb_c_init = pb_c_init
        self.pb_c_base = pb_c_base
        self.discount = discount
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.num_codes = num_codes
        self.use_dirichlet = use_dirichlet
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_fraction = dirichlet_fraction
        self.gumbel_cvisit = gumbel_cvisit
        self.gumbel_cscale = gumbel_cscale
        self.search_batch_size = search_batch_size
        self.virtual_loss_visits = virtual_loss_visits
        self.virtual_loss_value = virtual_loss_value
        self.penalty_type = penalty_type
        self.bootstrap_method = bootstrap_method
        self.num_players = num_players
        self.use_sequential_halving = use_sequential_halving
        self.gumbel_m = gumbel_m

    def __call__(
        self,
        batched_obs: Any,
        batched_info: Dict[str, Any],
        agent_network: Any,
        trajectory_actions: Optional[torch.Tensor] = None,
    ) -> SearchOutput:
        """Execute a full vectorized MCTS search and return the root policy.

        Args:
            batched_obs: Batched observations fed to
                ``agent_network.obs_inference``.
            batched_info: Dict containing optional ``\"legal_moves\"`` for
                action masking and strictly required ``\"player\"``.
            agent_network: Network with ``obs_inference`` and
                ``hidden_state_inference`` methods.
            trajectory_actions: Optional tensor of sequence actions (for reanalyze).

        Returns:
            :class:`~search.aos_search.search_output.SearchOutput` namedtuple
            with ``target_policy [B, A]``, ``exploratory_policy [B, A]``,
            and ``best_actions [B]``.
        """
        batched_to_play = batched_info["player"]
        if not torch.is_tensor(batched_to_play):
            batched_to_play = torch.tensor(
                batched_to_play, dtype=torch.int8, device=self.device
            )

        # ------------------------------------------------------------------
        # 1. Initial inference
        # ------------------------------------------------------------------
        with torch.inference_mode():
            outputs = agent_network.obs_inference(batched_obs)

        # outputs.value : [B]
        # outputs.policy.logits : [B, num_actions]
        B: int = outputs.value.shape[0]
        root_logits = outputs.policy.logits  # [B, num_actions]

        # Define a function to expand a state tensor to the full buffer shape
        def allocate_buffer(tensor):
            if tensor is None:
                return None
            shape = (B, self.max_nodes, *tensor.shape[1:])
            return torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)

        # Blindly map over whatever opaque structure the network returned!
        state_buffer = pytree.tree_map(allocate_buffer, outputs.network_state)

        # ------------------------------------------------------------------
        # 2. Functional modifiers (applied at the root)
        # ------------------------------------------------------------------
        # 2a. Build the valid mask FIRST so noise isn't wasted on illegal moves
        valid_mask = _build_valid_mask(batched_info, B, self.num_actions, self.device)

        # 2b. Dirichlet exploration noise (modifies the logits themselves)
        if self.use_dirichlet:
            # Apply valid mask to ensure Dirichlet noise doesn't leak to illegal actions
            root_logits = apply_dirichlet_noise(
                root_logits,
                self.dirichlet_alpha,
                self.dirichlet_fraction,
                valid_mask=valid_mask,
            )

        # ------------------------------------------------------------------
        # 3. Allocate FlatTree and write root node
        # ------------------------------------------------------------------
        tree = FlatTree.allocate(
            batch_size=B,
            max_nodes=self.max_nodes,
            num_actions=self.num_actions,
            num_codes=self.num_codes,
            device=self.device,
        )
        tree.network_state_buffer = state_buffer

        # Write root state (node 0) into the buffer
        def write_root(buffer_tensor, root_tensor):
            if buffer_tensor is not None:
                buffer_tensor[:, 0] = root_tensor

        pytree.tree_map(write_root, tree.network_state_buffer, outputs.network_state)

        # Root is node 0.  Write initial network value and PRISTINE policy priors.
        root_v = outputs.value.float()  # [B]
        tree.node_values[:, 0] = root_v
        # Root raw network value — never overwritten by backprop
        tree.raw_network_values[:, 0] = root_v
        tree.node_visits[:, 0] = 1  # count the initial \"visit\"

        # children_prior_logits[:, 0, :num_actions] = pristine (noised) logits
        tree.children_prior_logits[:, 0, : self.num_actions] = root_logits.float()

        # 3b. Mask invalid actions via the boolean mask (logits stay pristine)
        if valid_mask is not None:
            tree.children_action_mask[:, 0, : self.num_actions] = valid_mask
        else:
            # Fallback: All actions whose network prior is not -inf are valid
            tree.children_action_mask[:, 0, : self.num_actions] = root_logits > -float(
                "inf"
            )

        # Root player identity
        tree.to_play[:, 0] = batched_to_play.to(dtype=torch.int8, device=self.device)

        # ------------------------------------------------------------------
        # 3c. Setup Scoring Functions & Gumbel Noise
        # ------------------------------------------------------------------

        # 1. Sample noise if using Gumbel policy
        if self.policy_key == "gumbel":
            gumbel_dist = torch.distributions.Gumbel(0.0, 1.0)
            gumbel_noise_sample = gumbel_dist.sample((B, self.num_actions)).to(
                self.device
            )
        else:
            gumbel_noise_sample = None

        # 2. Build the base UCB kwargs
        ucb_kwargs: dict = {
            "pb_c_init": self.pb_c_init,
            "pb_c_base": self.pb_c_base,
            "bootstrap_method": self.bootstrap_method,
        }

        # 3. Assign the actual functions and kwargs dictionaries
        if self.scoring_key == "gumbel":
            root_scoring_fn: ScoringFn = gumbel_score_fn
            root_scoring_kwargs: dict = {
                "gumbel_cvisit": self.gumbel_cvisit,
                "gumbel_cscale": self.gumbel_cscale,
                "bootstrap_method": self.bootstrap_method,
                "gumbel_noise": gumbel_noise_sample,
            }
            interior_scoring_fn: ScoringFn = ucb_score_fn
            interior_scoring_kwargs: dict = ucb_kwargs
        else:
            root_scoring_fn = ucb_score_fn
            root_scoring_kwargs = ucb_kwargs.copy()
            # Even if UCB scoring, we might be tracking gumbel noise for policy extraction
            root_scoring_kwargs["gumbel_noise"] = gumbel_noise_sample

            interior_scoring_fn = ucb_score_fn
            interior_scoring_kwargs = ucb_kwargs.copy()

        # ------------------------------------------------------------------
        # 4. Initialise global min-max stats
        # ------------------------------------------------------------------
        known_bounds = self.config.known_bounds
        min_max_stats = VectorizedMinMaxStats.allocate(
            B,
            self.device,
            known_bounds=known_bounds,
            epsilon=self.config.min_max_epsilon,
        )

        # ------------------------------------------------------------------
        # 5. Main simulation loop
        # ------------------------------------------------------------------
        with torch.inference_mode():
            for sim_idx in range(
                math.ceil(self.num_simulations / self.search_batch_size)
            ):
                # --- Dynamic pruning (Sequential Halving) ---
                if self.use_sequential_halving:
                    apply_sequential_halving(
                        tree=tree,
                        current_sim_idx=sim_idx,
                        total_simulations=self.num_simulations,
                        base_m=self.gumbel_m,
                        gumbel_cvisit=self.gumbel_cvisit,
                        gumbel_cscale=self.gumbel_cscale,
                        gumbel_noise=gumbel_noise_sample,
                        min_max_stats=min_max_stats,
                        num_actions=self.num_actions,
                    )

                # Internal modifiers
                internal_decision_modifier = None
                if self.config.internal_decision_modifier != "none":
                    # Logic here to resolve string to function if needed
                    pass

                internal_chance_modifier = None
                if self.config.internal_chance_modifier != "none":
                    # Logic here to resolve string to function if needed
                    pass

                batched_mcts_step(
                    tree=tree,
                    agent_network=agent_network,
                    max_depth=self.max_depth,
                    pb_c_init=self.pb_c_init,
                    pb_c_base=self.pb_c_base,
                    discount=self.discount,
                    search_batch_size=self.search_batch_size,
                    virtual_loss_visits=self.virtual_loss_visits,
                    virtual_loss_value=self.virtual_loss_value,
                    penalty_type=self.penalty_type,
                    backprop_fn=self.backprop_fn,
                    min_max_stats=min_max_stats,
                    root_scoring_fn=root_scoring_fn,
                    root_scoring_kwargs=root_scoring_kwargs,
                    interior_scoring_fn=interior_scoring_fn,
                    interior_scoring_kwargs=interior_scoring_kwargs,
                    decision_modifier_fn=internal_decision_modifier,
                    chance_modifier_fn=internal_chance_modifier,
                    trajectory_actions=trajectory_actions,
                    num_players=self.num_players,
                )
                # Update global bounds using the root's child Q-values
                root_q = tree.children_values[:, 0, : self.num_actions].float()
                root_visited = tree.children_visits[:, 0, : self.num_actions] > 0
                min_max_stats.update(root_q, valid_mask=root_visited)

        # ------------------------------------------------------------------
        # 6. Extract final policies from root node (Temp is handled downstream by decorators)
        # ------------------------------------------------------------------
        if self.policy_key == "gumbel":
            policies = gumbel_max_q_policy(
                tree,
                gumbel_cvisit=self.gumbel_cvisit,
                gumbel_cscale=self.gumbel_cscale,
                gumbel_noise=gumbel_noise_sample,
                min_max_stats=min_max_stats,
                num_actions=self.num_actions,
            )
        elif self.policy_key == "minimax":
            policies = minimax_policy(tree, num_actions=self.num_actions)
        else:
            # Default: visit_count
            policies = visit_count_policy(tree, num_actions=self.num_actions)

        return policies


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_search_pipeline(
    config,
    device: torch.device,
    num_actions: int,
) -> MCTSPipeline:
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
            * ... (see implementation below for full list)

        device: Torch device to allocate tensors on.
        num_actions: Number of discrete actions (``max_edges`` in the tree).

    Returns:
        An :class:`MCTSPipeline` instance (callable) that handles the full MCTS lifecycle.
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

    # ---- Select scoring function -------------------------------------------
    scoring_key: str = config.scoring_method.lower()

    # ---- Hyperparameters ---------------------------------------------------
    search_batch_size: int = config.search_batch_size
    if search_batch_size == 0:
        search_batch_size = 1

    return MCTSPipeline(
        config=config,
        device=device,
        num_actions=num_actions,
        backprop_fn=backprop_fn,
        policy_key=policy_key,
        scoring_key=scoring_key,
        pb_c_init=config.pb_c_init,
        pb_c_base=config.pb_c_base,
        discount=config.discount_factor,
        num_simulations=config.num_simulations,
        max_depth=config.max_search_depth,
        max_nodes=config.max_nodes,
        num_codes=config.num_codes,
        use_dirichlet=config.use_dirichlet,
        dirichlet_alpha=config.dirichlet_alpha,
        dirichlet_fraction=config.dirichlet_fraction,
        gumbel_cvisit=config.gumbel_cvisit,
        gumbel_cscale=config.gumbel_cscale,
        search_batch_size=search_batch_size,
        virtual_loss_visits=1.0,
        virtual_loss_value=-config.virtual_loss,
        penalty_type="virtual_mean" if config.use_virtual_mean else "virtual_loss",
        bootstrap_method=config.bootstrap_method,
        num_players=config.game.num_players,
        use_sequential_halving=config.use_sequential_halving,
        gumbel_m=config.gumbel_m,
    )
