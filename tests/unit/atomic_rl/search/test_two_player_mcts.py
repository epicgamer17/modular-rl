"""Component-level checks for a 2-player alternating zero-sum MCTS.

The convention used throughout (and by the mp AlphaZero example) is:

  * The player to move is NOT stored in the tree; it is carried in the state
    `embeddings` (the environment supplies it).
  * Alternation + zero-sum is encoded in the edge `discount` on each transition:
      - ``discount = -1`` for a non-terminal transition (flips perspective),
      - ``discount = 0``   for a terminal transition (plateaus at reward).
  * ``children_values[parent, a]`` stores the value of the *child* node, so that
    ``Q(s, a) = reward + discount * children_values`` (mctx semantics).

These tests are written to isolate which component misbehaves for a 2-player game
rather than to reproduce any particular failure.
"""
import torch
import pytest

from atomic_rl.search import (
    backpropagate_,
    expand_node_,
    get_mcts_visit_policy,
    init_mcts_tree,
    mcts_search,
    puct_score,
    qtransform_by_parent_and_siblings,
    select_leaf,
)
from atomic_rl.search.qtransforms import get_qvalues
from atomic_rl.envs.functions.tictactoe import tictactoe_dynamics_fn

pytestmark = pytest.mark.unit


# ============================================================================
# Helpers
# ============================================================================
def _check_edge_child_value_invariant(tree):
    """mctx invariant: every edge must store its child node's value."""
    children_index = tree["children_index"]
    children_values = tree["children_values"]
    node_values = tree["node_values"]
    edge_exists = children_index >= 0
    assert edge_exists.any(), "tree has no expanded edges to check"
    b_idx, n_idx, a_idx = edge_exists.nonzero(as_tuple=True)
    child_idx = children_index[b_idx, n_idx, a_idx]
    child_values = node_values[b_idx, child_idx]
    edge_values = children_values[b_idx, n_idx, a_idx]
    assert torch.allclose(child_values, edge_values, atol=1e-5), (
        "children_values[parent, a] != node_values[child] for some edge. "
        "The backward pass is not storing child node values (mctx semantics)."
    )


# ============================================================================
# 1. init_mcts_tree
# ============================================================================
def test_init_tree_root_fields():
    """The root node must be populated with priors/value and neutral edges."""
    B, A, S = 2, 4, 10
    root_logits = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    tree = init_mcts_tree(
        torch.zeros(B, 4),
        root_logits=root_logits,
        root_value=torch.tensor([0.4, -0.6]),
        num_simulations=S,
        num_actions=A,
    )

    assert tree["node_visits"].shape == (B, S + 1)
    assert tree["children_index"].shape == (B, S + 1, A)
    assert tree["node_counts"].tolist() == [1, 1]  # root allocated only
    assert torch.equal(
        tree["children_index"][:, 1:], torch.full((B, S, A), -1)
    )  # only root occupies slot 0

    # Root priors stored exactly (logits; softmax happens at selection time)
    torch.testing.assert_close(tree["children_prior_logits"][:, 0], root_logits)
    # Root value/visits
    torch.testing.assert_close(tree["node_values"][:, 0], torch.tensor([0.4, -0.6]))
    torch.testing.assert_close(tree["raw_values"][:, 0], torch.tensor([0.4, -0.6]))
    assert tree["node_visits"][:, 0].tolist() == [0, 0]
    # Neutral edge stats: discounts default to 1.0, everything else 0/-1
    assert torch.all(tree["children_discounts"][:, 0] == 1.0)
    assert torch.all(tree["children_visits"][:, 0] == 0)
    assert torch.all(tree["children_values"][:, 0] == 0.0)


# ============================================================================
# 2. expand_node_
# ============================================================================
def test_expand_node_single_transition_two_player():
    """A single expansion must link parent->child and store model outputs."""
    B, A, S = 1, 3, 6
    tree = init_mcts_tree(
        torch.zeros(B, 2),
        root_logits=torch.zeros(B, A),
        root_value=torch.zeros(B),
        num_simulations=S,
        num_actions=A,
    )
    next_emb = torch.tensor([[1.0, -1.0]])

    expand_node_(
        tree=tree,
        parent_nodes=torch.tensor([0]),
        actions_taken=torch.tensor([1]),
        policy_logits=torch.tensor([[2.0, 3.0, 0.0]]),
        value=torch.tensor([-0.7]),
        rewards=torch.tensor([0.5]),
        discounts=torch.tensor([-1.0]),
        next_embeddings=next_emb,
    )

    # Structural edges are recorded both directions
    assert tree["children_index"][0, 0, 1].item() == 1
    assert tree["parents"][0, 1].item() == 0
    assert tree["action_from_parent"][0, 1].item() == 1
    # Edge statistics
    assert tree["children_visits"][0, 0, 1].item() == 0
    assert tree["children_values"][0, 0, 1].item() == 0.0
    assert tree["children_rewards"][0, 0, 1].item() == 0.5
    assert tree["children_discounts"][0, 0, 1].item() == -1.0
    # New node stats (visit starts at 1, like mctx update_tree_node)
    assert tree["node_visits"][0, 1].item() == 1
    assert tree["node_values"][0, 1].item() == pytest.approx(-0.7)
    assert tree["raw_values"][0, 1].item() == pytest.approx(-0.7)
    assert tree["is_terminal"][0, 1].item() is False
    torch.testing.assert_close(tree["embeddings"][0, 1], next_emb[0])
    torch.testing.assert_close(
        tree["children_prior_logits"][0, 1], torch.tensor([2.0, 3.0, 0.0])
    )
    assert tree["node_counts"][0].item() == 2


def test_expand_node_terminal_sets_flag_and_zeroes_prior_masking():
    """discount==0 marks the node terminal; legal_mask zeroes illegal priors."""
    B, A, S = 1, 3, 6
    tree = init_mcts_tree(
        torch.zeros(B, 2),
        root_logits=torch.zeros(B, A),
        root_value=torch.zeros(B),
        num_simulations=S,
        num_actions=A,
    )

    expand_node_(
        tree=tree,
        parent_nodes=torch.tensor([0]),
        actions_taken=torch.tensor([0]),
        policy_logits=torch.tensor([[2.0, 3.0, 1.0]]),
        value=torch.tensor([0.0]),
        rewards=torch.tensor([-1.0]),  # player to move lost
        discounts=torch.tensor([0.0]),  # terminal
        next_embeddings=torch.zeros(1, 2),
        legal_mask=torch.tensor([[True, True, False]]),
    )

    assert tree["is_terminal"][0, 1].item() is True
    assert tree["children_rewards"][0, 0, 0].item() == -1.0
    assert tree["children_discounts"][0, 0, 0].item() == 0.0
    # Illegal action 2 prior is floored, legal ones untouched
    merged = tree["children_prior_logits"][0, 1]
    assert merged[0].item() == 2.0
    assert merged[1].item() == 3.0
    assert merged[2].item() == torch.finfo(torch.float32).min


# ============================================================================
# 3. backpropagate_
# ============================================================================
def test_backprop_alternating_chain_negamax():
    """Hand-computed 2-ply alternating backup with discount = -1 / 0.

    Path: root(0) -[a0, d=-1, r=0]-> node(1) -[a1, d=0, r=+1]-> leaf(2).
    Leaf value v2 = 0.4 (should be irrelevant: terminal edge plates at reward).
    Expected (mctx semantics):
        node_values[1] = mean(-0.8, 1.0) = 0.10
        children_values[0, a0] = node_values[1] = 0.10
        Q(0, a0) = 0 + (-1.0) * 0.10 = -0.10  (player-0 perspective)
        node_values[0] = raw return propagated up = 0 + (-1.0)*1.0 = -1.0
    """
    B, A, S = 1, 2, 8
    tree = init_mcts_tree(
        torch.zeros(B, 2),
        root_logits=torch.zeros(B, A),
        root_value=torch.zeros(B),
        num_simulations=S,
        num_actions=A,
    )

    # Manually build the pre-search tree (nodes were "expanded" earlier)
    tree["children_index"][0, 0, 0] = 1  # root -> node1 via a0
    tree["children_rewards"][0, 0, 0] = 0.0
    tree["children_discounts"][0, 0, 0] = -1.0

    tree["children_index"][0, 1, 1] = 2  # node1 -> node2 via a1
    tree["children_rewards"][0, 1, 1] = 1.0
    tree["children_discounts"][0, 1, 1] = 0.0  # terminal edge

    tree["node_visits"][0, 1] = 1
    tree["node_values"][0, 1] = -0.8  # player-to-move-at-1 prediction
    tree["node_visits"][0, 2] = 1
    tree["node_values"][0, 2] = 0.4  # terminal node value (unused on d=0 edge)

    trajectory = [
        (torch.tensor([0]), torch.tensor([0]), torch.tensor([True])),
        (torch.tensor([1]), torch.tensor([1]), torch.tensor([True])),
    ]
    backpropagate_(tree, trajectory, leaf_value=torch.tensor([0.4]))

    # Edge values must equal the child node values
    assert tree["children_values"][0, 1, 1].item() == pytest.approx(0.4)
    assert tree["children_values"][0, 0, 0].item() == pytest.approx(0.10)
    # Node 1's own averaged value (mean of -0.8 and terminal return 1.0)
    assert tree["node_values"][0, 1].item() == pytest.approx(0.10)
    # Root absorbs the RAW propagated return, like mctx: -1.0 * 1.0 = -1.0.
    # (mctx does not substitute the child's *averaged* node value into the
    # return stream, so node_values below roots can diverge from Q.)
    assert tree["node_values"][0, 0].item() == pytest.approx(-1.0)
    # Q(root, a0) uses the stored child value: 0 + (-1.0) * 0.10 = -0.10
    q = get_qvalues(tree, torch.tensor([0]))
    assert q[0, 0].item() == pytest.approx(-0.10)
    # Visit accounting: parents +1, leaf untouched
    assert tree["node_visits"][0, 0].item() == 1
    assert tree["node_visits"][0, 1].item() == 2
    assert tree["node_visits"][0, 2].item() == 1
    assert tree["children_visits"][0, 0, 0].item() == 1
    assert tree["children_visits"][0, 1, 1].item() == 1


def test_backprop_does_not_double_count_leaf():
    """A freshly expanded leaf must keep node_visits == 1 after one backup."""
    B, A, S = 1, 2, 8
    tree = init_mcts_tree(
        torch.zeros(B, 2),
        root_logits=torch.zeros(B, A),
        root_value=torch.zeros(B),
        num_simulations=S,
        num_actions=A,
    )
    # Simulate one expand_node_ on action 0 of the root
    expand_node_(
        tree=tree,
        parent_nodes=torch.tensor([0]),
        actions_taken=torch.tensor([0]),
        policy_logits=torch.zeros(1, A),
        value=torch.tensor([0.6]),
        rewards=torch.tensor([0.0]),
        discounts=torch.tensor([-1.0]),
        next_embeddings=torch.zeros(1, 2),
    )
    trajectory = [(torch.tensor([0]), torch.tensor([0]), torch.tensor([True]))]
    backpropagate_(tree, trajectory, leaf_value=torch.tensor([0.6]))

    assert tree["node_visits"][0, 1].item() == 1  # mctx: leaf starts at 1, not 2
    assert tree["node_values"][0, 1].item() == pytest.approx(0.6)
    assert tree["node_visits"][0, 0].item() == 1


# ============================================================================
# 4. q-values / q-transform
# ============================================================================
def test_qvalues_alternating_sign_convention():
    """With discount=-1, Q must equal reward + (-1) * child_value (negamax)."""
    B, A, S = 1, 2, 8
    tree = init_mcts_tree(
        torch.zeros(B, 2),
        root_logits=torch.zeros(B, A),
        root_value=torch.tensor([0.5]),
        num_simulations=S,
        num_actions=A,
    )
    tree["node_values"][0, 0] = 0.5
    tree["node_visits"][0, 0] = 10
    # Two visited children with known child-perspective values
    tree["children_index"][0, 0, 0] = 1
    tree["children_index"][0, 0, 1] = 2
    tree["node_values"][0, 1] = 0.9
    tree["node_values"][0, 2] = 0.2
    tree["children_values"][0, 0, 0] = 0.9  # = node_values[child]
    tree["children_values"][0, 0, 1] = 0.2
    tree["children_discounts"][0, 0, 0] = -1.0
    tree["children_discounts"][0, 0, 1] = -1.0
    tree["children_visits"][0, 0, 0] = 4
    tree["children_visits"][0, 0, 1] = 4

    q = get_qvalues(tree, torch.tensor([0]))
    assert q[0, 0].item() == pytest.approx(-0.9)
    assert q[0, 1].item() == pytest.approx(-0.2)


def test_qtransform_parent_siblings_alternating():
    """Normalized Q for the above scenario: best action -> 0.5, rest -> 0.0."""
    B, A, S = 1, 3, 8
    tree = init_mcts_tree(
        torch.zeros(B, 2),
        root_logits=torch.zeros(B, A),
        root_value=torch.tensor([0.5]),
        num_simulations=S,
        num_actions=A,
    )
    tree["node_values"][0, 0] = 0.5
    tree["node_visits"][0, 0] = 10
    tree["children_index"][0, 0, 0] = 1
    tree["children_index"][0, 0, 1] = 2
    tree["node_values"][0, 1] = 0.9
    tree["node_values"][0, 2] = 0.2
    # Edges: action0 visited (Q=-0.9), action1 visited (Q=-0.2), action2 unvisited
    tree["children_values"][0, 0, 0] = 0.9
    tree["children_values"][0, 0, 1] = 0.2
    tree["children_discounts"][0, 0, 0] = -1.0
    tree["children_discounts"][0, 0, 1] = -1.0
    tree["children_visits"][0, 0, 0] = 4
    tree["children_visits"][0, 0, 1] = 4

    norm = qtransform_by_parent_and_siblings(tree, torch.tensor([0]))
    torch.testing.assert_close(
        norm[0], torch.tensor([0.0, 0.5, 0.0]), rtol=1e-6, atol=1e-6
    )


# ============================================================================
# 5. puct_score / select_leaf
# ============================================================================
def test_selection_prefers_correct_flipped_q():
    """With equal priors/visits, root must pick the action that MAXIMIZES Q
    after the perspective flip (i.e. minimises the opponent-perspective value)."""
    B, A, S = 1, 2, 8
    tree = init_mcts_tree(
        torch.zeros(B, 2),
        root_logits=torch.tensor([[0.0, 0.0]]),  # uniform prior
        root_value=torch.tensor([0.5]),
        num_simulations=S,
        num_actions=A,
    )
    tree["node_values"][0, 0] = 0.5
    tree["node_visits"][0, 0] = 10
    tree["children_index"][0, 0, 0] = 1
    tree["children_index"][0, 0, 1] = 2
    # Opponent-perspective child values: action0 is BAD for root (child 0.9),
    # action1 is GOOD for root (child 0.2) => root should choose action 1.
    tree["node_values"][0, 1] = 0.9
    tree["node_values"][0, 2] = 0.2
    tree["children_values"][0, 0, 0] = 0.9
    tree["children_values"][0, 0, 1] = 0.2
    tree["children_discounts"][0, 0, 0] = -1.0
    tree["children_discounts"][0, 0, 1] = -1.0
    tree["children_visits"][0, 0, 0] = 4
    tree["children_visits"][0, 0, 1] = 4

    leaf_parents, _, _, trajectory = select_leaf(
        tree, pb_c_base=19652.0, pb_c_init=1.25
    )

    # First decision at the root must be action 1, descending into node 2
    assert trajectory[0][1].item() == 1
    assert leaf_parents.item() == 2


def test_puct_score_obeys_root_legal_mask():
    """Illegal root actions must be floored so argmax can never select them."""
    B, A, S = 1, 3, 8
    tree = init_mcts_tree(
        torch.zeros(B, 2),
        root_logits=torch.zeros(B, A),
        root_value=torch.zeros(B),
        num_simulations=S,
        num_actions=A,
        legal_mask=torch.tensor([[True, False, True]]),
    )
    tree["node_visits"][0, 0] = 1

    scores = puct_score(tree, torch.tensor([0]), depth=0)
    assert scores[0, 1].item() == torch.finfo(torch.float32).min
    assert scores[0, 0].item() > -1e8
    assert scores[0, 2].item() > -1e8


# ============================================================================
# 6. get_mcts_visit_policy
# ============================================================================
def test_visit_policy_two_player_temperatures():
    visits = torch.tensor([[4, 2, 0]])
    torch.testing.assert_close(
        get_mcts_visit_policy(visits, temperature=1.0),
        torch.tensor([[4 / 6, 2 / 6, 0.0]]),
    )
    # temperature=0 is a one-hot over the max-visit action
    greedy = get_mcts_visit_policy(visits, temperature=0.0)
    assert greedy[0, 0].item() == 1.0
    assert greedy[0, 1].item() == 0.0


# ============================================================================
# 7. End-to-end mcts_search (non-terminal alternating game)
# ============================================================================
def test_search_alternating_game_preserves_edge_invariant():
    """Full search where every transition alternates (discount=-1, no reward).

    Every simulation must expand one fresh node (terminal-free), and the mctx
    edge invariant children_values == node_values[child] must hold everywhere.
    """
    B, A, S = 2, 3, 16

    def recurrent_fn(actions, embeddings):
        next_emb = embeddings + torch.ones_like(embeddings) * 0.01
        logits = embeddings.new_zeros(embeddings.shape[0], A)
        value = embeddings.new_full((embeddings.shape[0],), 0.3)
        reward = embeddings.new_zeros(embeddings.shape[0])
        discount = -embeddings.new_ones(embeddings.shape[0])  # always alternating
        return logits, value, reward, discount, next_emb

    _, action_probs, tree = mcts_search(
        root_embeddings=torch.zeros(B, 4),
        root_logits=torch.zeros(B, A),
        root_value=torch.zeros(B),
        recurrent_fn=recurrent_fn,
        num_simulations=S,
        num_actions=A,
        dirichlet_epsilon=0.0,
    )

    # Every simulation expanded a new node (no terminal states in this game)
    assert tree["node_counts"].tolist() == [S + 1, S + 1]
    # Root was visited once per simulation
    assert tree["node_visits"][:, 0].tolist() == [S, S]
    assert tree["children_visits"][:, 0].sum(dim=-1).tolist() == [S, S]
    # Policy is a normalized distribution over actions
    torch.testing.assert_close(action_probs.sum(dim=-1), torch.ones(B))
    assert torch.all(action_probs >= 0.0)
    # The backprop edge invariant holds for the whole tree
    _check_edge_child_value_invariant(tree)


def test_search_alternating_mp_style_tictactoe():
    """The mp example's exact pattern: tictactoe dynamics + discount -1/0.

    NOTE: child priors MUST be masked with the next state's legal mask inside
    the recurrent_fn (the mp example does this via ``apply_action_mask``). The
    search itself does not know the child's legal mask. Without it the search
    selects occupied cells and ``tictactoe_dynamics_fn`` raises FATAL ERROR.
    """
    B, A, S = 1, 9, 12
    board_embed = torch.zeros(B, 3, 3, 2)  # empty board, player 0
    legal_mask = (board_embed[0, ..., 0].view(-1) == 0).unsqueeze(0)
    min_logit = torch.finfo(torch.float32).min

    def recurrent_fn(actions_taken, embeddings):
        next_embed, reward, _, is_terminal, next_legal_mask = tictactoe_dynamics_fn(
            embeddings, actions_taken
        )
        logits = embeddings.new_zeros(embeddings.shape[0], A)
        logits = logits.masked_fill(~next_legal_mask, min_logit)
        value = embeddings.new_zeros(embeddings.shape[0])
        discount = torch.where(
            is_terminal, torch.zeros_like(reward), -torch.ones_like(reward)
        )
        return logits, value, reward, discount, next_embed

    search_action, action_probs, tree = mcts_search(
        root_embeddings=board_embed,
        root_logits=torch.zeros(B, A),
        root_value=torch.zeros(B),
        recurrent_fn=recurrent_fn,
        num_simulations=S,
        num_actions=A,
        legal_mask=legal_mask,
        dirichlet_epsilon=0.0,
    )

    assert not torch.isnan(tree["node_values"]).any()
    assert not torch.isnan(tree["children_values"]).any()
    assert tree["node_visits"][:, 0].sum(dim=-1).item() == S
    torch.testing.assert_close(action_probs.sum(dim=-1), torch.ones(B))
    _check_edge_child_value_invariant(tree)
    assert search_action.shape == (B,)
    # With a near-full board, no illegal (occupied) cell can ever receive visits
    occupied = torch.zeros(9, dtype=torch.bool)
    occupied[0] = True
    occupied[1] = True
    occupied[2] = True
    occupied[3] = True
    occupied[5] = True
    occupied[7] = True
    board_embed[0, ..., 0] = torch.tensor(
        [[1.0, -1.0, 1.0], [-1.0, 0.0, 1.0], [0.0, -1.0, 0.0]]
    )
    legal_mask = (~occupied).unsqueeze(0)

    _, _, tree2 = mcts_search(
        root_embeddings=board_embed,
        root_logits=torch.zeros(B, A),
        root_value=torch.zeros(B),
        recurrent_fn=recurrent_fn,
        num_simulations=12,
        num_actions=A,
        legal_mask=legal_mask,
        dirichlet_epsilon=0.0,
    )
    assert torch.all(tree2["children_visits"][:, 0, occupied] == 0)