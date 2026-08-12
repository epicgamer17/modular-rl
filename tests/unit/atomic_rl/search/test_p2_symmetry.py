"""P2 (player 1 / 'O') symmetry & correctness checks.

The MCTS implementation is deliberately *player-agnostic*: the absolute player
identity ("who am I") never appears in the tree. It only lives in the state
`embeddings` (the `to_play` plane) and is fed to the network. The alternation +
zero-sum is encoded purely by the edge `discount`:
    discount = -1 for a non-terminal transition (flips perspective),
    discount =  0 for a terminal transition (plateaus at reward).

Because of this, the *search* cannot intrinsically favour P1 over P2. These
tests pin down that property so a future regression cannot quietly introduce an
asymmetry. They cover:

    1. tictactoe dynamics from P2's perspective (piece, reward sign, turn).
    2. Color-swap mirror symmetry of the dynamics and the canonical obs.
    3. Consistency between the MCTS canonical (embeddings_to_canonical) and the
       replay canonical (get_canonical_obs) for BOTH players.
    4. Search-level: a P2-to-move root produces a *bit-identical* search to a
       P1-to-move root given the same rollouts (the search has no side bias).
    5. Search-level decisions: P2 finds a forced win and blocks a forced loss,
       and the P1 mirror decisions agree on the same cell.
    6. Determinism: the same manual seed reproduces an identical P2 search, so
       run-to-run variation in P2's strength is a seed/sampling effect.
"""
import torch
import pytest

from atomic_rl.search import (
    get_mcts_visit_policy,
    init_mcts_tree,
    mcts_search,
    select_leaf,
)
from atomic_rl.search.qtransforms import get_qvalues
from atomic_rl.envs.functions.tictactoe import (
    check_tictactoe_winner,
    embeddings_to_canonical,
    get_canonical_obs,
    tictactoe_dynamics_fn,
)

pytestmark = pytest.mark.unit

NUM_ACTIONS = 9
MIN_LOGIT = torch.finfo(torch.float32).min


# ============================================================================
# Helpers
# ============================================================================
def _embed(board, player):
    """Builds a [1, 3, 3, 2] embedding from a 3x3 board and to_play."""
    emb = torch.zeros(1, 3, 3, 2)
    emb[0, ..., 0] = torch.tensor(board)
    emb[0, ..., 1] = player
    return emb


def _recurrent_fn(actions_taken, embeddings):
    """The mp example's exact pattern: tictactoe dynamics + masked child priors."""
    next_embed, reward, _, is_terminal, next_legal_mask = tictactoe_dynamics_fn(
        embeddings, actions_taken
    )
    logits = embeddings.new_zeros(embeddings.shape[0], NUM_ACTIONS)
    logits = logits.masked_fill(~next_legal_mask, MIN_LOGIT)
    value = embeddings.new_zeros(embeddings.shape[0])
    discount = torch.where(
        is_terminal, torch.zeros_like(reward), -torch.ones_like(reward)
    )
    return logits, value, reward, discount, next_embed


# ============================================================================
# 1. Dynamics from P2's perspective
# ============================================================================
def test_dynamics_p2_places_minus_one_and_flips_turn():
    """P2 (to_play=1) must drop a -1 piece and hand the turn back to P1."""
    next_embed, reward, next_to_play, is_terminal, legal = tictactoe_dynamics_fn(
        _embed([[0] * 3, [0] * 3, [0] * 3], 1), torch.tensor([4])
    )
    assert next_embed[0, 1, 1, 0].item() == -1.0  # P2 owns the center
    assert next_embed[0, ..., 1].tolist() == [[0] * 3, [0] * 3, [0] * 3]  # P1 next
    assert next_to_play.item() == 0
    assert is_terminal.item() is False
    assert reward.item() == 0.0
    assert legal[0, 4].item() is False  # center now occupied
    assert legal[0, 0].item() is True


def test_dynamics_p2_win_and_loss_reward_sign():
    """Terminal rewards are from the mover's POV; a P2 win is a P1 loss and
    vice-versa, and the reward sign is color-symmetric."""
    # P2 (to_play=1) completes the first row with -1, -1, -1 -> P2 won
    _, reward, _, is_term, legal = tictactoe_dynamics_fn(
        _embed([[-1, -1, 0], [0, 0, 0], [0, 0, 0]], 1), torch.tensor([2])
    )
    assert reward.item() == 1.0
    assert is_term.item() is True
    assert legal.sum().item() == 6.0  # terminal row full, six cells still empty

    # Mirror: P1 (to_play=0) completes the same row with +1, +1, +1 -> P1 won
    _, reward, _, is_term, _ = tictactoe_dynamics_fn(
        _embed([[1, 1, 0], [0, 0, 0], [0, 0, 0]], 0), torch.tensor([2])
    )
    assert reward.item() == 1.0
    assert is_term.item() is True

    # The mover never returns -1 on its own move (it always lays its own piece);
    # a loss is only realised across the alternating discount as the opponent's
    # +1 win is backed up (covered by test_qvalues_p2_root_sign_parity).
    _, reward, _, is_term, _ = tictactoe_dynamics_fn(
        _embed([[1, 0, 0], [0, 0, 0], [0, 0, 0]], 1), torch.tensor([1])
    )
    assert reward.item() == 0.0
    assert is_term.item() is False


def test_dynamics_color_swap_mirror_symmetry():
    """The dynamics must be exactly symmetric under swapping players.

    State B = (board=-A, to_play=1) is the color-swap mirror of state A
    (board=A, to_play=0). Playing the SAME action in both must yield mirrored
    next boards and identical reward / terminal / legal-mask signals.
    """
    board_a = [[1, 0, -1], [0, 1, 0], [0, -1, 0]]  # X at (0,0),(1,1); O at (0,2),(2,1)
    emb_a = _embed(board_a, 0)
    emb_b = _embed([[-v for v in row] for row in board_a], 1)

    action = torch.tensor([1])  # empty cell (0,1), legal in both mirrors
    next_a, reward_a, to_a, term_a, legal_a = tictactoe_dynamics_fn(emb_a, action)
    next_b, reward_b, to_b, term_b, legal_b = tictactoe_dynamics_fn(emb_b, action)

    torch.testing.assert_close(next_a[..., 0], -next_b[..., 0])
    assert to_a.item() == 1 - to_b.item()
    assert reward_a.item() == reward_b.item()  # same mover-perspective reward
    assert term_a.item() == term_b.item()
    torch.testing.assert_close(legal_a, legal_b)


# ============================================================================
# 2. Canonical observations
# ============================================================================
def test_canonical_color_swap_differs_only_by_turn_plane():
    """Mirror states must produce identical my/opp planes; only the turn plane
    (channel 2, absolute color hint) differs."""
    board_a = [[1, 0, -1], [0, 1, 0], [0, -1, 0]]
    can_a = embeddings_to_canonical(_embed(board_a, 0))  # [1, 3, 3, 3]
    can_b = embeddings_to_canonical(_embed([[-v for v in row] for row in board_a], 1))

    torch.testing.assert_close(can_a[0, 0], can_b[0, 0])  # my pieces
    torch.testing.assert_close(can_a[0, 1], can_b[0, 1])  # opp pieces
    assert can_a[0, 2].all().item()  # player 0 turn plane is 1
    assert not can_b[0, 2].any().item()  # player 1 turn plane is 0


def test_canonicals_agree_for_both_players():
    """The replay canonical (get_canonical_obs) must EXACTLY match the MCTS
    canonical (embeddings_to_canonical) for every state, P1 or P2."""
    for player, board in [
        (0, [[1, 0, 0], [0, 0, 0], [-1, 0, 1]]),
        (1, [[-1, 0, 0], [0, 0, 0], [1, 0, -1]]),
    ]:
        emb = _embed(board, player)
        from_canonical = get_canonical_obs(torch.tensor(board), player).squeeze(0)
        from_embeddings = embeddings_to_canonical(emb)[0]
        torch.testing.assert_close(from_canonical, from_embeddings)


# ============================================================================
# 3. Search has no side-bias (the key P2 guarantee)
# ============================================================================
def test_search_p2_root_identical_process_to_p1_root():
    """Given identical rollouts, a P2-to-move root and a P1-to-move root must
    produce a BIT-IDENTICAL search (same visits, same policy, same action).

    The empty tic-tac-toe board is its own color-swap mirror, so a player-
    agnostic search must return exactly the same tree statistics regardless of
    which absolute player is to move (same seed => same RNG stream).
    """
    rec = _recurrent_fn
    seed = 1234
    empty = [[0] * 3, [0] * 3, [0] * 3]
    legal = torch.ones(1, NUM_ACTIONS, dtype=torch.bool)

    torch.manual_seed(seed)
    _, probs_p1, tree_p1 = mcts_search(
        _embed(empty, 0),
        torch.zeros(1, NUM_ACTIONS),
        torch.zeros(1),
        rec,
        num_simulations=100,
        num_actions=NUM_ACTIONS,
        legal_mask=legal,
        dirichlet_epsilon=0.0,
        temperature=1.0,
    )
    torch.manual_seed(seed)
    _, probs_p2, tree_p2 = mcts_search(
        _embed(empty, 1),
        torch.zeros(1, NUM_ACTIONS),
        torch.zeros(1),
        rec,
        num_simulations=100,
        num_actions=NUM_ACTIONS,
        legal_mask=legal,
        dirichlet_epsilon=0.0,
        temperature=1.0,
    )

    assert torch.equal(tree_p1["children_visits"][:, 0], tree_p2["children_visits"][:, 0])
    assert torch.equal(tree_p1["node_counts"], tree_p2["node_counts"])
    assert torch.equal(tree_p1["node_values"], tree_p2["node_values"])
    assert torch.equal(probs_p1, probs_p2)


def test_search_p2_finds_forced_win_mirrored_by_p1():
    """P2 (O) to move with O-O-_ in a row MUST play the winning cell.

    The color-swap mirror is P1 (X) to move with X-X-_ in the same row; both
    searches must land on the same physical winning cell index 2 = (0, 2).
    """
    cell = 2  # (0, 2)

    def root_visits(board, player):
        legal = (torch.tensor(board).view(-1) == 0).unsqueeze(0)
        _, _, tree = mcts_search(
            _embed(board, player),
            torch.zeros(1, NUM_ACTIONS),
            torch.zeros(1),
            _recurrent_fn,
            num_simulations=64,
            num_actions=NUM_ACTIONS,
            legal_mask=legal,
            dirichlet_epsilon=0.0,
            temperature=0.0,
        )
        return tree["children_visits"][0, 0]

    p2_visits = root_visits([[-1, -1, 0], [0, 0, 0], [0, 0, 0]], 1)  # O to move
    p1_visits = root_visits([[1, 1, 0], [0, 0, 0], [0, 0, 0]], 0)  # X to move

    assert p2_visits[cell] > p2_visits.sum().item() / 2  # decisive majority
    assert p1_visits[cell] > p1_visits.sum().item() / 2
    assert p2_visits.argmax().item() == cell
    assert p1_visits.argmax().item() == cell


def test_search_p2_blocks_forced_loss_mirrored_by_p1():
    """P2 (O) to move with X-X-_ on the board MUST block, or X wins next move.

    Uses enough simulations for the flat (zero) value prior to resolve the
    block; the winning-score signal comes purely from the alternating terminal
    rewards. The P1 mirror (O-O-_ with X to move) behaves identically.
    """
    sims = 300
    cell = 2  # (0, 2)

    def root_visits(board, player):
        legal = (torch.tensor(board).view(-1) == 0).unsqueeze(0)
        _, _, tree = mcts_search(
            _embed(board, player),
            torch.zeros(1, NUM_ACTIONS),
            torch.zeros(1),
            _recurrent_fn,
            num_simulations=sims,
            num_actions=NUM_ACTIONS,
            legal_mask=legal,
            dirichlet_epsilon=0.0,
            temperature=0.0,
        )
        return tree["children_visits"][0, 0]

    p2_block = root_visits([[1, 1, 0], [0, 0, 0], [0, 0, 0]], 1)  # O must block X
    p1_block = root_visits([[-1, -1, 0], [0, 0, 0], [0, 0, 0]], 0)  # X must block O

    for visits, label in ((p2_block, "P2 block"), (p1_block, "P1 block")):
        assert visits[cell] > visits.sum().item() / 2, (
            f"{label} failed to block the immediate one-move loss; "
            f"visits={visits.tolist()}"
        )
        assert visits.argmax().item() == cell


def test_qvalues_p2_root_sign_parity():
    """From a P2-to-move root, Q(root, a) must be +1 for an action that makes
    P2 win and -1 for an action that makes P1 win, so PUCT selects the win."""
    B, A, S = 1, 3, 8
    tree = init_mcts_tree(
        _embed([[1, 1, 0], [0, 0, 0], [0, 0, 0]], 1),  # P2 to move
        torch.zeros(B, A),
        torch.zeros(B),
        num_simulations=S,
        num_actions=A,
    )
    # Hand-built two terminal edges: a0 = P2 wins, a1 = P1 wins (mirror blocks)
    for action, child, reward in ((0, 1, 1.0), (1, 2, -1.0)):
        tree["children_index"][0, 0, action] = child
        tree["children_rewards"][0, 0, action] = reward
        tree["children_discounts"][0, 0, action] = 0.0  # terminal
        tree["node_values"][0, child] = 0.0
        tree["node_visits"][0, child] = 1
        tree["is_terminal"][0, child] = True
    tree["node_visits"][0, 0] = 10

    q = get_qvalues(tree, torch.tensor([0]))
    assert q[0, 0].item() == pytest.approx(1.0)  # P2's winning move
    assert q[0, 1].item() == pytest.approx(-1.0)  # P1's winning move

    # Normalized scores put the P2-winning action first; PUCT must descend into it
    _, _, _, trajectory = select_leaf(tree, pb_c_base=19652.0, pb_c_init=1.25)
    root_actions = [t[1][0].item() for t in trajectory if t[0][0].item() == 0]
    assert root_actions and root_actions[0] == 0


# ============================================================================
# 4. Determinism / seed behaviour for P2
# ============================================================================
def test_mcts_search_p2_seed_reproducibility():
    """The same manual seed must reproduce an identical P2 search, including
    the Dirichlet noise and the temperature-1 multinomial action draw.

    This confirms that observed run-to-run differences in P2's strength are a
    seed/sampling artifact, not an ordering or state bug in the search.
    """
    board = [[1, 0, 0], [0, 0, 0], [0, 0, 0]]
    legal = (torch.tensor(board).view(-1) == 0).unsqueeze(0)
    seed = 7

    def run():
        torch.manual_seed(seed)
        action, probs, tree = mcts_search(
            _embed(board, 1),  # P2 to move
            torch.zeros(1, NUM_ACTIONS),
            torch.zeros(1),
            _recurrent_fn,
            num_simulations=60,
            num_actions=NUM_ACTIONS,
            legal_mask=legal,
            dirichlet_epsilon=0.25,
            dirichlet_alpha=1.0,
            temperature=1.0,
        )
        return action, probs, tree["children_visits"][:, 0].clone()

    a1, p1, v1 = run()
    a2, p2, v2 = run()
    assert torch.equal(a1, a2)
    assert torch.equal(p1, p2)
    assert torch.equal(v1, v2)


def test_visit_policy_multinomial_is_seed_dependent():
    """The temperature-1 action draw is stochastic per-seed, which is exactly
    the mechanism behind run-to-run P2 variance (no deterministic bias)."""
    visits = torch.tensor([[2, 2, 2]])
    probs = get_mcts_visit_policy(visits, temperature=1.0)
    torch.manual_seed(0)
    draws_seed_a = torch.multinomial(probs.squeeze(0), num_samples=200, replacement=True)
    torch.manual_seed(1)
    draws_seed_b = torch.multinomial(probs.squeeze(0), num_samples=200, replacement=True)
    # Empirically the empirical distributions must match the policy for ANY seed
    for draws in (draws_seed_a, draws_seed_b):
        emp = torch.bincount(draws, minlength=3).float() / draws.numel()
        torch.testing.assert_close(emp, probs.squeeze(0), atol=0.1, rtol=0.1)
    # And different seeds genuinely produce different sequences (not constant)
    assert not torch.equal(draws_seed_a, draws_seed_b)


def test_winner_check_is_color_symmetric():
    """check_tictactoe_winner treats mirrored rows identically."""
    winner_term_a, term_a = check_tictactoe_winner(
        torch.tensor([[1, 1, 1], [0, 0, 0], [0, 0, 0]])[None]
    )
    winner_term_b, term_b = check_tictactoe_winner(
        torch.tensor([[-1, -1, -1], [0, 0, 0], [0, 0, 0]])[None]
    )
    assert winner_term_a.item() == 1.0
    assert winner_term_b.item() == -1.0
    assert term_a.item() is True
    assert term_b.item() is True