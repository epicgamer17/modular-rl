"""
Regression test for terminal to_play correctness in the data pipeline.

Before the fix, player_id_history had n_transitions entries but
observation_history had n_states (= n_transitions + 1) entries.
SequenceTensorProcessor defaulted the missing terminal entry to player 0,
making ~50% of terminal to_play targets wrong (whenever the last acting
player was player 0, the terminal to_play is player 1, but got stored as 0).

The fix appends the terminal to_play (next_player_id from the environment)
to player_id_history in SequenceBufferComponent before flushing, so every
observation has a correct to_play.
"""

from core import Blackboard
from components.memory.buffer import SequenceBufferComponent
from data.ingestion import Sequence
from registries import make_muzero_replay_buffer
import numpy as np
import pytest
import torch


pytestmark = pytest.mark.unit


def _make_episode(
    od: tuple,
    na: int,
    game_len: int,
    include_terminal_to_play: bool,
) -> Sequence:
    """Build a Sequence for a game of ``game_len`` moves with alternating players."""
    seq = Sequence(num_players=2)
    seq.append(
        observation=np.random.rand(*od).astype(np.float32),
        terminated=False,
        truncated=False,
        legal_moves=list(range(na)),
    )
    for i in range(game_len):
        is_last = i == game_len - 1
        acting_player = i % 2
        seq.append(
            observation=np.random.rand(*od).astype(np.float32),
            terminated=is_last,
            truncated=False,
            action=i % na,
            reward=1.0 if is_last else 0.0,
            policy=np.ones(na, dtype=np.float32) / na,
            value=0.5 if not is_last else 0.0,
            player_id=acting_player,
            legal_moves=list(range(na)),
        )
    if include_terminal_to_play:
        last_acting = (game_len - 1) % 2
        terminal_to_play = 1 - last_acting
        seq.player_id_history.append(terminal_to_play)
    return seq


@pytest.fixture
def buffer_with_terminal_to_play():
    """Buffer filled with episodes that have correct terminal to_play."""
    torch.manual_seed(42)
    np.random.seed(42)
    na = 9
    od = (9, 3, 3)
    buf = make_muzero_replay_buffer(
        obs_dim=od, num_actions=na, buffer_size=1000,
        batch_size=8, unroll_steps=5,
    )
    for _ in range(20):
        game_len = np.random.randint(5, 9)
        seq = _make_episode(od, na, game_len, include_terminal_to_play=True)
        buf.store_aggregate(seq)
    return buf


@pytest.fixture
def buffer_without_terminal_to_play():
    """Buffer filled with episodes that have the OLD bug (no terminal to_play)."""
    torch.manual_seed(42)
    np.random.seed(42)
    na = 9
    od = (9, 3, 3)
    buf = make_muzero_replay_buffer(
        obs_dim=od, num_actions=na, buffer_size=1000,
        batch_size=8, unroll_steps=5,
    )
    for _ in range(20):
        game_len = np.random.randint(5, 9)
        seq = _make_episode(od, na, game_len, include_terminal_to_play=False)
        buf.store_aggregate(seq)
    return buf


def test_terminal_to_play_not_always_player0(buffer_with_terminal_to_play):
    """Terminal states must have both player 0 and player 1 as to_play targets."""
    batch = buffer_with_terminal_to_play.sample()
    tp = batch["to_plays"]          # [B, U+1, num_players]
    dones = batch["dones"]          # [B, U+1]
    same_game = batch["is_same_episode"]

    terminal = dones & same_game
    terminal[:, 0] = False  # exclude root

    assert terminal.any(), "No terminal positions in batch"

    terminal_tp = tp[terminal]  # [N, 2]
    # One-hot: argmax gives the player index
    terminal_players = terminal_tp.argmax(dim=-1)

    has_p0 = (terminal_players == 0).any()
    has_p1 = (terminal_players == 1).any()
    assert has_p0 and has_p1, (
        f"Terminal to_play is always player {terminal_players[0].item()}. "
        f"Both players should appear as terminal to_play across the batch. "
        f"Unique values: {terminal_players.unique().tolist()}"
    )


def test_to_play_alternates_within_game(buffer_with_terminal_to_play):
    """Within a game, consecutive to_play values should alternate (for tic-tac-toe)."""
    batch = buffer_with_terminal_to_play.sample()
    tp = batch["to_plays"]
    same_game = batch["is_same_episode"]
    tp_mask = batch["to_play_mask"]

    for b in range(tp.shape[0]):
        prev_player = None
        for u in range(tp.shape[1]):
            if not tp_mask[b, u]:
                continue
            if not same_game[b, u]:
                break
            player = tp[b, u].argmax().item()
            if prev_player is not None:
                assert player != prev_player, (
                    f"Batch {b}, step {u}: to_play did not alternate. "
                    f"Got {player} after {prev_player}. "
                    f"Full to_plays: {[tp[b, s].argmax().item() for s in range(tp.shape[1]) if tp_mask[b, s]]}"
                )
            prev_player = player


def test_buggy_buffer_has_wrong_terminal_to_play():
    """Demonstrates that buggy buffers now correctly fail strict alignment assertions."""
    from data.ingestion import SequenceTensorProcessor
    
    # Create an episode that has the OLD bug (no terminal to_play)
    od = (9, 3, 3)
    na = 9
    seq = _make_episode(od, na, game_len=5, include_terminal_to_play=False)
    
    processor = SequenceTensorProcessor(
        num_actions=na,
        num_players=2,
        player_id_mapping={"player_0": 0, "player_1": 1}
    )
    
    # Should raise AssertionError due to history length mismatch
    with pytest.raises(AssertionError) as excinfo:
        processor.process_sequence(seq)
    
    assert "player_id_history length" in str(excinfo.value)
    assert "must match n_states" in str(excinfo.value)


def test_player_id_history_length_with_fix():
    """player_id_history should have n_states entries when terminal to_play is appended."""
    np.random.seed(42)
    od = (9, 3, 3)
    na = 9
    seq = _make_episode(od, na, game_len=5, include_terminal_to_play=True)

    n_states = len(seq.observation_history)
    n_pid = len(seq.player_id_history)
    assert n_pid == n_states, (
        f"player_id_history should have {n_states} entries (one per obs), "
        f"but has {n_pid}. The terminal to_play was not appended."
    )


def test_player_id_history_length_without_fix():
    """Without fix, player_id_history has n_transitions entries (one fewer than obs)."""
    np.random.seed(42)
    od = (9, 3, 3)
    na = 9
    seq = _make_episode(od, na, game_len=5, include_terminal_to_play=False)

    n_states = len(seq.observation_history)
    n_pid = len(seq.player_id_history)
    assert n_pid == n_states - 1, (
        f"Without fix, player_id_history should have {n_states - 1} entries, "
        f"but has {n_pid}."
    )
def test_root_player_id_correctness():
    """Verify that SequenceBufferComponent correctly records the root player ID."""
    from unittest.mock import MagicMock
    
    buf = MagicMock()
    comp = SequenceBufferComponent(replay_buffer=buf, num_players=2)
    
    bb = Blackboard()
    bb.data["obs"] = torch.zeros((1, 9, 3, 3))
    bb.data["player_id"] = 1 # Root is player 1
    bb.data["info"] = {"legal_moves": [0, 1]}
    
    # Execute first step
    comp.execute(bb)
    
    seq = comp._sequence
    assert len(seq.player_id_history) == 1
    assert seq.player_id_history[0] == 1, f"Expected root player 1, got {seq.player_id_history[0]}"
    assert seq.observation_history[0].shape == (9, 3, 3)

def test_transition_player_id_correctness():
    """Verify that transitions use next_player_id for their observations."""
    from unittest.mock import MagicMock
    
    buf = MagicMock()
    comp = SequenceBufferComponent(replay_buffer=buf, num_players=2)
    
    # 1. Root step
    bb1 = Blackboard()
    bb1.data["obs"] = torch.zeros((1, 9, 3, 3))
    bb1.data["player_id"] = 0
    bb1.data["next_player_id"] = 1 # Player 1 will see the next state
    bb1.meta["action"] = 4
    bb1.data["reward"] = 0.0
    bb1.data["dones"] = False
    comp.execute(bb1)
    
    seq = comp._sequence
    # Root (s0, p0) + Transition (s1, p1)
    assert len(seq.player_id_history) == 2
    assert seq.player_id_history[0] == 0
    assert seq.player_id_history[1] == 1
