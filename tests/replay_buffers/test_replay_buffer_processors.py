from types import SimpleNamespace

import numpy as np
import pytest
import torch

from replay_buffers.processors import (
    GAEProcessor,
    NStepInputProcessor,
    NStepUnrollProcessor,
    SequenceTensorProcessor,
    TerminationFlagsInputProcessor,
)
from replay_buffers.sequence import Sequence

pytestmark = pytest.mark.unit


def test_replay_buffer_processors_nstep_input_non_terminal_chain():
    processor = NStepInputProcessor(n_step=3, gamma=0.5, num_players=1)

    assert (
        processor.process_single(
            player=0,
            rewards=1.0,
            dones=False,
            terminated=False,
            truncated=False,
            next_observations=np.array([1.0], dtype=np.float32),
            next_infos={"frame": 1},
            next_legal_moves=[0, 1],
        )
        is None
    )
    assert (
        processor.process_single(
            player=0,
            rewards=2.0,
            dones=False,
            terminated=False,
            truncated=False,
            next_observations=np.array([2.0], dtype=np.float32),
            next_infos={"frame": 2},
            next_legal_moves=[1],
        )
        is None
    )
    emitted = processor.process_single(
        player=0,
        rewards=3.0,
        dones=False,
        terminated=False,
        truncated=False,
        next_observations=np.array([3.0], dtype=np.float32),
        next_infos={"frame": 3},
        next_legal_moves=[0],
    )

    assert emitted is not None
    assert emitted["rewards"] == pytest.approx(2.75)
    assert emitted["dones"] is False
    assert emitted["terminated"] is False
    assert emitted["truncated"] is False
    assert emitted["next_infos"] == {"frame": 3}
    assert np.allclose(emitted["next_observations"], np.array([3.0], dtype=np.float32))


def test_replay_buffer_processors_nstep_input_stops_at_done():
    processor = NStepInputProcessor(n_step=3, gamma=0.9, num_players=1)

    processor.process_single(
        player=0,
        rewards=1.0,
        dones=False,
        terminated=False,
        truncated=False,
        next_observations=np.array([10.0], dtype=np.float32),
        next_infos={"s": "mid"},
        next_legal_moves=[0, 1],
    )
    processor.process_single(
        player=0,
        rewards=5.0,
        dones=True,
        terminated=True,
        truncated=False,
        next_observations=np.array([20.0], dtype=np.float32),
        next_infos={"s": "terminal"},
        next_legal_moves=[1],
    )
    emitted = processor.process_single(
        player=0,
        rewards=100.0,
        dones=False,
        terminated=False,
        truncated=False,
        next_observations=np.array([30.0], dtype=np.float32),
        next_infos={"s": "post"},
        next_legal_moves=[],
    )

    assert emitted is not None
    assert emitted["rewards"] == pytest.approx(5.5)
    assert emitted["dones"] is True
    assert emitted["terminated"] is True
    assert emitted["truncated"] is False
    assert emitted["next_infos"] == {"s": "terminal"}
    assert np.allclose(emitted["next_observations"], np.array([20.0], dtype=np.float32))


def test_replay_buffer_processors_nstep_sequence_flushes_terminal_remainder():
    processor = NStepInputProcessor(n_step=3, gamma=1.0, num_players=1)
    transitions = [
        {
            "player": 0,
            "rewards": 1.0,
            "dones": False,
            "terminated": False,
            "truncated": False,
            "next_observations": np.array([1.0], dtype=np.float32),
            "next_infos": {"t": 0},
            "next_legal_moves": [0, 1],
        },
        {
            "player": 0,
            "rewards": 2.0,
            "dones": True,
            "terminated": True,
            "truncated": False,
            "next_observations": np.array([2.0], dtype=np.float32),
            "next_infos": {"t": 1},
            "next_legal_moves": [1],
        },
    ]
    sequence = SimpleNamespace(done_history=[False, False, True])

    processed = processor.process_sequence(sequence=sequence, transitions=transitions)
    flushed = processed["transitions"]

    assert len(flushed) == 2
    assert flushed[0]["rewards"] == pytest.approx(3.0)
    assert flushed[1]["rewards"] == pytest.approx(2.0)
    assert flushed[0]["dones"] is True
    assert flushed[1]["dones"] is True


def test_replay_buffer_processors_termination_flags_defaults_and_overrides():
    processor = TerminationFlagsInputProcessor()

    defaulted = processor.process_single(dones=True)
    explicit = processor.process_single(dones=True, terminated=False, truncated=True)

    assert defaulted["terminated"] is True
    assert defaulted["truncated"] is False
    assert explicit["terminated"] is False
    assert explicit["truncated"] is True


def test_replay_buffer_processors_sequence_tensor_happy_path():
    sequence = Sequence(num_players=1)
    sequence.append(
        observation=np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float32),
        terminated=False,
        truncated=False,
        value=0.25,
        player_id="player_0",
        legal_moves=[0, 1],
        chance=0,
    )
    sequence.append(
        observation=np.array([1.0, 1.1, 1.2, 1.3], dtype=np.float32),
        terminated=False,
        truncated=False,
        reward=1.5,
        policy=np.array([0.7, 0.3], dtype=np.float32),
        value=0.5,
        action=1,
        player_id="player_0",
        legal_moves=[1],
        chance=1,
    )
    sequence.append(
        observation=np.array([2.0, 2.1, 2.2, 2.3], dtype=np.float32),
        terminated=True,
        truncated=False,
        reward=-0.5,
        policy=np.array([0.4, 0.6], dtype=np.float32),
        value=0.0,
        action=0,
        player_id="player_0",
        legal_moves=[0],
        chance=0,
    )

    processor = SequenceTensorProcessor(
        num_actions=2,
        num_players=1,
        player_id_mapping={"player_0": 0},
    )
    out = processor.process_sequence(sequence)

    assert out["n_states"] == 3
    assert out["observations"].shape == (3, 4)
    assert out["actions"].shape == (2,)
    assert out["rewards"].shape == (2,)
    assert out["policies"].shape == (2, 2)
    assert out["to_plays"].tolist() == [0, 0, 0]
    assert out["terminated"].tolist() == [False, False, True]
    assert out["truncated"].tolist() == [False, False, False]
    assert out["dones"].tolist() == [False, False, True]
    assert out["legal_masks"].shape == (3, 2)


def test_replay_buffer_processors_sequence_tensor_rejects_unknown_player():
    sequence = Sequence(num_players=1)
    sequence.append(
        observation=np.array([0.0, 0.0], dtype=np.float32),
        terminated=False,
        truncated=False,
        value=0.0,
        player_id="known",
    )
    sequence.append(
        observation=np.array([1.0, 1.0], dtype=np.float32),
        terminated=True,
        truncated=False,
        reward=1.0,
        action=0,
        value=0.0,
        player_id="unknown",
    )

    processor = SequenceTensorProcessor(
        num_actions=2,
        num_players=1,
        player_id_mapping={"known": 0},
    )

    with pytest.raises(ValueError, match="player_id 'unknown' not found"):
        processor.process_sequence(sequence)


def test_replay_buffer_processors_sequence_tensor_rejects_bad_lengths():
    sequence = Sequence(num_players=1)
    sequence.observation_history = [
        np.array([0.0], dtype=np.float32),
        np.array([1.0], dtype=np.float32),
    ]
    sequence.action_history = [0, 1]
    sequence.rewards = [1.0, 2.0]
    sequence.policy_history = [
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.5, 0.5], dtype=np.float32),
    ]
    sequence.value_history = [0.0, 0.0]
    sequence.player_id_history = [0, 0]
    sequence.chance_history = [0, 0]
    sequence.legal_moves_history = [[0], [1]]
    sequence.terminated_history = [False, True]
    sequence.truncated_history = [False, False]
    sequence.done_history = [False, True]

    processor = SequenceTensorProcessor(
        num_actions=2,
        num_players=1,
        player_id_mapping={0: 0},
    )

    with pytest.raises(
        ValueError, match="observation_history must have exactly one more entry"
    ):
        processor.process_sequence(sequence)


def test_replay_buffer_processors_gae_uses_last_value_and_done_mask():
    processor = GAEProcessor(gamma=1.0, gae_lambda=1.0)
    transitions = [
        {"rewards": 1.0, "values": 0.5, "dones": False, "policies": 0.1},
        {"rewards": 2.0, "values": 1.0, "dones": True, "policies": 0.2},
    ]
    sequence = SimpleNamespace(value_history=[0.5, 1.0, 1.5])

    out = processor.process_sequence(sequence=sequence, transitions=transitions)
    processed = out["transitions"]

    assert processed[0]["advantages"] == pytest.approx(2.5)
    assert processed[1]["advantages"] == pytest.approx(1.0)
    assert processed[0]["returns"] == pytest.approx(4.5)
    assert processed[1]["returns"] == pytest.approx(3.5)
    assert processed[0]["log_probabilities"] == pytest.approx(0.1)
    assert processed[1]["log_probabilities"] == pytest.approx(0.2)


def test_replay_buffer_processors_unroll_respects_truncated_done_branch():
    np.random.seed(42)

    processor = NStepUnrollProcessor(
        unroll_steps=1,
        n_step=1,
        gamma=1.0,
        num_actions=2,
        num_players=1,
        max_size=5,
    )
    buffers = {
        "observations": torch.arange(20, dtype=torch.float32).view(5, 4),
        "rewards": torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0], dtype=torch.float32),
        "values": torch.zeros(5, dtype=torch.float32),
        "policies": torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        "actions": torch.tensor([0, 1, 0, 0, 0], dtype=torch.int64),
        "to_plays": torch.zeros(5, dtype=torch.int16),
        "chances": torch.tensor([[0], [1], [0], [0], [0]], dtype=torch.int16),
        "game_ids": torch.tensor([7, 7, 7, 7, 7], dtype=torch.int64),
        "legal_masks": torch.ones((5, 2), dtype=torch.bool),
        "terminated": torch.tensor([False, False, False, False, False], dtype=torch.bool),
        "truncated": torch.tensor([False, True, False, False, False], dtype=torch.bool),
        "ids": torch.tensor([10, 11, 12, 13, 14], dtype=torch.int64),
        "training_steps": torch.zeros(5, dtype=torch.int64),
    }

    batch = processor.process_batch(indices=[0], buffers=buffers)

    assert batch["action_mask"].tolist() == [[True, False]]
    assert batch["dones"].tolist() == [[False, True]]
    assert batch["rewards"][0].tolist() == pytest.approx([0.0, 1.0])
    assert torch.allclose(batch["policies"][0, 1], torch.tensor([0.5, 0.5]))
