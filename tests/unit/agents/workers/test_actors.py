import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

from agents.workers.actors import RolloutActor
from agents.action_selectors.types import InferenceResult
from replay_buffers.sequence import Sequence
from replay_buffers.processors import SequenceTensorProcessor

pytestmark = pytest.mark.unit


def test_actor_trajectory_payload_contract():
    """
    CONTRACT: The Actor must step the environment, consume MCTS outputs (target policies),
    and package a trajectory dictionary with exact expected keys and tensor shapes.

    This test verifies that:
    1. 'target_policies' from extras are prioritized over 'probs'.
    2. 'value' (root value) is captured correctly.
    3. The payload matches the MuZero training contract.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Mock Dependencies
    num_envs = 1
    num_actions = 4
    obs_shape = (3, 3)

    # Mock Environment (Adapter)
    # Returns 2 steps, then terminates
    mock_adapter = MagicMock()
    mock_adapter.num_envs = num_envs

    obs_step_0 = torch.zeros((num_envs, *obs_shape))
    obs_step_1 = torch.ones((num_envs, *obs_shape))
    obs_step_2 = torch.full((num_envs, *obs_shape), 2.0)

    mock_adapter.reset.return_value = (
        obs_step_0,
        {"player_id": torch.zeros(num_envs, dtype=torch.long)},
    )
    mock_adapter.current_lengths = np.zeros(num_envs)
    mock_adapter.get_metrics.return_value = ([], [])

    # step() -> next_obs, reward, terminal, truncated, info
    mock_adapter.step.side_effect = [
        (
            obs_step_1,
            torch.tensor([1.0]),
            torch.tensor([False]),
            torch.tensor([False]),
            {"player_id": torch.zeros(num_envs, dtype=torch.long)},
        ),
        (
            obs_step_2,
            torch.tensor([2.0]),
            torch.tensor([True]),
            torch.tensor([False]),
            {"player_id": torch.zeros(num_envs, dtype=torch.long)},
        ),
    ]

    # Mock Policy Source (MCTS)
    mock_policy_source = MagicMock()

    # target_policies: [B, A]
    target_policy_0 = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    target_policy_1 = torch.tensor([[0.0, 0.0, 1.0, 0.0]])

    # InferenceResult at step 0
    res_0 = InferenceResult(
        value=torch.tensor([0.5]),
        probs=torch.tensor([[0.25, 0.25, 0.25, 0.25]]),  # Exploratory
        extras={"target_policies": target_policy_0},
    )

    # InferenceResult at step 1
    res_1 = InferenceResult(
        value=torch.tensor([1.5]),
        probs=torch.tensor([[0.0, 1.0, 0.0, 0.0]]),  # Exploratory
        extras={"target_policies": target_policy_1},
    )

    mock_policy_source.get_inference.side_effect = [res_0, res_1]

    # Mock Action Selector
    mock_selector = MagicMock()
    mock_selector.select_action.side_effect = [
        (torch.tensor([1]), {}),  # Action 1 at step 0
        (torch.tensor([2]), {}),  # Action 2 at step 1
    ]

    # Mock Buffer to capture the sequence
    mock_buffer = MagicMock()
    captured_sequence = None

    def store_aggregate_capture(sequence, **kwargs):
        nonlocal captured_sequence
        captured_sequence = sequence

    mock_buffer.store_aggregate.side_effect = store_aggregate_capture

    # 2. Initialize RolloutActor
    # adapter_cls is called in __init__, so we make it return our mock_adapter
    mock_adapter_cls = MagicMock(return_value=mock_adapter)

    actor = RolloutActor(
        adapter_cls=mock_adapter_cls,
        adapter_args=(),
        network=MagicMock(),  # agent_network
        policy_source=mock_policy_source,
        buffer=mock_buffer,
        action_selector=mock_selector,
        num_actions=num_actions,
        num_players=1,
        flush_incomplete=False,
    )

    # 3. Collect Data (Trigger Episode Completion)
    # 2 steps should finish the episode in our mock
    actor.collect(num_steps=2)

    # 4. Verify Payload Contract
    assert (
        captured_sequence is not None
    ), "Actor should have flushed the completed sequence to the buffer."

    # In MuZero, targets are built from the sequence using a processor.
    # The 'contract' usually refers to the dictionary derived from the sequence.
    processor = SequenceTensorProcessor(
        num_actions=num_actions,
        num_players=1,
        player_id_mapping={0: 0},  # Dummy mapping for testing
    )

    # process_sequence returns a dict of tensors
    payload = processor.process_sequence(captured_sequence)

    # verify standard keys
    expected_keys = [
        "observations",
        "actions",
        "rewards",
        "policies",
        "values",
        "to_plays",
    ]
    for key in expected_keys:
        assert key in payload, f"Missing key '{key}' in trajectory payload."

    # MATH VERIFICATION: Check target policies
    # payload['policies'] should have indexed 0 and 1 from our mocks
    # Note: Sequence lengths usually T=K+1 for state-aligned data, or K for transitions.
    # SequenceTensorProcessor returns [K, ...] for transitions or [T, ...] for states.

    # Step 0: policy was target_policy_0
    assert torch.allclose(
        payload["policies"][0], target_policy_0[0]
    ), "Policy at step 0 should match MCTS target_policies."
    # Step 1: policy was target_policy_1
    assert torch.allclose(
        payload["policies"][1], target_policy_1[0]
    ), "Policy at step 1 should match MCTS target_policies."

    # VALUE VERIFICATION: Check root values
    assert payload["values"][0] == 0.5, "Value at step 0 should match MCTS root value."
    assert payload["values"][1] == 1.5, "Value at step 1 should match MCTS root value."

    # SHAPE VERIFICATION
    assert payload["observations"].shape == (3, *obs_shape)  # T=3 (including terminal)
    assert payload["actions"].shape == (2,)  # K=2 transitions
    assert payload["rewards"].shape == (2,)  # K=2 transitions

    # MAP TO USER CONTRACT NAMES (Syntactic Parity)
    muzero_contract = {
        "observations": payload["observations"],
        "actions": payload["actions"],
        "rewards": payload["rewards"],
        "root_values": payload["values"],  # Mapping values -> root_values
        "policies": payload["policies"],
        "to_play": payload["to_plays"],  # Mapping to_plays -> to_play
    }

    assert muzero_contract["root_values"][0] == 0.5
    assert muzero_contract["to_play"].dtype == torch.int16


def test_rollout_actor_collect_metrics_scalar():
    """
    REGRESSION: RolloutActor should return scalar floats in 'batch_scores' for 
    single-player games, not arrays of shape (1,).
    """
    # 1. Mock setup
    num_envs = 1
    num_actions = 2
    mock_adapter = MagicMock()
    mock_adapter.num_envs = num_envs
    mock_adapter.reset.return_value = (torch.zeros((1, 1)), {"player_id": torch.zeros(1)})
    # Return terminal on first step to trigger score capture
    mock_adapter.step.return_value = (
        torch.zeros((1, 1)), 
        torch.tensor([10.0]), 
        torch.tensor([True]), 
        torch.tensor([False]), 
        {"player_id": torch.zeros(1)}
    )
    mock_adapter.get_metrics.return_value = ([10.0], [1])
    mock_adapter.current_lengths = np.zeros(1)
    
    mock_selector = MagicMock()
    mock_selector.select_action.return_value = (torch.tensor([0]), {})
    
    mock_policy_source = MagicMock()
    mock_policy_source.get_inference.return_value = InferenceResult(
        value=torch.tensor([0.0]),
        probs=torch.tensor([[0.5, 0.5]]),
        extras={}
    )

    actor = RolloutActor(
        adapter_cls=MagicMock(return_value=mock_adapter),
        adapter_args=(),
        network=MagicMock(),
        policy_source=mock_policy_source,
        buffer=MagicMock(),
        action_selector=mock_selector,
        num_players=1,
    )

    # 2. Collect (finishes 1 episode)
    metrics = actor.collect(num_steps=1)

    # 3. VERIFY: batch_scores should contain a float, not a numpy array
    assert "batch_scores" in metrics
    assert len(metrics["batch_scores"]) == 1
    score = metrics["batch_scores"][0]
    
    # This was the bug: score was an np.ndarray of shape (1,)
    assert isinstance(score, (float, int, np.float64, np.float32)), \
        f"Expected scalar score for single-player game, got {type(score)} shape {getattr(score, 'shape', 'N/A')}"
    # Actually let's check what SequenceTensorProcessor returns.
    # Line 588 of processors.py: torch.tensor(sequence.player_id_history, ...)
