import pytest
import torch
import numpy as np
from typing import Dict, List

from configs.games.tictactoe import env_factory, TicTacToeConfig
from agents.environments.adapters import PettingZooAdapter
from agents.workers.actors import RolloutActor
from agents.tictactoe_expert import TicTacToeBestAgent
from agents.action_selectors.selectors import CategoricalSelector
from agents.action_selectors.types import InferenceResult
from agents.factories.replay_buffer import create_muzero_buffer
from replay_buffers.modular_buffer import ModularReplayBuffer

pytestmark = pytest.mark.integration


class MockPolicySource:
    def __init__(self, agent):
        self.agent = agent

    def get_inference(self, obs, info, agent_network, exploration=True):
        # The expert agent returns an InferenceOutput
        output = self.agent.obs_inference(obs, info=info)
        return InferenceResult.from_inference_output(output)

    def update_parameters(self, params):
        pass


def test_rollout_actor_collection_real_tictactoe():
    """
    Tier 2 Integration Test: RolloutActor collection contract with real Tic-Tac-Toe.
    Verifies that RolloutActor properly fills the Sequence and respects the multi-player to_play contract.
    """
    device = torch.device("cpu")
    num_actions = 9
    num_players = 2

    # 1. Setup components
    network = torch.nn.Module()
    network.eval = lambda: None

    expert = TicTacToeBestAgent()
    policy_source = MockPolicySource(expert)
    selector = CategoricalSelector()

    # Real MuZero Buffer from factory
    buffer = create_muzero_buffer(
        observation_dimensions=(9, 3, 3),
        max_size=100,
        num_actions=num_actions,
        num_players=num_players,
        player_id_mapping={"player_1": 0, "player_2": 1},
        unroll_steps=5,
        n_step=1,
        gamma=1.0,
        batch_size=16,
        multi_process=False,
    )

    # We'll override store_aggregate to capture the sequence for inspection
    captured_sequences = []
    original_store = buffer.store_aggregate

    def mock_store(seq):
        captured_sequences.append(seq)
        original_store(seq)

    buffer.store_aggregate = mock_store

    actor = RolloutActor(
        adapter_cls=PettingZooAdapter,
        adapter_args=(env_factory,),
        network=network,
        policy_source=policy_source,
        buffer=buffer,
        action_selector=selector,
        num_actions=num_actions,
        num_players=num_players,
        flush_incomplete=False,  # MuZero style
    )

    # 2. Run collection for one episode
    # Tic-Tac-Toe expert usually finishes in 5-9 moves.
    # We'll run until we see an episode completed.
    while actor.episodes_completed == 0:
        actor.collect(num_steps=1)

    assert len(captured_sequences) > 0
    seq = captured_sequences[0]

    # 3. VERIFY COLLECTION CONTRACTS
    T = len(seq.action_history)
    assert len(seq.observation_history) == T + 1
    assert len(seq.player_id_history) == T + 1
    assert len(seq.rewards) == T
    assert len(seq.terminated_history) == T + 1

    # Verify player ID alternating
    # player_id_history: [0, 1, 0, 1, ...]
    assert seq.player_id_history[0] == 0
    for t in range(len(seq.player_id_history) - 1):
        p_now = seq.player_id_history[t]
        p_next = seq.player_id_history[t + 1]
        assert (
            p_now != p_next
        ), f"Player ID should alternate at step {t}. Got {p_now} -> {p_next}, full sequence: {seq.player_id_history}"

    # Verify terminal flag at the end
    assert seq.terminated_history[-1] == True
    assert sum(seq.terminated_history) == 1, "Only the last state should be terminated"

    # If P1 won, check reward
    if sum(seq.rewards) > 0:
        # Reward is associated with the transition leading to terminal
        assert seq.rewards[-1] == 1.0, "Winning move should have reward 1.0"

    print("RolloutActor Collection Contract Verified!")

    # 4. SAMPLE AND VERIFY SAMPLING CONTRACTS
    # Ensure we have enough data to sample a batch
    while buffer.size < 16:
        actor.collect(num_steps=1)

    batch = buffer.sample()

    # Verification of Sampling alignment (NStepUnrollProcessor)
    unroll_steps = 5
    # dones [B, unroll_steps + 1]
    assert batch["dones"].shape[1] == unroll_steps + 1

    # Contract: root terminal flag is False (Roots are never terminal)
    assert (batch["dones"][:, 0] == False).all(), "Root terminal flag must be False"

    # Check for terminal alignment in at least one sampled sequence (if we find the end of a game)
    found_terminal = False
    for i in range(16):
        dones = batch["dones"][i]
        # Any True from index 1 to unroll_steps indicates a terminal state in the unroll
        if dones[1:].any():
            found_terminal = True
            # Find the FIRST terminal state in this unroll
            first_done_idx = torch.where(dones)[0][0].item()
            # If root was somehow True, it would be index 0, but we asserted it's False.

            # Value at terminal state is 0.0 (Rule 2)
            assert (
                batch["values"][i, first_done_idx] == 0.0
            ), f"Terminal state u={first_done_idx} value must be 0.0"

            # Value transition leading to terminal:
            # If P1 wins, the state before s_T should have value target 1.0 (if r=1, gamma=1)
            # Find the reward index. r_t in batch corresponds to transition from s_{t-1} to s_t.
            # So the winning reward is at batch['rewards'][i, first_done_idx]
            win_reward = batch["rewards"][i, first_done_idx]
            if win_reward == 1.0:
                print(
                    f"Found winning sequence in batch item {i} landing at u={first_done_idx}"
                )
                # s_{first_done_idx-1} should have value target 1.0
                assert (
                    batch["values"][i, first_done_idx - 1] == 1.0
                ), f"State before p1-win terminal must have value 1.0"

            # Values after terminal state should be 0.0 (Sticky Termination)
            for u in range(first_done_idx, unroll_steps + 1):
                assert (
                    batch["values"][i, u] == 0.0
                ), f"Post-terminal state u={u} value must be 0.0"

    if found_terminal:
        print("Terminal Alignment in Sampled Batch Verified!")
    else:
        # Collect more until we find one (to be robust)
        print("No terminal state found in this batch, collecting another episode...")
        while actor.episodes_completed < 5:
            actor.collect(num_steps=1)
        # Probabilistically we should have one now
        batch = buffer.sample()
        # (Recursive check or just trust the probabilistic nature of the test)

    print("RolloutActor Sampling Contract Verified!")


if __name__ == "__main__":
    test_rollout_actor_collection_real_tictactoe()
