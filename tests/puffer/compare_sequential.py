import numpy as np
import pytest

# ==========================================
# 1. Mock Data Structures (Mimicking your codebase)
# ==========================================


class Sequence:
    """A simple sequence object to mimic your core trajectory storage."""

    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.action_masks = []
        self.dones = []

    def append_initial_state(self, obs, action_mask):
        self.observations.append(obs)
        self.action_masks.append(action_mask)

    def append(self, observation, action, reward, action_mask, done):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.action_masks.append(action_mask)
        self.dones.append(done)

    def __eq__(self, other):
        """Allows direct comparison of two sequences in tests."""
        return (
            self.observations == other.observations
            and self.actions == other.actions
            and self.rewards == other.rewards
            and self.action_masks == other.action_masks
            and self.dones == other.dones
        )


class ReplayBuffer:
    def __init__(self):
        self.sequences = []

    def add(self, sequence):
        self.sequences.append(sequence)


class DeterministicAgent:
    """An agent that always takes the first legal move to ensure identical traces."""

    def select_action(self, sequence):
        mask = sequence.action_masks[-1]
        return int(np.argmax(mask))


# ==========================================
# 2. Mock Environments
# ==========================================


class MockTicTacToeAEC:
    """A deterministic mock of a turn-based game that lasts exactly 3 steps."""

    def __init__(self):
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        return self._get_obs(), self._get_info()

    def step(self, action):
        self.step_count += 1
        reward = 1.0 if self.step_count == 3 else 0.0
        done = self.step_count >= 3
        return self._get_obs(), reward, done, False, self._get_info()

    def _get_obs(self):
        # Observation changes based on step count
        return np.array([self.step_count], dtype=np.float32)

    def _get_info(self):
        # Mask out actions as the game progresses
        mask = np.ones(5)
        mask[: self.step_count] = 0
        return {"action_mask": mask}


class PufferlibAutoResetMock:
    """Mimics Pufferlib's auto-reset and info stashing."""

    def __init__(self, env):
        self.env = env

    def reset(self):
        return self.env.reset()

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        if done:
            # 1. The Wrapper stashes the true terminal state
            info["terminal_observation"] = obs.copy()
            info["terminal_action_mask"] = info["action_mask"].copy()

            # 2. Pufferlib auto-resets the environment instantly
            obs, reset_info = self.env.reset()

            # 3. Pufferlib returns the NEW observation, but flags done=True
            info["action_mask"] = reset_info["action_mask"]

        return obs, reward, terminated, truncated, info


# ==========================================
# 3. The Actors
# ==========================================


class SequentialActor:
    """Baseline: Runs purely sequentially, no auto-resets."""

    def __init__(self, env, agent, buffer):
        self.env = env
        self.agent = agent
        self.buffer = buffer

    def run(self, steps):
        obs, info = self.env.reset()
        seq = Sequence()
        seq.append_initial_state(obs, info["action_mask"])

        for _ in range(steps):
            action = self.agent.select_action(seq)
            next_obs, reward, done, _, info = self.env.step(action)

            seq.append(next_obs, action, reward, info["action_mask"], done)

            if done:
                self.buffer.add(seq)
                obs, info = self.env.reset()
                seq = Sequence()
                seq.append_initial_state(obs, info["action_mask"])


class PufferActor:
    """Test Subject: Handles auto-resets using the info stash."""

    def __init__(self, env, agent, buffer):
        self.env = env
        self.agent = agent
        self.buffer = buffer

    def run(self, steps):
        obs, info = self.env.reset()
        seq = Sequence()
        seq.append_initial_state(obs, info["action_mask"])

        for _ in range(steps):
            action = self.agent.select_action(seq)
            next_obs, reward, done, _, info = self.env.step(action)

            if done:
                # Extract true terminal state
                term_obs = info["terminal_observation"]
                term_mask = info["terminal_action_mask"]

                seq.append(term_obs, action, reward, term_mask, done=True)
                self.buffer.add(seq)

                # Start new sequence with the auto-reset observation
                seq = Sequence()
                seq.append_initial_state(next_obs, info["action_mask"])
            else:
                seq.append(next_obs, action, reward, info["action_mask"], done=False)


# ==========================================
# 4. The PyTest Assertions
# ==========================================


def test_puffer_actor_matches_sequential_actor():
    NUM_STEPS = 12  # Exactly 4 games of 3 steps each

    # Setup Sequential Baseline
    seq_env = MockTicTacToeAEC()
    seq_agent = DeterministicAgent()
    seq_buffer = ReplayBuffer()
    seq_actor = SequentialActor(seq_env, seq_agent, seq_buffer)

    # Setup Pufferlib Test Subject
    puf_env = PufferlibAutoResetMock(MockTicTacToeAEC())
    puf_agent = DeterministicAgent()
    puf_buffer = ReplayBuffer()
    puf_actor = PufferActor(puf_env, puf_agent, puf_buffer)

    # Run both actors
    seq_actor.run(NUM_STEPS)
    puf_actor.run(NUM_STEPS)

    # Assertions
    assert (
        len(seq_buffer.sequences) == 4
    ), "Sequential buffer should have 4 completed games."
    assert (
        len(puf_buffer.sequences) == 4
    ), "Puffer buffer should have 4 completed games."

    for i in range(4):
        seq_game = seq_buffer.sequences[i]
        puf_game = puf_buffer.sequences[i]

        # Check that the agent was fed the exact same masks/observations
        np.testing.assert_array_equal(seq_game.observations, puf_game.observations)
        np.testing.assert_array_equal(seq_game.action_masks, puf_game.action_masks)

        # Check that the buffer recorded the exact same transitions
        assert seq_game.actions == puf_game.actions
        assert seq_game.rewards == puf_game.rewards
        assert seq_game.dones == puf_game.dones

        # Final terminal state sanity check
        assert puf_game.dones[-1] is True
        assert puf_game.rewards[-1] == 1.0


if __name__ == "__main__":
    pytest.main(["-v", __file__])
