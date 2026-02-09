import numpy as np
import torch
from replay_buffers.sequence import Sequence
from replay_buffers.transition import Transition


def test_sequence_iterable():
    num_players = 1
    seq = Sequence(num_players)

    # Simulate a small game
    obs0 = np.zeros((4,))
    info0 = {"step": 0}
    seq.append(obs0, info0)

    actions = [1, 2, 3]
    rewards = [0.5, 1.0, -0.5]
    obs_list = [np.ones((4,)) * i for i in range(1, 4)]

    for i in range(3):
        seq.append(
            observation=obs_list[i],
            info={"step": i + 1},
            action=actions[i],
            reward=rewards[i],
        )

    print(f"Sequence length: {len(seq)}")
    assert len(seq) == 3

    transitions = list(seq)
    print(f"Number of transitions: {len(transitions)}")
    assert len(transitions) == 3

    for i, t in enumerate(transitions):
        assert isinstance(t, Transition)
        print(f"Step {i}: Action {t.action}, Reward {t.reward}, Done {t.done}")

        # Verify values
        assert t.action == actions[i]
        assert t.reward == rewards[i]
        if i < 2:
            assert not t.done
            assert np.array_equal(t.next_observation, obs_list[i])
        else:
            assert t.done
            assert np.array_equal(t.next_observation, obs_list[i])

    print("Test passed!")


if __name__ == "__main__":
    test_sequence_iterable()
