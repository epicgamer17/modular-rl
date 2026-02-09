import numpy as np
from typing import NamedTuple, Optional, Any, Iterator
import copy

from .transition import Transition


# 1. THE INTERFACE
# A lightweight, immutable container.
# Great for passing data between functions: process_step(step)
class TimeStep(NamedTuple):
    observation: Any
    info: dict
    terminated: bool
    truncated: bool
    action: Optional[Any] = None
    reward: Optional[float] = 0.0
    value: Optional[float] = 0.0
    policy: Optional[Any] = None


# class Game:
#     def __init__(self, num_players: int):
#         self.num_players = num_players
#         self.length = 0

#         # 2. THE STORAGE (Columnar)
#         # We store data in separate lists internally.
#         # This is CRITICAL for speed when converting to Tensors later.
#         self._obs = []
#         self._info = []
#         self._rewards = []
#         self._actions = []
#         self._values = []
#         self._policies = []

#     def append(self, step: TimeStep):
#         """
#         Takes a TimeStep object, breaks it down, and stores it efficiently.
#         """
#         self._obs.append(step.observation)
#         self._info.append(step.info)
#         self._rewards.append(step.reward if step.reward is not None else 0.0)
#         self._actions.append(step.action)
#         self._values.append(step.value)
#         self._policies.append(step.policy)
#         self.length += 1

#     def __getitem__(self, index) -> TimeStep:
#         """
#         Reconstructs the TimeStep object when you ask for it.
#         This allows you to treat the Game like a list of objects.
#         """
#         return TimeStep(
#             observation=self._obs[index],
#             info=self._info[index],
#             reward=self._rewards[index],
#             action=self._actions[index],
#             value=self._values[index],
#             policy=self._policies[index],
#         )

#     def set_rewards(self):
#         """
#         Optimized reward setting. Because self._rewards is a simple list,
#         we can iterate over it extremely fast without object overhead.
#         """
#         if self.length == 0:
#             return

#         final_reward_vector = self._rewards[-1]

#         # Vectorized-style logic is easier here
#         for i in reversed(range(self.length)):
#             if self._rewards[i] is not None:
#                 # Direct list access is faster than object attribute access
#                 self._rewards[i] = final_reward_vector[i % self.num_players]

#     def get_batch(self):
#         """
#         The Scaling Win: Convert columns to Numpy/Tensors instantly.
#         This is O(1) complexity overhead vs O(N) for list-of-objects.
#         """
#         return {
#             "obs": np.array(self._obs),
#             "actions": np.array(self._actions),
#             "rewards": np.array(self._rewards, dtype=np.float32),
#             "values": np.array(self._values, dtype=np.float32),
#         }

#     def __len__(self):
#         return self.length


class Game:
    def __init__(
        self, num_players: int
    ):  # num_actions, discount=1.0, n_step=1, gamma=0.99
        self.length = 0
        self.observation_history = []
        self.rewards = []
        self.policy_history = []
        self.value_history = []
        self.action_history = []
        self.info_history = []
        self.player_id_history = []  # Track which player took each action

        self.num_players = num_players
        self.duration_seconds: float = 0.0  # For FPS tracking across processes

    def append(
        self,
        observation,
        info,
        reward: int = None,
        policy=None,
        value=None,
        action=None,
        player_id=None,
    ):
        self.observation_history.append(copy.deepcopy(observation))
        self.info_history.append(copy.deepcopy(info))
        if reward is not None:
            self.rewards.append(reward)
        if policy is not None:
            self.policy_history.append(policy)
        if value is not None:
            self.value_history.append(value)
        if action is not None:
            self.action_history.append(action)
        if player_id is not None:
            self.player_id_history.append(player_id)

    def __len__(self):
        # SHOULD THIS BE LEN OF ACTIONS INSTEAD???
        # AS THIS ALLOWS SAMPLING THE TERMINAL STATE WHICH HAS NO FURTHER ACTIONS
        return len(self.action_history)

    def __iter__(self) -> Iterator[Transition]:
        """
        Allows iterating over the game transitions.
        Yields Transition objects.
        """
        for i in range(len(self.action_history)):
            yield Transition(
                observation=self.observation_history[i],
                action=self.action_history[i],
                reward=float(self.rewards[i]) if self.rewards else 0.0,
                next_observation=self.observation_history[i + 1],
                done=(i == len(self.action_history) - 1),
                info=self.info_history[i] if self.info_history else None,
                next_info=self.info_history[i + 1] if self.info_history else None,
            )
