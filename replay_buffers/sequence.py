from typing import NamedTuple, Optional, Any, Iterator


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


class Sequence:
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
        self.observation_history.append(observation)
        self.info_history.append(info)
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
        Allows iterating over the sequence transitions.
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
