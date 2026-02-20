from dataclasses import dataclass, field
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


@dataclass
class Sequence:
    num_players: int
    observation_history: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    policy_history: list = field(default_factory=list)
    value_history: list = field(default_factory=list)
    action_history: list = field(default_factory=list)
    info_history: list = field(default_factory=list)
    terminated_history: list = field(default_factory=list)
    truncated_history: list = field(default_factory=list)
    done_history: list = field(default_factory=list)
    player_id_history: list = field(default_factory=list)  # per transition
    duration_seconds: float = 0.0  # For FPS tracking across processes

    def append(
        self,
        observation,
        info,
        terminated: bool,
        truncated: bool,
        reward: int = None,
        policy=None,
        value=None,
        action=None,
        player_id=None,
    ):
        self.observation_history.append(observation)
        self.info_history.append(info)
        self.terminated_history.append(bool(terminated))
        self.truncated_history.append(bool(truncated))
        self.done_history.append(bool(terminated or truncated))
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
        n_states = len(self.observation_history)
        if len(self.info_history) != n_states:
            raise ValueError("info_history length must match observation_history length")
        if len(self.terminated_history) != n_states:
            raise ValueError(
                "terminated_history length must match observation_history length"
            )
        if len(self.truncated_history) != n_states:
            raise ValueError(
                "truncated_history length must match observation_history length"
            )
        if len(self.done_history) != n_states:
            raise ValueError("done_history length must match observation_history length")
        if len(self.action_history) + 1 != n_states:
            raise ValueError(
                "observation_history must have exactly one more entry than action_history"
            )

        for i in range(len(self.action_history)):
            next_info = self.info_history[i + 1] if self.info_history else None
            terminated = bool(self.terminated_history[i + 1])
            truncated = bool(self.truncated_history[i + 1])
            yield Transition(
                observation=self.observation_history[i],
                action=self.action_history[i],
                reward=float(self.rewards[i]) if self.rewards else 0.0,
                next_observation=self.observation_history[i + 1],
                done=terminated or truncated,
                terminated=terminated,
                truncated=truncated,
                info=self.info_history[i] if self.info_history else None,
                next_info=next_info,
            )
