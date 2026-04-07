from dataclasses import dataclass, field
from typing import NamedTuple, Optional, Any, Iterator, List, Dict


from data.transition import Transition


class TimeStep(NamedTuple):
    observation: Any
    terminated: bool
    truncated: bool
    action: Optional[Any] = None
    reward: Optional[float] = 0.0
    value: Optional[float] = 0.0
    policy: Optional[Any] = None
    legal_moves: Optional[List[int]] = None
    chance: Optional[int] = None


# TODO: update sequence to strip the batch dimension, as a sequence is implied batch 1 (then update replay buffer accordingly)
@dataclass
class Sequence:
    num_players: int
    observation_history: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    policy_history: list = field(default_factory=list)
    value_history: list = field(default_factory=list)
    action_history: list = field(default_factory=list)
    legal_moves_history: list = field(default_factory=list)
    chance_history: list = field(default_factory=list)
    all_player_rewards_history: list = field(default_factory=list)
    terminated_history: list = field(default_factory=list)
    truncated_history: list = field(default_factory=list)
    done_history: list = field(default_factory=list)
    player_id_history: list = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0

    def append(
        self,
        observation,
        terminated: bool,
        truncated: bool,
        reward: Optional[float] = None,
        policy=None,
        value=None,
        action=None,
        player_id=None,
        legal_moves: Optional[List[int]] = None,
        chance: Optional[int] = None,
        all_player_rewards: Optional[Dict[str, float]] = None,
    ):
        self.observation_history.append(observation)
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
        if legal_moves is not None:
            self.legal_moves_history.append(legal_moves)
        if chance is not None:
            self.chance_history.append(chance)
        if all_player_rewards is not None:
            self.all_player_rewards_history.append(all_player_rewards)

    def __len__(self):
        return len(self.action_history)

    def __iter__(self) -> Iterator[Transition]:
        """
        Allows iterating over the sequence transitions.
        Yields Transition objects.
        """
        n_states = len(self.observation_history)
        if len(self.terminated_history) != n_states:
            raise ValueError(
                "terminated_history length must match observation_history length"
            )
        if len(self.truncated_history) != n_states:
            raise ValueError(
                "truncated_history length must match observation_history length"
            )
        if len(self.done_history) != n_states:
            raise ValueError(
                "done_history length must match observation_history length"
            )
        if len(self.action_history) + 1 != n_states:
            raise ValueError(
                "observation_history must have exactly one more entry than action_history"
            )

        for i in range(len(self.action_history)):
            terminated = bool(self.terminated_history[i + 1])
            truncated = bool(self.truncated_history[i + 1])
            legal_moves = (
                self.legal_moves_history[i]
                if i < len(self.legal_moves_history)
                else None
            )
            next_legal_moves = (
                self.legal_moves_history[i + 1]
                if i + 1 < len(self.legal_moves_history)
                else None
            )
            yield Transition(
                observation=self.observation_history[i],
                action=self.action_history[i],
                reward=float(self.rewards[i]) if self.rewards else 0.0,
                next_observation=self.observation_history[i + 1],
                done=terminated or truncated,
                terminated=terminated,
                truncated=truncated,
                legal_moves=legal_moves,
                next_legal_moves=next_legal_moves,
            )
