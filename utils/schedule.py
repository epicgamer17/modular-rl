import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class ScheduleConfig:
    type: str = "constant"
    initial: Optional[float] = None
    final: Optional[float] = None
    decay_steps: Optional[int] = None
    steps: Optional[List[int]] = None
    values: Optional[List[float]] = None
    period: Optional[int] = None
    with_training_steps: bool = False

    def __post_init__(self):
        if self.type == "stepwise":
            if self.steps is None or self.values is None:
                raise ValueError("stepwise schedule requires 'steps' and 'values'")
            if len(self.steps) + 1 != len(self.values):
                raise ValueError("'values' must have one more element than 'steps'")
        elif self.type == "linear":
            if self.initial is None or self.final is None or self.decay_steps is None:
                raise ValueError(
                    "linear schedule requires 'initial', 'final', and 'decay_steps'"
                )
        elif self.type == "inverse_sqrt":
            if self.initial is None:
                raise ValueError("inverse_sqrt schedule requires 'initial'")
        elif self.type == "cyclical":
            if self.initial is None or self.final is None or self.period is None:
                raise ValueError(
                    "cyclical schedule requires 'initial', 'final', and 'period'"
                )

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "ScheduleConfig":
        if d is None:
            return cls(type="constant")
        if isinstance(d, cls):
            return d
        return cls(
            type=d.get("type", "constant"),
            initial=d.get("initial"),
            final=d.get("final"),
            decay_steps=d.get("decay_steps"),
            steps=d.get("steps"),
            values=d.get("values"),
            period=d.get("period"),
            with_training_steps=d.get("with_training_steps", False),
        )

    @classmethod
    def constant(cls, value: float) -> "ScheduleConfig":
        return cls(type="constant", initial=value, final=value)

    @classmethod
    def linear(cls, initial: float, final: float, decay_steps: int) -> "ScheduleConfig":
        return cls(type="linear", initial=initial, final=final, decay_steps=decay_steps)

    @classmethod
    def stepwise(cls, steps: List[int], values: List[float]) -> "ScheduleConfig":
        return cls(type="stepwise", steps=steps, values=values)

    @classmethod
    def inverse_sqrt(cls, initial: float) -> "ScheduleConfig":
        return cls(type="inverse_sqrt", initial=initial)


class Schedule(ABC):
    def __init__(self):
        self._step = 0

    @abstractmethod
    def get_value(self, step: Optional[int] = None) -> float:
        """
        Calculates the schedule value.
        If `step` is provided, calculates the value for that specific step (stateless).
        If `step` is None, uses the internal self._step counter (stateful).
        """
        pass

    @property
    @abstractmethod
    def initial_value(self) -> float:
        """Returns the initial value of the schedule."""
        pass

    def step(self, count: int = 1) -> None:
        self._step += count

    @property
    def step_count(self) -> int:
        return self._step

    def state_dict(self) -> Dict[str, Any]:
        return {"_step": self._step}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._step = state.get("_step", 0)

    def reset(self) -> None:
        self._step = 0


class ConstantSchedule(Schedule):
    def __init__(self, initial_value: float = 1.0):
        super().__init__()
        self._value = initial_value

    def get_value(self, step: Optional[int] = None) -> float:
        return self._value

    @property
    def initial_value(self) -> float:
        return self._value


class LinearSchedule(Schedule):
    def __init__(self, initial: float, final: float, decay_steps: int):
        super().__init__()
        self._initial = initial
        self._final = final
        self._decay_steps = decay_steps

    def get_value(self, step: Optional[int] = None) -> float:
        current_step = step if step is not None else self._step
        progress = min(1.0, current_step / self._decay_steps)
        value = self._initial + (self._final - self._initial) * progress
        if self._initial < self._final:
            return min(self._final, value)
        else:
            return max(self._final, value)

    @property
    def initial_value(self) -> float:
        return self._initial


class StepwiseSchedule(Schedule):
    def __init__(self, steps: List[int], values: List[float]):
        super().__init__()
        self._steps = steps
        self._values = values

    def get_value(self, step: Optional[int] = None) -> float:
        current_step = step if step is not None else self._step
        value = self._values[0]
        for i, s in enumerate(self._steps):
            if current_step >= s:
                value = self._values[i + 1]
        return value

    @property
    def initial_value(self) -> float:
        return self._values[0]


class InverseSqrtSchedule(Schedule):
    def __init__(self, initial: float):
        super().__init__()
        self._initial = initial

    def get_value(self, step: Optional[int] = None) -> float:
        current_step = step if step is not None else self._step
        return self._initial / math.sqrt(current_step + 1)

    @property
    def initial_value(self) -> float:
        return self._initial


class CyclicalSchedule(Schedule):
    def __init__(self, initial: float, final: float, period: int):
        super().__init__()
        self._initial = initial
        self._final = final
        self._period = period

    def get_value(self, step: Optional[int] = None) -> float:
        current_step = step if step is not None else self._step

        cycle_progress = (current_step % self._period) / self._period
        half_period = self._period / 2

        if (current_step % self._period) < half_period:
            progress = (current_step % self._period) / half_period
            return self._initial + (self._final - self._initial) * progress
        else:
            progress = ((current_step % self._period) - half_period) / half_period
            return self._final - (self._final - self._initial) * progress

    @property
    def initial_value(self) -> float:
        return self._initial


def create_schedule(config: Optional[ScheduleConfig]) -> Schedule:
    if config is None:
        return ConstantSchedule(initial_value=1.0)

    schedule_type = config.type

    if schedule_type == "constant":
        return ConstantSchedule(initial_value=config.initial if config.initial is not None else 1.0)
    elif schedule_type == "linear":
        return LinearSchedule(
            initial=config.initial, final=config.final, decay_steps=config.decay_steps
        )
    elif schedule_type == "stepwise":
        return StepwiseSchedule(steps=config.steps, values=config.values)
    elif schedule_type == "inverse_sqrt":
        return InverseSqrtSchedule(initial=config.initial)
    elif schedule_type == "cyclical":
        return CyclicalSchedule(
            initial=config.initial, final=config.final, period=config.period
        )
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
