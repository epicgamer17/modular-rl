import torch
from typing import TYPE_CHECKING
from core import PipelineComponent, Blackboard

if TYPE_CHECKING:
    from utils.schedule import Schedule


class TemperatureComponent(PipelineComponent):
    """
    Pure temperature scaling component.
    Modifies inference_result.logits in predictions.
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def execute(self, blackboard: Blackboard) -> None:
        if self.temperature == 1.0:
            return

        result = blackboard.predictions["inference_result"]

        # Ensure we have logits
        if result.logits is None:
            if result.probs is not None:
                result.logits = torch.log(result.probs + 1e-8)
            elif result.q_values is not None:
                result.logits = result.q_values
            else:
                return

        if self.temperature == 0.0:
            # Collapses to argmax
            best = torch.argmax(result.logits, dim=-1)
            result.logits = torch.full_like(result.logits, -float("inf"))
            result.logits.scatter_(-1, best.unsqueeze(-1), 0.0)
        else:
            result.logits = result.logits / self.temperature

        # Clear probs so selector is forced to use heat-treated logits
        result.probs = None


class EpisodeTemperatureComponent(TemperatureComponent):
    """
    Temperature component that uses a schedule based on episode steps.
    """

    def __init__(self, schedule: "Schedule"):
        super().__init__(temperature=1.0)
        self.schedule = schedule
        self._last_episode_step = -1

    def execute(self, blackboard: Blackboard) -> None:
        info = blackboard.meta.get("info", {})
        # Episode step tracking depends on environment/wrapper providing it
        # or we can track it here if we know when reset() happens.
        # Typically info['episode_step']
        step = info.get("episode_step", 0)

        if step != self._last_episode_step:
            self.schedule.step(
                max(1, step - self._last_episode_step)
                if self._last_episode_step >= 0
                else step
            )
            self._last_episode_step = step
            if step == 0:
                self.schedule.reset()

        self.temperature = self.schedule.get_value()
        super().execute(blackboard)


class TrainingStepTemperatureComponent(TemperatureComponent):
    """
    Temperature component that uses a schedule based on global training steps.
    """

    def __init__(self, schedule: "Schedule"):
        super().__init__(temperature=1.0)
        self.schedule = schedule
        self._last_training_step = -1

    def execute(self, blackboard: Blackboard) -> None:
        # Training step is often broadcasted via blackboard.meta or results
        step = blackboard.meta.get("training_step", 0)

        if step > self._last_training_step:
            self.schedule.step(step - self._last_training_step)
            self._last_training_step = step

        self.temperature = self.schedule.get_value()
        super().execute(blackboard)
