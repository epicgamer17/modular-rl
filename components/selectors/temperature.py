import torch
from typing import Optional, TYPE_CHECKING
from core import PipelineComponent, Blackboard

if TYPE_CHECKING:
    from utils.schedule import Schedule


class TemperatureComponent(PipelineComponent):
    """
    Temperature scaling component for inference logits.

    Three modes:
    - Fixed: pass a static ``temperature`` float (default 1.0).
    - Episode-scheduled: pass a ``Schedule`` + ``schedule_source="episode"``.
      Temperature updates based on ``info["episode_step"]``.
    - Training-scheduled: pass a ``Schedule`` + ``schedule_source="training"``.
      Temperature updates based on ``blackboard.meta["training_step"]``.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        schedule: Optional["Schedule"] = None,
        schedule_source: str = "episode",
    ):
        self.temperature = temperature
        self.schedule = schedule
        self.schedule_source = schedule_source
        self._last_step = -1

    def execute(self, blackboard: Blackboard) -> None:
        # Update temperature from schedule if one is provided
        if self.schedule is not None:
            if self.schedule_source == "episode":
                self._update_from_episode(blackboard)
            elif self.schedule_source == "training":
                self._update_from_training(blackboard)

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

    def _update_from_episode(self, blackboard: Blackboard) -> None:
        """Update temperature from episode step schedule."""
        info = blackboard.meta.get("info", {})
        step = info.get("episode_step", 0)

        if step != self._last_step:
            self.schedule.step(
                max(1, step - self._last_step)
                if self._last_step >= 0
                else step
            )
            self._last_step = step
            if step == 0:
                self.schedule.reset()

        self.temperature = self.schedule.get_value()

    def _update_from_training(self, blackboard: Blackboard) -> None:
        """Update temperature from global training step."""
        step = blackboard.meta.get("training_step", 0)

        if step > self._last_step:
            self.schedule.step(step - self._last_step)
            self._last_step = step

        self.temperature = self.schedule.get_value()
