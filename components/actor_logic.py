import torch
import time
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING
from core import PipelineComponent, Blackboard
from actors.action_selectors.types import InferenceResult

if TYPE_CHECKING:
    from modules.agent_nets.base import BaseAgentNetwork
    from utils.schedule import Schedule


class NetworkInferenceComponent(PipelineComponent):
    """
    Performs neural network inference for an actor.
    Reads 'observations' from data, writes 'inference_result' to predictions.
    """

    def __init__(self, agent_network: "BaseAgentNetwork", input_shape: Tuple[int, ...]):
        self.agent_network = agent_network
        self.input_shape = input_shape

    def execute(self, blackboard: Blackboard) -> None:
        obs = blackboard.data["observations"]

        # Ensure batch dimension [1, ...] if single observation
        if obs.dim() == len(self.input_shape):
            obs = obs.unsqueeze(0)

        with torch.inference_mode():
            output = self.agent_network.obs_inference(obs)
            result = InferenceResult.from_inference_output(output)

        blackboard.predictions["inference_result"] = result


class SearchInferenceComponent(PipelineComponent):
    """
    Performs MCTS-based search inference for an actor.
    """

    def __init__(
        self,
        search_engine: Any,
        agent_network: Optional["BaseAgentNetwork"],
        input_shape: Tuple[int, ...],
        num_actions: int,
        exploration: bool = True,
    ):
        self.search = search_engine
        self.agent_network = agent_network
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.exploration = exploration

    def execute(self, blackboard: Blackboard) -> None:
        obs = blackboard.data["observations"]
        info = blackboard.meta.get("info", {})

        # Extract player info for Multi-agent environments
        # MCTS traditionally expects 'player' to be in info
        if "player" not in info:
            # Try to find 'to_play' in meta if PettingZooAECComponent was used
            if "to_play" in blackboard.meta:
                info = {**info, "player": blackboard.meta["to_play"]}

        start_time = time.time()

        # Handle batched vs non-batched search
        is_batched = obs.dim() > len(self.input_shape) and obs.shape[0] > 1

        with torch.inference_mode():
            if is_batched:
                res = self.search.run_vectorized(obs, info, self.agent_network)
                (
                    root_values,
                    exploratory_policies,
                    target_policies,
                    best_actions,
                    sm_list,
                ) = res

                probs = torch.stack(
                    [
                        torch.as_tensor(p, device=obs.device, dtype=torch.float32)
                        for p in exploratory_policies
                    ]
                )
                values = torch.as_tensor(
                    root_values, device=obs.device, dtype=torch.float32
                )
                if values.dim() == 1:
                    values = values.unsqueeze(-1)

                result = InferenceResult(
                    probs=probs,
                    value=values,
                    extra_metadata={
                        "target_policies": torch.stack(
                            [
                                torch.as_tensor(
                                    p, device=obs.device, dtype=torch.float32
                                )
                                for p in target_policies
                            ]
                        ),
                        "search_duration": time.time() - start_time,
                        "search_metadata": sm_list,
                        "best_actions": torch.as_tensor(
                            best_actions, device=obs.device, dtype=torch.long
                        ),
                        "value": values.squeeze(-1),
                        "root_value": values.squeeze(-1),
                    },
                )
            else:
                res = self.search.run(
                    obs, info, self.agent_network, exploration=self.exploration
                )
                (
                    root_value,
                    exploratory_policy,
                    target_policy,
                    best_action,
                    search_metadata,
                ) = res

                probs = exploratory_policy.to(obs.device)
                value = torch.tensor(
                    [root_value], device=obs.device, dtype=torch.float32
                )

                if obs.dim() > len(self.input_shape):
                    probs = probs.unsqueeze(0)
                    target_policies_out = target_policy.unsqueeze(0).to(obs.device)
                    best_actions_out = torch.tensor([best_action], device=obs.device)
                else:
                    target_policies_out = target_policy.to(obs.device)
                    best_actions_out = torch.tensor(best_action, device=obs.device)

                result = InferenceResult(
                    probs=probs,
                    value=value,
                    extra_metadata={
                        "target_policies": target_policies_out,
                        "search_duration": time.time() - start_time,
                        "search_metadata": search_metadata,
                        "best_actions": best_actions_out,
                        "value": value.squeeze(0),
                        "root_value": value.squeeze(0),
                    },
                )

        blackboard.predictions["inference_result"] = result


class BaseSelectorComponent(PipelineComponent):
    """
    Base class for selection components that read from 'inference_result'
    and write to 'meta["action"]', 'meta["action_metadata"]'.
    """

    def mask_actions(
        self,
        values: torch.Tensor,
        legal_moves: Any,
        mask_value: float = -float("inf"),
    ) -> torch.Tensor:
        mask = torch.zeros_like(values, dtype=torch.bool)
        device = values.device
        mask = mask.to(device)

        if values.dim() == 1:
            mask[legal_moves] = True
        elif values.dim() == 2:
            # detect if legal_moves is a single mask for a batch-of-1,
            # or a batch of masks.
            is_batched = True
            if torch.is_tensor(legal_moves) and legal_moves.dim() == 1:
                is_batched = False
            elif isinstance(legal_moves, (list, tuple)) and len(legal_moves) > 0:
                if not isinstance(legal_moves[0], (list, tuple, torch.Tensor)):
                    is_batched = False

            if not is_batched and values.shape[0] == 1:
                mask[0, legal_moves] = True
            else:
                for i, legal in enumerate(legal_moves):
                    if legal is not None and i < mask.shape[0]:
                        mask[i, legal] = True

        return torch.where(mask, values, torch.tensor(mask_value, device=device))

    def write_to_blackboard(
        self, blackboard: Blackboard, action: torch.Tensor, metadata: Dict[str, Any]
    ) -> None:
        result = blackboard.predictions["inference_result"]

        # Merge extra metadata from search/network
        if result.extra_metadata:
            for k, v in result.extra_metadata.items():
                if k not in metadata:
                    metadata[k] = v

        # Squeeze for single-actor consistency
        if action.dim() > 0 and action.shape[0] == 1:
            raw_action = action.item()
            sq_action = action.squeeze(0)
        else:
            raw_action = action.item() if action.numel() == 1 else action.cpu().numpy()
            sq_action = action

        for k, v in metadata.items():
            if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == 1:
                metadata[k] = v.squeeze(0)

        blackboard.meta["action"] = raw_action
        blackboard.meta["action_tensor"] = sq_action
        blackboard.meta["action_metadata"] = metadata


class CategoricalSelectorComponent(BaseSelectorComponent):
    """
    Selects action by sampling from a Categorical distribution.
    """

    def __init__(self, exploration: bool = True):
        self.exploration = exploration

    def execute(self, blackboard: Blackboard) -> None:
        result = blackboard.predictions["inference_result"]
        info = blackboard.meta.get("info", {})
        mask = info.get("legal_moves_mask", info.get("legal_moves"))

        from torch.distributions import Categorical

        if result.logits is not None:
            logits = result.logits
            if mask is not None:
                logits = self.mask_actions(logits, mask)
            dist = Categorical(logits=logits)
        else:
            probs = result.probs
            if mask is not None:
                # Assuming mask is a binary mask of same shape
                probs = probs * mask.float()
                probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            dist = Categorical(probs=probs)

        if self.exploration:
            action = dist.sample()
        else:
            action = torch.argmax(dist.probs, dim=-1)

        metadata = {
            "policy_dist": dist,
            "policy": dist.probs.detach().cpu(),
            "value": result.value.detach().cpu() if result.value is not None else None,
        }

        self.write_to_blackboard(blackboard, action, metadata)


class ArgmaxSelectorComponent(BaseSelectorComponent):
    """
    Selects the action with the highest value/logit (Greedy).
    """

    def execute(self, blackboard: Blackboard) -> None:
        result = blackboard.predictions["inference_result"]
        info = blackboard.meta.get("info", {})
        mask = info.get("legal_moves_mask", info.get("legal_moves"))

        values = (
            result.q_values
            if result.q_values is not None
            else (result.logits if result.logits is not None else result.probs)
        )

        if mask is not None:
            values = self.mask_actions(values, mask)

        action = torch.argmax(values, dim=-1)

        metadata = {
            "value": result.value.detach().cpu() if result.value is not None else None,
            "policy": result.probs.detach().cpu() if result.probs is not None else None,
        }

        self.write_to_blackboard(blackboard, action, metadata)


class PPODecoratorComponent(PipelineComponent):
    """
    Sequential decorator that adds log_prob to the action_metadata.
    MUST be placed after a selector that provides 'policy_dist' in metadata.
    """

    def execute(self, blackboard: Blackboard) -> None:
        action = blackboard.meta["action_tensor"]
        metadata = blackboard.meta["action_metadata"]

        dist = metadata.get("policy_dist")
        if dist is not None:
            # We assume action and dist are compatible in shape (handled by selector)
            metadata["log_prob"] = dist.log_prob(action).detach().cpu()

        # 'value' is already added by the selectors from result.value


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
