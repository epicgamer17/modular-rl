import torch
import time
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from core import PipelineComponent, Blackboard
from actors.action_selectors.types import InferenceResult

if TYPE_CHECKING:
    from modules.agent_nets.base import BaseAgentNetwork
    from utils.schedule import Schedule


class NetworkInferenceComponent(PipelineComponent):
    """
    Performs neural network inference for an actor.
    Reads 'obs' from data, writes results to the specified output key in predictions.
    """

    def __init__(
        self,
        agent_network: "BaseAgentNetwork",
        input_shape: Tuple[int, ...],
        output_key: str = "inference_result",
    ):
        self.agent_network = agent_network
        self.input_shape = input_shape
        self.output_key = output_key

    def execute(self, blackboard: Blackboard) -> None:
        obs = blackboard.data["obs"]
        if obs is None:
            return

        # Ensure batch dimension [1, ...] if single observation
        if obs.dim() == len(self.input_shape):
            obs = obs.unsqueeze(0)

        with torch.inference_mode():
            output = self.agent_network.obs_inference(obs)
            result = InferenceResult.from_inference_output(output)

        blackboard.predictions[self.output_key] = result
        # Also set default keys for single-inference components
        if self.output_key == "inference_result":
            blackboard.predictions["logits"] = result.logits
            blackboard.predictions["value"] = result.value





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
        obs = blackboard.data["obs"]
        done = blackboard.data.get("done", False)
        if obs is None or done:
            return
        info = blackboard.data.get("info", {})
        
        # Determine player_id/to_play
        # PettingZooObservationComponent provides 'player_id'
        player_id = blackboard.data.get("player_id", 0)
        if "player" not in info:
            info = {**info, "player": player_id}

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

        blackboard.predictions["logits"] = result.logits
        blackboard.predictions["value"] = result.value
        blackboard.predictions["inference_result"] = result


class BaseSelectorComponent(PipelineComponent):
    """
    Base class for selection components that read from 'inference_result'
    and write to 'meta["action"]', 'meta["action_metadata"]'.
    """

    def mask_actions(
        self,
        values: torch.Tensor,
        info: Dict[str, Any],
        mask_value: float = -float("inf"),
    ) -> torch.Tensor:
        # Extract mask from various potential keys
        legal_moves = info.get("legal_moves")
        if legal_moves is None:
            mask = info.get("action_mask", info.get("legal_moves_mask"))
            if mask is not None:
                if isinstance(mask, np.ndarray):
                    if mask.dtype == bool or (mask.ndim == 1 and np.all((mask == 0) | (mask == 1))):
                        # Convert boolean/binary mask to indices
                        legal_moves = np.where(mask)[0].tolist()
                    else:
                        legal_moves = mask.tolist()
                elif torch.is_tensor(mask):
                    if mask.dtype == torch.bool:
                        legal_moves = torch.where(mask)[0].cpu().tolist()
                    else:
                        legal_moves = mask.cpu().tolist()
                elif isinstance(mask, (list, tuple)) and len(mask) > 0:
                     # If it's already a list of indices or bools, use as is or convert
                     if isinstance(mask[0], (bool, np.bool_)):
                         legal_moves = [i for i, m in enumerate(mask) if m]
                     else:
                         legal_moves = mask

        if legal_moves is None:
            return values

        mask_tensor = torch.zeros_like(values, dtype=torch.bool)
        device = values.device
        mask_tensor = mask_tensor.to(device)

        if values.dim() == 1:
            mask_tensor[legal_moves] = True
        elif values.dim() == 2:
            # detect if legal_moves is a single mask for a batch-of-1,
            # or a batch of masks.
            is_batched = True
            if torch.is_tensor(legal_moves) and legal_moves.dim() == 1:
                is_batched = False
            elif isinstance(legal_moves, (list, tuple)) and len(legal_moves) > 0:
                if not isinstance(legal_moves[0], (list, tuple, torch.Tensor, np.ndarray)):
                    is_batched = False

            if not is_batched and values.shape[0] == 1:
                mask_tensor[0, legal_moves] = True
            else:
                for i, legal in enumerate(legal_moves):
                    if legal is not None and i < mask_tensor.shape[0]:
                        mask_tensor[i, legal] = True

        return torch.where(mask_tensor, values, torch.tensor(mask_value, device=device))

    def write_to_blackboard(
        self, blackboard: Blackboard, action: torch.Tensor, metadata: Dict[str, Any]
    ) -> None:
        # Use .get() to avoid KeyError if inference was skipped
        result = blackboard.predictions.get("inference_result")

        # Merge extra metadata from search/network
        if result is not None and result.extra_metadata:
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
        # Skip if agent is already done
        if blackboard.data.get("done", False):
            self.write_to_blackboard(blackboard, torch.tensor([0]), {"action_is_none": True}) # Dummy, step handles None
            blackboard.meta["action"] = None
            return

        result = blackboard.predictions.get("inference_result")
        info = blackboard.meta.get("info", blackboard.data.get("info", {}))

        from torch.distributions import Categorical

        if result is not None:
            if result.logits is not None:
                logits = result.logits
                logits = self.mask_actions(logits, info)
                dist = Categorical(logits=logits)
            else:
                # Manual re-masking for probs
                probs = result.probs
                # Extract mask via base helper
                # We reuse mask_actions logic by creating a dummy one
                # If no legal moves indices were found, we want mask_tensor to be ALL TRUE
                dummy_masked = self.mask_actions(torch.ones_like(probs), info, mask_value=0.0)
                mask_tensor = (dummy_masked == 1.0)
                
                probs = probs * mask_tensor.float()
                probs_sum = probs.sum(dim=-1, keepdim=True)
                
                # Check for NaNs or all-zeros
                if torch.isnan(probs).any() or (probs_sum < 1e-9).any():
                    # Fallback to uniform over legal moves
                    valid_mask = mask_tensor.float()
                    if valid_mask.sum() < 1e-9:
                         valid_mask = torch.ones_like(probs) # Ultimate fallback
                    probs = valid_mask / valid_mask.sum(dim=-1, keepdim=True)
                else:
                    probs = probs / probs_sum
                
                dist = Categorical(probs=probs)
            
            value = result.value
        else:
            # Try direct keys
            logits = blackboard.predictions.get("logits")
            value = blackboard.predictions.get("value")
            probs = blackboard.predictions.get("probs")
            
            if logits is not None:
                logits = self.mask_actions(logits, info)
                dist = Categorical(logits=logits)
            elif probs is not None:
                dist = Categorical(probs=probs)
            else:
                # Fallback to None if no policy info at all
                blackboard.meta["action"] = None
                return

        if self.exploration:
            action = dist.sample()
        else:
            action = torch.argmax(dist.probs, dim=-1)

        metadata = {
            "policy_dist": dist,
            "policy": dist.probs.detach().cpu(),
            "value": value.detach().cpu() if value is not None else None,
        }

        self.write_to_blackboard(blackboard, action, metadata)


class ArgmaxSelectorComponent(BaseSelectorComponent):
    """
    Selects the action with the highest value/logit (Greedy).
    """

    def execute(self, blackboard: Blackboard) -> None:
        # Skip if agent is already done
        if blackboard.data.get("done", False):
            self.write_to_blackboard(blackboard, torch.tensor([0]), {"action_is_none": True})
            blackboard.meta["action"] = None
            return

        result = blackboard.predictions.get("inference_result")
        info = blackboard.meta.get("info", blackboard.data.get("info", {}))

        if result is not None:
            values = (
                result.q_values
                if result.q_values is not None
                else (result.logits if result.logits is not None else result.probs)
            )
            value = result.value
            probs = result.probs
        else:
            # Try direct keys
            values = blackboard.predictions.get("logits", blackboard.predictions.get("probs"))
            value = blackboard.predictions.get("value")
            probs = blackboard.predictions.get("probs")

        values = self.mask_actions(values, info)

        action = torch.argmax(values, dim=-1)

        metadata = {
            "value": value.detach().cpu() if value is not None else None,
            "policy": probs.detach().cpu() if probs is not None else None,
        }

        self.write_to_blackboard(blackboard, action, metadata)



class EpsilonGreedySelectorComponent(BaseSelectorComponent):
    """
    Selects action using epsilon-greedy strategy.
    """

    def __init__(self, epsilon: float = 0.05):
        self.epsilon = epsilon

    def execute(self, blackboard: Blackboard) -> None:
        # Skip if agent is already done
        if blackboard.data.get("done", False):
            self.write_to_blackboard(blackboard, torch.tensor([0]), {"action_is_none": True})
            blackboard.meta["action"] = None
            return

        result = blackboard.predictions.get("inference_result")
        info = blackboard.meta.get("info", blackboard.data.get("info", {}))

        if result is not None:
            q_values = result.q_values
            probs = result.probs
        else:
            q_values = blackboard.predictions.get("q_values")
            probs = blackboard.predictions.get("probs")

        if q_values.dim() == 2:
            batch_size = q_values.shape[0]
        else:
            batch_size = 1

        q_values = self.mask_actions(q_values, info)

        # Update epsilon from metadata if available
        epsilon = blackboard.meta.get("epsilon", self.epsilon)

        r = torch.rand(batch_size, device=q_values.device)
        explore_mask = r < epsilon

        if batch_size == 1:
            if explore_mask.item():
                # Random action from legal moves
                legal_moves_dummy = self.mask_actions(torch.zeros_like(q_values), info, mask_value=1.0)
                legal_indices = torch.where(legal_moves_dummy == 1.0)[1] if q_values.dim() == 2 else torch.where(legal_moves_dummy == 1.0)[0]
                
                if len(legal_indices) > 0:
                    idx = torch.randint(0, len(legal_indices), (1,), device=q_values.device)
                    action = legal_indices[idx]
                else:
                    action = torch.argmax(q_values, dim=-1) # Fallback
            else:
                action = torch.argmax(q_values, dim=-1)
        else:
            # Batched version (simplified)
            greedy_actions = torch.argmax(q_values, dim=-1)
            # For simplicity, we just use greedy for now in batch > 1 if not easily vectorized
            # or we do it properly
            actions = greedy_actions # TODO: implement batched random actions if needed
            action = actions

        metadata = {
            "value": q_values.max(dim=-1)[0].detach().cpu(),
            "policy": probs.detach().cpu() if probs is not None else None,
            "epsilon": epsilon,
        }

        self.write_to_blackboard(blackboard, action, metadata)


class NFSPSelectorComponent(BaseSelectorComponent):
    """
    Manages selection between Best Response and Average Strategy for NFSP.
    Expects two inference results in the blackboard.
    """

    def __init__(
        self,
        br_key: str = "br_inference",
        avg_key: str = "avg_inference",
        eta: float = 0.1,
    ):
        self.br_key = br_key
        self.avg_key = avg_key
        self.eta = eta
        self.br_selector = ArgmaxSelectorComponent()
        self.avg_selector = ArgmaxSelectorComponent()

    def execute(self, blackboard: Blackboard) -> None:
        import random

        eta = blackboard.meta.get("eta", self.eta)

        if random.random() < eta:
            blackboard.meta["policy_used"] = "best_response"
            # Swap inference_result to BR for the selector
            old_result = blackboard.predictions.get("inference_result")
            blackboard.predictions["inference_result"] = blackboard.predictions[
                self.br_key
            ]
            self.br_selector.execute(blackboard)
            blackboard.predictions["inference_result"] = old_result
        else:
            blackboard.meta["policy_used"] = "average_strategy"
            # Swap inference_result to AVG for the selector
            old_result = blackboard.predictions.get("inference_result")
            blackboard.predictions["inference_result"] = blackboard.predictions[
                self.avg_key
            ]
            self.avg_selector.execute(blackboard)
            blackboard.predictions["inference_result"] = old_result


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
