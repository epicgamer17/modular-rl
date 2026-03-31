"""
OldNetworkBridge: Wraps the OLD ModularAgentNetwork to work with NEW training
infrastructure (new learner, new buffer, new executor, new actors).

The OLD network's weights and forward pass are used exactly as-is.
Only the interface is adapted to match what new code expects.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass, field

from modules.agent_nets.modular import (
    ModularAgentNetwork as OldModularAgentNetwork,
)
from modules.world_models.inference_output import (
    MuZeroNetworkState,
    InferenceOutput as OldInferenceOutput,
)


@dataclass
class InferenceOutput:
    """New-style InferenceOutput for compatibility with new search/actors."""

    recurrent_state: Dict[str, Tensor] = field(default_factory=dict)
    value: Optional[Tensor] = None
    policy: Any = None  # Distribution
    probs: Optional[Tensor] = None
    action: Optional[Tensor] = None
    q_values: Optional[Tensor] = None
    reward: Optional[Tensor] = None
    to_play: Optional[Tensor] = None
    chance: Any = None
    extras: Dict[str, Any] = field(default_factory=dict)


class _FakeRepresentation:
    """Stub so loss pipeline can read head.representation without crashing."""

    def __init__(self, head_module):
        # Try to find the real representation on the old head
        self._rep = getattr(head_module, "representation", None)

    def to_expected_value(self, logits):
        if self._rep is not None:
            return self._rep.to_expected_value(logits)
        return logits

    def transform_target(self, target):
        if self._rep is not None:
            return self._rep.transform_target(target)
        return target


class _FakeHead(nn.Module):
    """Placeholder so agent_network.components['behavior_heads'][name].representation works."""

    def __init__(self, representation):
        super().__init__()
        self.representation = representation
        self.input_source = "default"


class OldNetworkBridge(nn.Module):
    """Wraps OLD ModularAgentNetwork with the interface the NEW infra expects.

    The new learner, actors, search, and executor all talk to this bridge.
    The OLD network does all the actual computation.
    """

    def __init__(self, old_network: OldModularAgentNetwork, config: Any):
        super().__init__()
        self.old_net = old_network
        self.config = config
        self.input_shape = old_network.input_shape
        self.num_actions = old_network.num_actions
        self.num_players = getattr(config.game, "num_players", 1)

        # Register as submodule so parameters() works
        self.components = nn.ModuleDict()

        # The new loss pipeline reads representations from:
        #   agent_network.components["behavior_heads"]["state_value"].representation
        #   agent_network.components["behavior_heads"]["policy_logits"].representation
        #   agent_network.components["world_model"].heads["reward_logits"].representation
        #   agent_network.components["world_model"].heads["to_play_logits"].representation
        # We create stubs that delegate to the old heads.

        old_wm = old_network.components.get("world_model")
        old_val = old_network.components.get("value_head")
        old_pol = old_network.components.get("policy_head")

        behavior_heads = nn.ModuleDict()
        behavior_heads["state_value"] = _FakeHead(
            _FakeRepresentation(old_val) if old_val else _FakeRepresentation(None)
        )
        behavior_heads["policy_logits"] = _FakeHead(
            _FakeRepresentation(old_pol) if old_pol else _FakeRepresentation(None)
        )
        self.components["behavior_heads"] = behavior_heads

        # World model heads stub
        if old_wm is not None:
            wm_heads = nn.ModuleDict()
            wm_heads["reward_logits"] = _FakeHead(
                _FakeRepresentation(old_wm.reward_head)
            )
            if hasattr(old_wm, "to_play_head"):
                wm_heads["to_play_logits"] = _FakeHead(
                    _FakeRepresentation(old_wm.to_play_head)
                )
            # Expose as a module with .heads attribute
            self.components["world_model"] = _WorldModelStub(wm_heads)

        # Device indicator for AgentNetwork.device property
        self.register_buffer("_device_indicator", torch.zeros(1), persistent=False)

    @property
    def device(self) -> torch.device:
        return self._device_indicator.device

    def parameters(self, recurse=True):
        """Delegate to old network so optimizer sees all weights."""
        return self.old_net.parameters(recurse=recurse)

    def named_parameters(self, prefix="", recurse=True):
        return self.old_net.named_parameters(prefix=prefix, recurse=recurse)

    def state_dict(self, *args, **kwargs):
        return self.old_net.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        return self.old_net.load_state_dict(state_dict, strict=strict)

    def train(self, mode=True):
        self.old_net.train(mode)
        return super().train(mode)

    def eval(self):
        self.old_net.eval()
        return super().eval()

    def share_memory(self):
        self.old_net.share_memory()

    def initialize(self, initializer=None):
        if hasattr(self.old_net, "initialize"):
            self.old_net.initialize(initializer)

    def reset_noise(self):
        self.old_net.reset_noise()

    def to(self, *args, **kwargs):
        self.old_net.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    # ================================================================
    # NEW ACTOR / SEARCH API
    # ================================================================

    def obs_inference(
        self, obs: Tensor, recurrent_state=None, **kwargs
    ) -> InferenceOutput:
        """Called by new search engine and new actors for root inference."""
        old_out: OldInferenceOutput = self.old_net.obs_inference(obs)

        # Convert old MuZeroNetworkState to flat dict
        rs = {}
        if old_out.network_state is not None:
            ns = old_out.network_state
            if hasattr(ns, "dynamics") and ns.dynamics is not None:
                rs["dynamics"] = ns.dynamics
            if hasattr(ns, "wm_memory") and ns.wm_memory is not None:
                if isinstance(ns.wm_memory, dict):
                    rs.update(ns.wm_memory)

        probs = None
        if old_out.policy is not None and hasattr(old_out.policy, "probs"):
            probs = old_out.policy.probs

        return InferenceOutput(
            recurrent_state=rs,
            value=old_out.value,
            policy=old_out.policy,
            probs=probs,
            reward=old_out.reward,
            to_play=old_out.to_play,
            extras=old_out.extras or {},
        )

    def hidden_state_inference(
        self, network_state: Dict[str, Tensor], action: Tensor, **kwargs
    ) -> InferenceOutput:
        """Called by new search for dynamics stepping."""
        # Reconstruct old MuZeroNetworkState from flat dict
        old_ns = MuZeroNetworkState(
            dynamics=network_state.get("dynamics"),
            wm_memory=network_state.get("wm_memory"),
        )

        old_out: OldInferenceOutput = self.old_net.hidden_state_inference(
            old_ns, action
        )

        rs = {}
        if old_out.network_state is not None:
            ns = old_out.network_state
            if hasattr(ns, "dynamics") and ns.dynamics is not None:
                rs["dynamics"] = ns.dynamics
            if hasattr(ns, "wm_memory") and ns.wm_memory is not None:
                if isinstance(ns.wm_memory, dict):
                    rs.update(ns.wm_memory)

        probs = None
        if old_out.policy is not None and hasattr(old_out.policy, "probs"):
            probs = old_out.policy.probs

        return InferenceOutput(
            recurrent_state=rs,
            value=old_out.value,
            policy=old_out.policy,
            probs=probs,
            reward=old_out.reward,
            to_play=old_out.to_play,
            extras=old_out.extras or {},
        )

    # ================================================================
    # NEW LEARNER API
    # ================================================================

    def learner_inference(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Called by new UniversalLearner. Returns dict with new-style keys."""
        old_out = self.old_net.learner_inference(batch)

        # Old keys: values, policies, rewards, to_plays, latents
        # New keys: state_value, policy_logits, reward_logits, to_play_logits, latents
        result = {
            "state_value": old_out["values"],
            "policy_logits": old_out["policies"],
            "reward_logits": old_out.get("rewards"),
            "latents": old_out.get("latents"),
        }

        # to_play_logits
        tp = old_out.get("to_plays")
        if tp is not None:
            result["to_play_logits"] = tp

        # Pass through any extras
        for key in [
            "latents_afterstates",
            "chance_logits",
            "chance_values",
            "chance_encoder_embeddings",
        ]:
            if key in old_out:
                result[key] = old_out[key]

        return result


class _WorldModelStub(nn.Module):
    """Stub so bridge.components['world_model'].heads works."""

    def __init__(self, heads: nn.ModuleDict):
        super().__init__()
        self.heads = heads
