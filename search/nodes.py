from copy import deepcopy
from math import log, sqrt, inf
import copy
import math
import numpy as np
import torch
import torch.nn.functional as F


class ChanceNode:
    """
    Represents the Afterstate (s, a).
    According to the paper: Expanded by querying stochastic model.
    Model returns:
      1. Value (afterstate value)
      2. Prior distribution over codes P(c|as)
    """

    estimation_method = None
    discount = None
    value_prefix = None

    def __init__(self, prior, parent):
        # assert isinstance(parent, DecisionNode)
        self.parent = parent  # DecisionNode
        self.prior = prior  # P(a|s) from Policy

        self.visits = 0
        self.value_sum = 0.0

        # NEW: The Value of the afterstate predicted by the Dynamics Network
        self.network_value = None

        # NEW: The distribution P(c|as) predicted by the Dynamics Network
        self.code_probs = {}

        self.to_play = parent.to_play
        # Children are DecisionNodes, indexed by code
        self.children = {}
        self.network_state = None

        # Vectorized child stats
        self.child_visits = None
        self.child_values = None
        self.child_priors = None

        # Cache for v_mix
        self._v_mix = None

    def expand(
        self,
        to_play,
        network_state,
        network_value,
        code_probs,
    ):
        """
        Called when the Dynamics Network is run on (parent_state, action).
        """
        self.to_play = to_play
        self.network_state = network_state

        self.network_value = network_value
        # code_probs should be a dict or array mapping code_index -> probability
        num_chance = len(code_probs)
        self.code_probs = {
            # Warning non differentiable
            a: code_probs[a]
            for a in range(num_chance)
        }

        # Initialize vectorized stats
        self.child_priors = code_probs.detach().cpu()
        if self.child_priors.dim() == 0:
            self.child_priors = self.child_priors.unsqueeze(0)

        num_actions = self.child_priors.shape[0]
        self.child_visits = torch.zeros(num_actions, dtype=torch.float32)
        self.child_values = torch.zeros(num_actions, dtype=torch.float32)

        self._v_mix = None

        # Lazy child creation: don't create children yet

    def expanded(self):
        # We are expanded if we have populated our priors/value
        return self.child_priors is not None

    def value(self):
        """
        Returns Q(s,a). initial value set to zt
        If unvisited, use the Network's predicted Afterstate Value
        (Bootstrap from the dynamics model, as per the text).
        """
        if self.visits == 0:
            return self._get_bootstrap_value()
        return self.value_sum / self.visits

    def _get_bootstrap_value(self):
        """Helper to determine value when visits are 0 based on estimation method."""
        if self.estimation_method == "v_mix":
            value = self.parent.get_v_mix()
        elif self.estimation_method == "mcts_value":
            value = self.parent.value()
        elif self.estimation_method == "network_value":
            value = self.parent.network_value
        else:
            value = 0.0
        return value

    def _sample_code(self, codes, probs):
        """Helper to sample a single code index from probabilities."""
        # Normalize probs just in case
        probs = [(p.detach().item() if torch.is_tensor(p) else float(p)) for p in probs]
        probs = np.array(probs)
        probs /= probs.sum()
        return np.random.choice(len(codes), p=probs)

    def get_child(self, code):
        if code not in self.children:
            # Create lazy child
            p = self.child_priors[code]
            self.children[code] = DecisionNode(p.item(), self)
        return self.children[code]

    def child_reward(self, child):
        # assert isinstance(child, DecisionNode)
        if self.value_prefix:
            if child.is_reset:
                return child.reward
            else:
                return child.reward - self.parent.reward
        else:
            true_reward = child.reward

        assert true_reward is not None
        return true_reward

    def get_child_q_from_parent(self, child):
        # assert isinstance(child, DecisionNode)
        r = self.child_reward(child)
        if torch.is_tensor(r):
            r = r.detach().item()
        else:
            r = float(r)

        # child.value() if visited else v_mix
        v = child.value()
        if torch.is_tensor(v):
            v = v.detach().item()
        else:
            v = float(v)

        # sign = +1 if child.to_play == self.to_play else -1 (multi-agent).
        sign = 1.0 if child.to_play == self.to_play else -1.0
        q_from_parent = r + self.discount * (sign * v)

        assert q_from_parent is not None
        return q_from_parent

    def get_v_mix(self):
        if self._v_mix is not None:
            return self._v_mix

        # Vectorized v_mix
        # sum_N = visits.sum()
        sum_N = self.child_visits.sum()

        if sum_N > 0:
            term = self._calculate_visited_policy_mass(sum_N)
        else:
            term = 0.0

        v_mix = (self.value() + term) / (1.0 + sum_N)
        self._v_mix = v_mix
        return v_mix

    def _calculate_visited_policy_mass(self, sum_N):
        # Vectorized implementation for ChanceNode
        # Check if we have child stats populated
        if self.child_visits is None:
            return 0.0

        visited_mask = self.child_visits > 0
        if not visited_mask.any():
            return 0.0

        # For ChanceNodes, child_values should store the Values of the children (afterstates/states)
        # The equation for v_mix uses Q(s,a).
        # For ChanceNode (s, a_chance), the "actions" are codes.
        # The Q-value is just the value of the next state (child.value()).
        # So self.child_values should store these.

        q_vis = self.child_values[visited_mask]
        p_vis = self.child_priors[visited_mask]

        p_vis_sum = p_vis.sum()
        expected_q_vis = (p_vis * q_vis).sum()

        if p_vis_sum == 0:
            return 0.0

        term = sum_N * (expected_q_vis / p_vis_sum)
        return term


class DecisionNode:
    estimation_method = None
    discount = None
    value_prefix = None
    pb_c_init = None
    pb_c_base = None
    gumbel = None
    cvisit = None
    cscale = None
    stochastic = None

    def __init__(self, prior, parent=None):
        assert (
            (self.stochastic and isinstance(parent, ChanceNode))
            or (parent is None)
            or (not self.stochastic and isinstance(parent, DecisionNode))
        )
        self.visits = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.value_sum = 0
        self.children = {}
        self.network_state = None

        self.reward = 0
        self.parent = parent

        self.network_policy = None  # dense policy vector (numpy or torch)
        self.network_policy_dist = None  # optional full distribution for compatibility
        self.network_value = None  # network scalar value estimate (float)

        # Vectorized child stats
        self.child_visits = None
        self.child_values = None
        self.child_priors = None

    def expand(
        self,
        allowed_actions,
        to_play,
        priors,
        network_policy,
        network_state,
        reward,
        value=None,
        network_policy_dist=None,
    ):
        self.to_play = to_play
        self.reward = reward
        self.network_state = network_state

        self.network_policy = network_policy.detach().cpu()
        self.network_policy_dist = network_policy_dist
        self.network_value = value

        # Initialize vectorized stats
        num_actions = len(self.network_policy)

        self.child_visits = torch.zeros(num_actions, dtype=torch.float32)
        # Initialize values to 0? Or bootstrap?
        # Usually initialized to 0, and handling bootstrap in scoring.
        self.child_values = torch.zeros(num_actions, dtype=torch.float32)

        self._v_mix = None

        if priors is not None:
            self.child_priors = priors.detach().cpu()
        else:
            self.child_priors = self.network_policy

        # Apply allowed_actions mask if provided
        if allowed_actions is not None:
            # allowed_actions is a Tensor of indices
            # Create boolean mask
            mask = torch.zeros_like(self.child_priors, dtype=torch.bool)
            mask[allowed_actions] = True

            # Mask priors
            self.child_priors[~mask] = 0.0

            # Renormalize to ensure sum is 1 (avoiding div by zero)
            sum_priors = self.child_priors.sum()
            if sum_priors > 0:
                self.child_priors /= sum_priors
            else:
                # If sum is 0 (should imply all allowed had 0 prior), uniformly distribute
                self.child_priors[mask] = 1.0 / mask.sum()

        # Lazy child creation: do NOT populate children dict yet.

    def _populate_children(self, allowed_actions, priors):
        pass  # Deprecated by vectorized stats

    def expanded(self):
        # assert (len(self.children) > 0) == (self.visits > 0)
        return self.child_priors is not None

    def value(self):
        if self.visits == 0:
            return self._get_bootstrap_value()
        else:
            value = self.value_sum / self.visits
        assert value is not None
        return value

    def _get_bootstrap_value(self):
        """Helper to determine value when visits are 0."""
        if self.estimation_method == "v_mix":
            value = self.parent.get_v_mix()
        elif self.estimation_method == "mcts_value":
            value = self.parent.value()
        elif self.estimation_method == "network_value":
            value = self.parent.network_value
        else:
            value = 0.0
        return value

    def get_child(self, action):
        if action not in self.children:
            # Lazy create
            NodeType = ChanceNode if self.stochastic else DecisionNode
            p = self.child_priors[action].item()
            self.children[action] = NodeType(p, self)
        return self.children[action]

    def child_reward(self, child):
        # assert isinstance(child, DecisionNode)
        # Value Prefix subtraction is now handled by the AgentNetwork (Composer)
        # and passed as an instant reward in InferenceOutput.
        true_reward = child.reward
        return true_reward

    def get_v_mix(self):
        if self._v_mix is not None:
            return self._v_mix

        # sum_N = visits.sum()
        sum_N = self.child_visits.sum()

        if sum_N > 0:
            term = self._calculate_visited_policy_mass(sum_N)
        else:
            term = 0.0

        v_mix = (self.value() + term) / (1.0 + sum_N)
        self._v_mix = v_mix
        assert v_mix is not None
        return v_mix

    def _calculate_visited_policy_mass(self, sum_N):
        """Calculates the weighted value term for v_mix based on visited actions."""
        # Vectorized implementation
        # Check if we have child stats populated
        if self.child_visits is None:
            return 0.0

        visited_mask = self.child_visits > 0
        if not visited_mask.any():
            return 0.0

        q_vis = self.child_values[visited_mask]
        p_vis = self.child_priors[visited_mask]

        p_vis_sum = p_vis.sum()
        expected_q_vis = (p_vis * q_vis).sum()

        if p_vis_sum == 0:
            return 0.0

        term = sum_N * (expected_q_vis / p_vis_sum)
        return term

    def get_child_q_for_unvisited(self):
        """Returns the bootstrap Q-value for an unvisited child."""
        # For unvisited children, we don't have a reward estimate (r = 0 or unknown).
        # We assume Q(s,a) ~ V(s) or V_mix(s).
        # This effectively assumes the immediate reward is 0 (or centered) relative to value scale,
        # OR that the value estimate V(s) includes the expected reward.
        # In MuZero, V(s) is the value of state s. Q(s,a) = r + gamma * V(s').
        # If we approximate Q(s,a) with V(s), we are assuming r + gamma*V(s') ~ V(s).

        if self.estimation_method == "v_mix":
            val = self.get_v_mix()
        elif self.estimation_method == "mcts_value":
            val = self.value()
        elif self.estimation_method == "network_value":
            val = self.network_value if self.network_value is not None else 0.0
        else:
            val = 0.0

        # Ensure tensor output if needed?
        # Usually returns float, but can be broadcasted.
        return val

    def get_child_q_from_parent(self, child):
        # This legacy method is still useful for non-vectorized parts or debugging
        if isinstance(child, DecisionNode):
            if not child.expanded():
                v = child.value()
                return v.detach().item() if torch.is_tensor(v) else float(v)

            r = self.child_reward(child)
            if torch.is_tensor(r):
                r = r.detach().item()
            else:
                r = float(r)

            v = child.value()
            if torch.is_tensor(v):
                v = v.detach().item()
            else:
                v = float(v)
            sign = 1.0 if child.to_play == self.to_play else -1.0
            q_from_parent = r + self.discount * (sign * v)
        elif isinstance(child, ChanceNode):
            assert (
                child.to_play == self.to_play
            ), "chance nodes should be the same player as their parent"
            q_from_parent = child.value()

        return q_from_parent
