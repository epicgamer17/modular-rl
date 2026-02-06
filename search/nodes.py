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

    def expand(
        self,
        to_play,
        afterstate,
        network_value,
        code_probs,
        reward_h_state=None,
        reward_c_state=None,
    ):
        """
        Called when the Dynamics Network is run on (parent_state, action).
        """
        self.to_play = to_play
        self.afterstate = afterstate
        self.reward_h_state = reward_h_state
        self.reward_c_state = reward_c_state

        self.network_value = network_value
        # code_probs should be a dict or array mapping code_index -> probability
        num_chance = len(code_probs)
        self.code_probs = {
            # Warning non differentiable
            a: code_probs[a]
            for a in range(num_chance)
        }

        for code, p in self.code_probs.items():
            self.children[code] = DecisionNode(
                p.item(),  # TODO: IS THIS RIGHT?
                self,
            )

    def expanded(self):
        # We are expanded if we have populated our priors/value
        # Relaxing this for batched search where visits > 0 (virtual visits) can happen before expansion
        # assert (
        #     (len(self.children) > 0) == (self.visits > 0) == (len(self.code_probs) > 0)
        # )
        return len(self.code_probs) > 0

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
        # grab probabilities for candidate actions
        visits = torch.tensor(
            [float(child.visits) for (code, child) in self.children.items()]
        )
        visited_codes = [
            code for code, child in self.children.items() if child.expanded()
        ]
        sum_N = float(visits.sum())

        if sum_N > 0:
            term = self._calculate_visited_policy_mass(visited_codes, sum_N)
        else:
            term = 0.0

        v_mix = (self.value() + term) / (1.0 + sum_N)
        assert v_mix is not None
        return v_mix

    def _calculate_visited_policy_mass(self, visited_codes, sum_N):
        """Calculates the weighted value term for v_mix based on visited codes."""
        num_codes = len(self.code_probs)
        q_vals = torch.zeros(num_codes)

        # Convert code_probs dict to tensor
        code_probs_list = [self.code_probs[i] for i in range(num_codes)]
        code_probs_tensor = torch.tensor(code_probs_list)

        p_vis_sum = 0
        for code in visited_codes:
            child = self.children[code]
            q_vals[code] = self.get_child_q_from_parent(child)
            p_vis_sum += code_probs_tensor[code]
        expected_q_vis = (code_probs_tensor * q_vals).sum()
        if torch.is_tensor(expected_q_vis):
            expected_q_vis = expected_q_vis.detach().item()
        else:
            expected_q_vis = float(expected_q_vis)

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
        self.hidden_state = None
        self.reward_h_state = None
        self.reward_c_state = None

        self.reward = 0
        self.parent = parent

        self.network_policy = None  # dense policy vector (numpy or torch)
        self.network_value = None  # network scalar value estimate (float)

    def expand(
        self,
        allowed_actions,
        to_play,
        priors,
        network_policy,
        hidden_state,
        reward,
        value=None,
        reward_h_state=None,
        reward_c_state=None,
        is_reset=True,
    ):
        self.to_play = to_play
        self.reward = reward
        self.hidden_state = hidden_state
        self.reward_h_state = reward_h_state
        self.reward_c_state = reward_c_state
        self.is_reset = is_reset

        self.network_policy = network_policy.detach().cpu()
        self.network_value = value

        self._populate_children(allowed_actions, priors)

    def _populate_children(self, allowed_actions, priors):
        """Helper to create child nodes based on policy and allowed actions."""
        allowed_priors = {a: priors[a] for a in allowed_actions}
        allowed_priors_sum = sum(allowed_priors.values())

        NodeType = ChanceNode if self.stochastic else DecisionNode

        for action, p in allowed_priors.items():
            # normalized_p = (p / (allowed_priors_sum + 1e-10)).item()
            # TODO: SHOULD I NORMALIZE PRIORS OR SHOULD THIS BE DONE BEFOREHAND?? I THINK WE SHOULD NOT NORMALIZE PRIORS FOR THINGS LIKE SAMPLE MUZERO BUT IM NOT SURE
            self.children[action] = NodeType(p, self)

    def expanded(self):
        # assert (len(self.children) > 0) == (self.visits > 0)
        return len(self.children) > 0

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

    def child_reward(self, child):
        # assert isinstance(child, DecisionNode)
        if self.value_prefix:
            assert child.expanded()
            if child.is_reset:
                return child.reward
            else:
                return child.reward - self.reward
        else:
            true_reward = child.reward

        assert true_reward is not None
        return true_reward

    def get_v_mix(self):
        # # grab probabilities for candidate actions
        visits = torch.tensor(
            [float(child.visits) for (action, child) in self.children.items()]
        )
        visited_actions = [
            action for action, child in self.children.items() if child.expanded()
        ]
        sum_N = float(visits.sum())

        if sum_N > 0:
            term = self._calculate_visited_policy_mass(visited_actions, sum_N)
        else:
            term = 0.0

        v_mix = (self.value() + term) / (1.0 + sum_N)
        assert v_mix is not None
        return v_mix

    def _calculate_visited_policy_mass(self, visited_actions, sum_N):
        """Calculates the weighted value term for v_mix based on visited actions."""
        q_vals = torch.zeros(len(self.network_policy))
        # q(a) for visited actions (use child.value() empirical)
        p_vis_sum = 0
        for action in visited_actions:
            child = self.children[action]
            q_vals[action] = self.get_child_q_from_parent(child)
            p_vis_sum += self.network_policy[action]
        expected_q_vis = (self.network_policy * q_vals).sum()
        if torch.is_tensor(expected_q_vis):
            expected_q_vis = expected_q_vis.detach().item()
        else:
            expected_q_vis = float(
                expected_q_vis
            )  # sum_pi(a) * q(a) but q(a)=0 for unvisited

        if p_vis_sum == 0:
            return 0.0

        term = sum_N * (expected_q_vis / p_vis_sum)
        return term

    def get_child_q_from_parent(self, child):
        if isinstance(child, DecisionNode):
            if not child.expanded():
                # For unexpanded nodes (e.g. during pruning of candidates),
                # we rely on the bootstrap value (value()) which handles the estimation method.
                v = child.value()
                return v.detach().item() if torch.is_tensor(v) else float(v)

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
        elif isinstance(child, ChanceNode):
            # TODO: should this have a discount??
            assert (
                child.to_play == self.to_play
            ), "chance nodes should be the same player as their parent"
            q_from_parent = child.value()

        return q_from_parent
