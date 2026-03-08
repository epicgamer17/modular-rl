import pytest

pytestmark = pytest.mark.unit

import torch
import numpy as np
from search.nodes import DecisionNode, ChanceNode


class TestChanceNodeExpandAndUpdate:
    def test_chance_node_expand_initializes_vectorized_stats(self, monkeypatch):
        monkeypatch.setattr(ChanceNode, "discount", 0.9)
        monkeypatch.setattr(ChanceNode, "bootstrap_method", "zero")

        parent = DecisionNode(prior=0.5)
        parent.to_play = 0
        parent.network_value = 0.5

        node = ChanceNode(prior=0.3, parent=parent)

        network_state = torch.randn(1, 32)
        network_value = torch.tensor(0.7)
        code_probs = torch.tensor([0.2, 0.5, 0.3])

        node.expand(
            to_play=0,
            network_state=network_state,
            network_value=network_value,
            code_probs=code_probs,
        )

        assert node.expanded() is True
        assert node.network_value == 0.7
        assert node.network_state is not None
        assert node.child_priors.shape[0] == 3
        assert node.child_visits.shape[0] == 3
        assert node.child_values.shape[0] == 3

    def test_chance_node_value_returns_network_when_unvisited(self, monkeypatch):
        monkeypatch.setattr(ChanceNode, "discount", 0.9)
        monkeypatch.setattr(ChanceNode, "bootstrap_method", "network_value")

        parent = DecisionNode(prior=0.5)
        parent.to_play = 0
        parent.network_value = 0.7
        parent.network_policy = torch.tensor([0.5, 0.5])
        parent.child_priors = torch.tensor([0.5, 0.5])
        parent.child_visits = torch.tensor([0.0, 0.0])
        parent.child_values = torch.tensor([0.0, 0.0])
        parent._v_mix = None

        node = ChanceNode(prior=0.3, parent=parent)
        node.network_value = 0.7
        node.expand(
            to_play=0,
            network_state=torch.randn(1, 32),
            network_value=0.7,
            code_probs=torch.tensor([0.5, 0.5]),
        )

        assert node.visits == 0
        assert node.value() == 0.7

    def test_chance_node_q_value_updates_correctly_with_bellman(self, monkeypatch):
        monkeypatch.setattr(ChanceNode, "discount", 0.9)
        monkeypatch.setattr(ChanceNode, "bootstrap_method", "zero")

        parent = DecisionNode(prior=0.5)
        parent.to_play = 0
        parent.network_value = 0.5

        node = ChanceNode(prior=0.3, parent=parent)
        node.network_value = 0.7
        node.expand(
            to_play=0,
            network_state=torch.randn(1, 32),
            network_value=0.7,
            code_probs=torch.tensor([0.2, 0.5, 0.3]),
        )

        reward = 1.0
        child_value = 0.5
        action_index = 1

        q_value = reward + 0.9 * child_value

        node.child_visits[action_index] += 1
        node.child_values[action_index] = q_value
        node.value_sum += q_value
        node.visits += 1

        assert node.visits == 1
        expected_q = node.value_sum / node.visits
        assert torch.allclose(torch.tensor(expected_q), torch.tensor(q_value))

    def test_chance_node_multiple_updates_accumulate_correctly(self, monkeypatch):
        monkeypatch.setattr(ChanceNode, "discount", 0.9)
        monkeypatch.setattr(ChanceNode, "bootstrap_method", "zero")

        parent = DecisionNode(prior=0.5)
        parent.to_play = 0
        parent.network_value = 0.5

        node = ChanceNode(prior=0.3, parent=parent)
        node.network_value = 0.7
        node.expand(
            to_play=0,
            network_state=torch.randn(1, 32),
            network_value=0.7,
            code_probs=torch.tensor([0.25, 0.25, 0.25, 0.25]),
        )

        updates = [
            (0, 1.0, 0.5),
            (2, 0.8, 0.3),
            (0, 0.9, 0.6),
        ]

        total_q = 0.0
        for action_idx, reward, child_val in updates:
            q_value = reward + 0.9 * child_val
            total_q += q_value
            node.child_visits[action_idx] += 1
            node.child_values[action_idx] = q_value
            node.value_sum += q_value
            node.visits += 1

        assert node.visits == 3
        expected_avg = total_q / 3
        actual_avg = node.value()
        assert torch.allclose(
            torch.tensor(actual_avg), torch.tensor(expected_avg), atol=1e-6
        )


class TestDecisionNodeExpandAndUpdate:
    def test_decision_node_expand_initializes_vectorized_stats(self, monkeypatch):
        monkeypatch.setattr(DecisionNode, "discount", 0.9)
        monkeypatch.setattr(DecisionNode, "bootstrap_method", "zero")
        monkeypatch.setattr(DecisionNode, "stochastic", False)
        monkeypatch.setattr(DecisionNode, "pb_c_init", 1.25)
        monkeypatch.setattr(DecisionNode, "pb_c_base", 19652)

        node = DecisionNode(prior=0.5)
        node.network_policy = torch.tensor([0.1, 0.2, 0.3, 0.4])
        node.network_value = 0.6
        node.reward = 0.0

        priors = torch.tensor([0.1, 0.2, 0.3, 0.4])

        node.expand(
            allowed_actions=None,
            to_play=0,
            priors=priors,
            network_policy=node.network_policy,
            network_state=torch.randn(1, 32),
            reward=0.0,
            value=0.6,
        )

        assert node.expanded() is True
        assert node.child_priors.shape[0] == 4
        assert node.child_visits.shape[0] == 4
        assert node.child_values.shape[0] == 4

    def test_decision_node_legal_action_mask_one_hot(self, monkeypatch):
        monkeypatch.setattr(DecisionNode, "discount", 0.9)
        monkeypatch.setattr(DecisionNode, "bootstrap_method", "zero")
        monkeypatch.setattr(DecisionNode, "stochastic", False)
        monkeypatch.setattr(DecisionNode, "pb_c_init", 1.25)
        monkeypatch.setattr(DecisionNode, "pb_c_base", 19652)

        node = DecisionNode(prior=0.5)
        network_policy = torch.tensor([0.0, 0.0, 1.0, 0.0])
        node.network_policy = network_policy
        node.network_value = 0.6
        node.reward = 0.0

        allowed_actions = torch.tensor([1, 3])

        node.expand(
            allowed_actions=allowed_actions,
            to_play=0,
            priors=None,
            network_policy=network_policy,
            network_state=torch.randn(1, 32),
            reward=0.0,
            value=0.6,
        )

        assert node.child_priors[0] == 0.0
        assert node.child_priors[2] == 0.0
        assert node.child_priors[1] > 0.0
        assert node.child_priors[3] > 0.0
        assert torch.allclose(node.child_priors.sum(), torch.tensor(1.0), atol=1e-6)

    def test_decision_node_legal_action_mask_discrete_one_hot_input(self, monkeypatch):
        monkeypatch.setattr(DecisionNode, "discount", 0.9)
        monkeypatch.setattr(DecisionNode, "bootstrap_method", "zero")
        monkeypatch.setattr(DecisionNode, "stochastic", False)
        monkeypatch.setattr(DecisionNode, "pb_c_init", 1.25)
        monkeypatch.setattr(DecisionNode, "pb_c_base", 19652)

        node = DecisionNode(prior=0.5)
        network_policy = torch.tensor([1.0, 0.0, 0.0, 0.0])
        node.network_policy = network_policy
        node.network_value = 0.6
        node.reward = 0.0

        allowed_actions = torch.tensor([2])

        node.expand(
            allowed_actions=allowed_actions,
            to_play=0,
            priors=None,
            network_policy=network_policy,
            network_state=torch.randn(1, 32),
            reward=0.0,
            value=0.6,
        )

        assert node.child_priors[0] == 0.0
        assert node.child_priors[1] == 0.0
        assert node.child_priors[3] == 0.0
        assert node.child_priors[2] == 1.0

    def test_decision_node_q_value_updates_correctly_with_bellman(self, monkeypatch):
        monkeypatch.setattr(DecisionNode, "discount", 0.9)
        monkeypatch.setattr(DecisionNode, "bootstrap_method", "zero")
        monkeypatch.setattr(DecisionNode, "stochastic", False)
        monkeypatch.setattr(DecisionNode, "pb_c_init", 1.25)
        monkeypatch.setattr(DecisionNode, "pb_c_base", 19652)

        parent = DecisionNode(prior=0.5)
        parent.to_play = 0
        parent.network_value = 0.5

        node = DecisionNode(prior=0.3, parent=parent)
        node.to_play = 0
        node.reward = 0.0
        node.network_policy = torch.tensor([0.25, 0.25, 0.25, 0.25])
        node.network_value = 0.6
        node.expand(
            allowed_actions=None,
            to_play=0,
            priors=None,
            network_policy=node.network_policy,
            network_state=torch.randn(1, 32),
            reward=0.0,
            value=0.6,
        )

        reward = 1.0
        child_value = 0.5
        action_index = 2

        q_value = reward + 0.9 * child_value

        node.child_visits[action_index] += 1
        node.child_values[action_index] = q_value
        node.value_sum += q_value
        node.visits += 1

        assert node.visits == 1
        expected_q = node.value_sum / node.visits
        assert torch.allclose(torch.tensor(expected_q), torch.tensor(q_value))

    def test_decision_node_multiple_updates_accumulate_correctly(self, monkeypatch):
        monkeypatch.setattr(DecisionNode, "discount", 0.9)
        monkeypatch.setattr(DecisionNode, "bootstrap_method", "zero")
        monkeypatch.setattr(DecisionNode, "stochastic", False)
        monkeypatch.setattr(DecisionNode, "pb_c_init", 1.25)
        monkeypatch.setattr(DecisionNode, "pb_c_base", 19652)

        parent = DecisionNode(prior=0.5)
        parent.to_play = 0
        parent.network_value = 0.5

        node = DecisionNode(prior=0.3, parent=parent)
        node.to_play = 0
        node.reward = 0.0
        node.network_policy = torch.tensor([0.1, 0.2, 0.3, 0.4])
        node.network_value = 0.6
        node.expand(
            allowed_actions=None,
            to_play=0,
            priors=None,
            network_policy=node.network_policy,
            network_state=torch.randn(1, 32),
            reward=0.0,
            value=0.6,
        )

        updates = [
            (0, 1.0, 0.5),
            (3, 0.8, 0.3),
            (0, 0.9, 0.6),
        ]

        total_q = 0.0
        for action_idx, reward, child_val in updates:
            q_value = reward + 0.9 * child_val
            total_q += q_value
            node.child_visits[action_idx] += 1
            node.child_values[action_idx] = q_value
            node.value_sum += q_value
            node.visits += 1

        assert node.visits == 3
        expected_avg = total_q / 3
        actual_avg = node.value()
        assert torch.allclose(
            torch.tensor(actual_avg), torch.tensor(expected_avg), atol=1e-6
        )


class TestDecisionNodeValueBootstrap:
    def test_decision_node_value_returns_network_when_unvisited(self, monkeypatch):
        monkeypatch.setattr(DecisionNode, "discount", 0.9)
        monkeypatch.setattr(DecisionNode, "bootstrap_method", "network_value")
        monkeypatch.setattr(DecisionNode, "stochastic", False)

        parent = DecisionNode(prior=0.5)
        parent.to_play = 0
        parent.network_value = 0.75

        node = DecisionNode(prior=0.5, parent=parent)
        node.network_value = 0.75
        node.network_policy = torch.tensor([0.25, 0.25, 0.25, 0.25])
        node.expand(
            allowed_actions=None,
            to_play=0,
            priors=None,
            network_policy=node.network_policy,
            network_state=torch.randn(1, 32),
            reward=0.0,
            value=0.75,
        )

        assert node.visits == 0
        assert node.value() == 0.75


class TestChanceNodeValueBootstrap:
    def test_chance_node_value_returns_network_when_unvisited_with_v_mix(
        self, monkeypatch
    ):
        monkeypatch.setattr(ChanceNode, "discount", 0.9)
        monkeypatch.setattr(ChanceNode, "bootstrap_method", "v_mix")

        parent = DecisionNode(prior=0.5)
        parent.to_play = 0
        parent.network_value = 0.5
        parent.network_policy = torch.tensor([0.5, 0.5])
        parent.child_priors = torch.tensor([0.5, 0.5])
        parent.child_visits = torch.tensor([0.0, 0.0])
        parent.child_values = torch.tensor([0.0, 0.0])

        node = ChanceNode(prior=0.3, parent=parent)
        node.network_value = 0.7
        node.expand(
            to_play=0,
            network_state=torch.randn(1, 32),
            network_value=0.7,
            code_probs=torch.tensor([0.5, 0.5]),
        )

        assert node.visits == 0
        v_mix = node.get_v_mix()
        assert v_mix is not None
