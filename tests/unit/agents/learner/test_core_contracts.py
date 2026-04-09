import torch
import pytest
from learner.core import Blackboard, BlackboardEngine
from learner.pipeline.base import PipelineComponent

pytestmark = pytest.mark.unit


class MockComponent(PipelineComponent):
    def __init__(self, key, value):
        self.key = key
        self.value = value

    def execute(self, blackboard: Blackboard) -> None:
        blackboard.meta[self.key] = self.value


def test_blackboard_initialization():
    """Verify Blackboard is a pure data hub with correct fields."""
    batch = {"obs": torch.zeros(1)}
    bb = Blackboard(data=batch)

    assert bb.data == batch
    assert isinstance(bb.predictions, dict)
    assert isinstance(bb.targets, dict)
    assert isinstance(bb.losses, dict)
    assert isinstance(bb.meta, dict)


def test_universal_learner_device_transfer():
    """Verify BlackboardEngine enforces the Universal Time Mandate (device transfer)."""
    device = torch.device("cpu")
    # If a GPU is available, we'd use it, but for unit tests CPU is fine.
    # The key is checking if .to(device) is called.

    batch = {"obs": torch.randn(2, 4), "non_tensor": [1, 2, 3]}

    learner = BlackboardEngine(components=[], device=device)

    # We can't easily mock torch.Tensor.to because it's a C-extension method,
    # but we can check if the result is on the correct device.
    iterator = [batch]
    for meta in learner.step(iterator):
        pass  # step() performs the transfer

    # Actually, we should check if the components received the tensor on the correct device.
    received_device = None

    class DeviceCheckComponent(PipelineComponent):
        def execute(self, blackboard: Blackboard) -> None:
            nonlocal received_device
            received_device = blackboard.data["obs"].device

    learner.components = [DeviceCheckComponent()]
    for _ in learner.step([batch]):
        pass

    assert received_device.type == device.type


def test_universal_learner_execution_order():
    """Verify components are executed in the order they are provided."""
    bb_history = []

    class TapeComponent(PipelineComponent):
        def __init__(self, val):
            self.val = val

        def execute(self, blackboard: Blackboard) -> None:
            bb_history.append(self.val)

    components = [TapeComponent(1), TapeComponent(2), TapeComponent(3)]
    learner = BlackboardEngine(components=components, device=torch.device("cpu"))

    for _ in learner.step([{"dummy": 0}]):
        pass

    assert bb_history == [1, 2, 3]


def test_universal_learner_yields_metrics():
    """Verify BlackboardEngine yields the blackboard.meta and losses dictionaries."""
    class LossComponent(PipelineComponent):
        def execute(self, blackboard: Blackboard) -> None:
            blackboard.losses["main"] = torch.tensor(1.0)
            blackboard.meta["acc"] = 0.9

    learner = BlackboardEngine(components=[LossComponent()], device=torch.device("cpu"))

    results = list(learner.step([{"dummy": 0}]))

    assert len(results) == 1
    assert results[0]["losses"]["main"] == 1.0
    assert results[0]["meta"]["acc"] == 0.9

