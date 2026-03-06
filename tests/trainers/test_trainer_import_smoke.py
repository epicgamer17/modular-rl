import pytest
pytestmark = pytest.mark.unit

import importlib
import sys
from unittest.mock import MagicMock


def test_trainer_import_smoke(monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib", MagicMock())
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", MagicMock())

    modules = [
        "agents.executors.factory",
        "agents.learners.muzero_learner",
        "agents.workers",
        "modules.agent_nets.modular",
        "stats.stats",
        "agents.trainers.muzero_trainer",
    ]

    for module_name in modules:
        importlib.import_module(module_name)
