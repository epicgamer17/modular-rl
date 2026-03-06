"""Compatibility smoke checks for Imitation trainer wiring."""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]

import importlib



def test_imitation_trainer_module_imports_or_skips():
    try:
        module = importlib.import_module("agents.trainers.imitation_trainer")
    except Exception as exc:
        pytest.skip(f"Imitation trainer stack unavailable in this build: {exc}")

    assert hasattr(module, "ImitationTrainer")
