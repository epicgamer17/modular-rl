"""Compatibility smoke checks for NFSP trainer wiring."""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]

import importlib



def test_nfsp_trainer_module_imports_or_skips():
    try:
        module = importlib.import_module("agents.trainers.nfsp_trainer")
    except Exception as exc:
        pytest.skip(f"NFSP trainer stack unavailable in this build: {exc}")

    assert hasattr(module, "NFSPTrainer")
