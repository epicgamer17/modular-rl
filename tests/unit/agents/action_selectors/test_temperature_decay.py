import pytest
import torch
from agents.action_selectors.decorators import TemperatureSelector
from agents.action_selectors.selectors import CategoricalSelector
from agents.action_selectors.types import InferenceResult
from utils.schedule import ScheduleConfig

pytestmark = pytest.mark.unit


def test_temperature_decay_selection():
    """
    Tier 1: [ANALYTICAL ORACLE] Verify TemperatureSelector decay schedule.
    Assert: T steps down from 1.0 -> 0.5 -> 0.25 at step 5 and 10.
    """
    # 1. Setup stepwise schedule
    # Step 0-4: 1.0
    # Step 5-9: 0.5
    # Step 10+: 0.25
    schedule_config = ScheduleConfig(
        type="stepwise",
        steps=[5, 10],
        values=[1.0, 0.5, 0.25],
        with_training_steps=True,
    )

    inner_selector = CategoricalSelector()
    selector = TemperatureSelector(inner_selector, schedule_config)

    # 2. Mock Inference Result (logits for 2 actions)
    # Logits: [log(0.2), log(0.8)]
    # At T=1.0: probs = [0.2, 0.8]
    # At T=0.5: heated_logits = [log(0.2)/0.5, log(0.8)/0.5] = [2*log(0.2), 2*log(0.8)] = [log(0.04), log(0.64)]
    #          probs = [0.04/0.68, 0.64/0.68] approx [0.058, 0.941]
    logits = torch.tensor(
        [[torch.log(torch.tensor(0.2)), torch.log(torch.tensor(0.8))]]
    )
    result = InferenceResult(logits=logits)
    info = {}

    # --- Step 0 (T=1.0) ---
    # We call select_action or update_parameters to advance the schedule.
    # In TemperatureSelector, update_parameters(training_step=...) advances it.
    selector.update_parameters({"training_step": 0})

    # We use a hook or just check the heated logits returned in the metadata (if available)
    # Actually, TemperatureSelector replaces result.logits before passing to inner.
    # We can mock the inner selector to capture the modified result.
    from unittest.mock import MagicMock

    selector.inner_selector.select_action = MagicMock(
        return_value=(torch.tensor([1]), {})
    )

    selector.select_action(result, info, training_step=0)
    # Capture the result passed to inner
    args, kwargs = selector.inner_selector.select_action.call_args
    passed_result = args[0]

    # T=1.0, logits should be unchanged
    torch.testing.assert_close(passed_result.logits, logits)

    # --- Step 5 (T=0.5) ---
    selector.update_parameters({"training_step": 5})
    selector.select_action(result, info, training_step=5)
    args, kwargs = selector.inner_selector.select_action.call_args
    passed_result = args[0]

    expected_logits_5 = logits / 0.5
    torch.testing.assert_close(passed_result.logits, expected_logits_5)

    # --- Step 10 (T=0.25) ---
    selector.update_parameters({"training_step": 10})
    selector.select_action(result, info, training_step=10)
    args, kwargs = selector.inner_selector.select_action.call_args
    passed_result = args[0]

    expected_logits_10 = logits / 0.25
    torch.testing.assert_close(passed_result.logits, expected_logits_10)

    print("Verified Temperature stepwise decay: 1.0 -> 0.5 -> 0.25")
