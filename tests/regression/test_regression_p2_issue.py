import pytest

from agents.trainers.base_trainer import BaseTrainer
from stats.stats import StatTracker


class _DummyTrainer:
    _process_test_results = BaseTrainer._process_test_results

    def __init__(self):
        self.stats = StatTracker(name="regression_p2_issue")
        self.num_players = 2


@pytest.mark.regression
def test_regression_p2_issue():
    """
    Regression guard for P2 result handling in test logging.
    Ensures `vs_*_p1` results are mapped to the `p1` series.
    """
    trainer = _DummyTrainer()
    all_results = {"vs_expert_p1": {"score": 0.75}}

    trainer._process_test_results(all_results, step=10)

    recorded = trainer.stats.get_data()
    assert "vs_expert_score" in recorded
    assert "p1" in recorded["vs_expert_score"]
    assert recorded["vs_expert_score"]["p1"] == [0.75]
