import pytest
from observability.metrics.store import MetricStore

pytestmark = pytest.mark.unit

def test_metrics_store_log_and_get():
    store = MetricStore()
    store.log("reward", 10.5, step=1)
    store.log("reward", 12.0, step=2)
    
    rewards = store.get("reward")
    assert len(rewards) == 2
    assert rewards[0].value == 10.5
    assert rewards[1].value == 12.0
    assert rewards[0].step == 1
    assert rewards[1].step == 2

def test_metrics_store_compute_rates():
    store = MetricStore()
    
    # Simulate some steps and time
    rates = store.compute_rates(current_actor_step=100, current_learner_step=50)
    
    # Since elapsed time is small in tests, it might return previous rates or very high rates
    # But it should return a dict with sps and ups
    assert "sps" in rates
    assert "ups" in rates
    assert isinstance(rates["sps"], float)
    assert isinstance(rates["ups"], float)
