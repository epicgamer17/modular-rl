import pytest
from observability.metrics.store import MetricStore

from observability.tracing.event_schema import EventEmitter, Event, EventType

pytestmark = pytest.mark.unit

def test_metric_store_log_and_get():
    store = MetricStore()
    store.log("reward", 10.5, step=1)
    store.log("reward", 12.0, step=2)
    
    rewards = store.get("reward")
    assert len(rewards) == 2
    assert rewards[0].value == 10.5
    assert rewards[1].value == 12.0
    assert rewards[0].step == 1
    assert rewards[1].step == 2

def test_metric_store_ema():
    store = MetricStore()
    store.log("loss", 1.0, step=1)
    store.log("loss", 0.5, step=2)
    
    ema = store.get_ema("loss")
    # alpha = 0.05, ema0 = 1.0, val1 = 0.5
    # ema1 = 0.05 * 0.5 + 0.95 * 1.0 = 0.025 + 0.95 = 0.975
    assert pytest.approx(ema) == 0.975

def test_event_emitter():

    emitter = EventEmitter()
    received_events = []
    
    def callback(event: Event):
        received_events.append(event)
        
    emitter.subscribe(callback)
    
    emitter.emit_metric("test_metric", 1.0, step=10)
    
    assert len(received_events) == 1
    assert received_events[0].type == EventType.METRIC
    assert received_events[0].name == "test_metric"
    assert received_events[0].value == 1.0
    assert received_events[0].step == 10
