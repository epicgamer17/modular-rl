import pytest

pytestmark = pytest.mark.unit

from types import SimpleNamespace

from agents.factories.executor import create_executor
from agents.executors.local_executor import LocalExecutor
from agents.executors.torch_mp_executor import TorchMPExecutor



def test_executor_factory_routes_local():
    executor = create_executor(SimpleNamespace(executor_type="local"))
    assert isinstance(executor, LocalExecutor)



def test_executor_factory_routes_torch_mp():
    executor = create_executor(SimpleNamespace(executor_type="torch_mp"))
    assert isinstance(executor, TorchMPExecutor)



def test_executor_factory_rejects_unknown_executor_type():
    with pytest.raises(ValueError, match="Unknown executor_type"):
        create_executor(SimpleNamespace(executor_type="puffer"))
