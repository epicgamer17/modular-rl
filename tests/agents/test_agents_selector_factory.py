import pytest
from agents.action_selectors.factory import SelectorFactory
from agents.action_selectors.selectors import CategoricalSelector
from agents.action_selectors.decorators import PPODecorator

pytestmark = pytest.mark.unit


def test_selector_factory_valid_chain(base_ppo_config_dict):
    """Verifies factory correctly nests decorators around base selectors using a real config."""
    # base_ppo_config_dict["action_selector"] defines categorical + ppo_injector
    selector_config = base_ppo_config_dict["action_selector"]
    selector = SelectorFactory.create(selector_config)

    # Outer layer should be PPODecorator, inner layer should be CategoricalSelector
    assert isinstance(selector, PPODecorator)
    assert isinstance(selector.inner_selector, CategoricalSelector)


def test_selector_factory_missing_base():
    """Verifies missing base type raises the correct ValueError."""
    config = {"base": {}}  # Missing 'type' field
    with pytest.raises(ValueError, match="Selector config must specify a 'base' type"):
        SelectorFactory.create(config)


def test_selector_factory_unknown_base():
    """Verifies fake base types crash safely."""
    config = {"base": {"type": "not_a_real_selector"}}
    with pytest.raises(ValueError, match="Unknown selector type"):
        SelectorFactory.create(config)


def test_selector_factory_unknown_decorator():
    """Verifies fake decorator types crash safely while preserving the base."""
    config = {
        "base": {"type": "categorical"},
        "decorators": [{"type": "fake_decorator"}],
    }
    with pytest.raises(ValueError, match="Unknown decorator type"):
        SelectorFactory.create(config)
