import pytest
import torch
import numpy as np
from agents.catan_player_wrapper import CatanPlayerWrapper

pytestmark = pytest.mark.integration


def test_catan_wrapper_out_of_bounds_action(make_muzero_config_dict):
    torch.manual_seed(42)
    np.random.seed(42)

    config = make_muzero_config_dict()
    wrapper = CatanPlayerWrapper(config)

    # Attempt to pass an action index that exceeds the action space
    invalid_action = 9999

    with pytest.raises((IndexError, ValueError)):
        wrapper.step(invalid_action)
