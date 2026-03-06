import pytest
pytestmark = pytest.mark.integration

import copy
import torch
import torch.optim as optim
import pickle
from modules.utils import get_lr_scheduler
from utils.schedule import ScheduleConfig


def test_pickling(rainbow_cartpole_replay_config):
    print("Testing pickling of lr_scheduler...")

    model = torch.nn.Linear(10, 2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    config = copy.deepcopy(rainbow_cartpole_replay_config)

    # Test linear
    print("  Testing linear...")
    config.lr_schedule = ScheduleConfig.linear(
        initial=0.001, final=0.00001, decay_steps=1000
    )
    scheduler = get_lr_scheduler(optimizer, config)
    pickle.dumps(scheduler)
    print("  Linear pickling successful!")

    # Test stepwise
    print("  Testing stepwise...")
    config.lr_schedule = ScheduleConfig.stepwise(
        steps=[100, 200], values=[0.001, 0.0001, 0.00001]
    )
    scheduler = get_lr_scheduler(optimizer, config)
    pickle.dumps(scheduler)
    print("  Stepwise pickling successful!")

    # Test constant
    print("  Testing constant...")
    config.lr_schedule = ScheduleConfig.constant(0.001)
    scheduler = get_lr_scheduler(optimizer, config)
    pickle.dumps(scheduler)
    print("  Constant pickling successful!")
