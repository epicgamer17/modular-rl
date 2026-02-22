import torch
import torch.optim as optim
import pickle
from modules.utils import get_lr_scheduler
from utils.schedule import ScheduleConfig
from unittest.mock import MagicMock


def test_pickling():
    print("Testing pickling of lr_scheduler...")

    model = torch.nn.Linear(10, 2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Test linear
    print("  Testing linear...")
    config = MagicMock()
    config.lr_schedule = ScheduleConfig.linear(
        initial=0.001, final=0.00001, decay_steps=1000
    )
    scheduler = get_lr_scheduler(optimizer, config)
    pickle.dumps(scheduler)
    print("  Linear pickling successful!")

    # Test stepwise
    print("  Testing stepwise...")
    config.lr_schedule = ScheduleConfig.stepwise(
        steps=[100, 200], values=[0.0001, 0.00001]
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


if __name__ == "__main__":
    try:
        test_pickling()
        print("All pickling tests passed!")
    except Exception as e:
        print(f"Pickling failed: {e}")
        import traceback

        traceback.print_exc()
