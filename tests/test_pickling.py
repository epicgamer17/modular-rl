import torch
import torch.optim as optim
import pickle
from modules.utils import get_lr_scheduler
from configs.agents.muzero import MuZeroConfig
from configs.games.cartpole_config import CartPoleConfig
from unittest.mock import MagicMock


def test_pickling():
    print("Testing pickling of lr_scheduler...")

    # Mock config
    config = MagicMock()
    config.lr_schedule_type = "linear"
    config.training_steps = 1000
    config.learning_rate = 0.001

    model = torch.nn.Linear(10, 2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Test linear
    print("  Testing linear...")
    scheduler = get_lr_scheduler(optimizer, config)
    pickle.dumps(scheduler)
    print("  Linear pickling successful!")

    # Test step_wise
    print("  Testing step_wise...")
    config.lr_schedule_type = "step_wise"
    config.lr_schedule_steps = [100, 200]
    config.lr_schedule_values = [0.0001, 0.00001]
    scheduler = get_lr_scheduler(optimizer, config)
    pickle.dumps(scheduler)
    print("  Step-wise pickling successful!")

    # Test none
    print("  Testing none...")
    config.lr_schedule_type = "none"
    scheduler = get_lr_scheduler(optimizer, config)
    pickle.dumps(scheduler)
    print("  None (constant) pickling successful!")


if __name__ == "__main__":
    try:
        test_pickling()
        print("All pickling tests passed!")
    except Exception as e:
        print(f"Pickling failed: {e}")
        import traceback

        traceback.print_exc()
