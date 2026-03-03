import numpy as np
import torch
from stats.stats import StatTracker


def reproduce():
    tracker = StatTracker("test_tracker")
    # Simulate appending a list of empty tensors or something that leads to (B, 0) shape
    # Actually, if we append empty tensors:
    tracker.append("empty_stat", torch.tensor([]))
    tracker.append("empty_stat", torch.tensor([]))

    print("Attempting to process data in _to_numpy...")
    data = tracker.get_data()["empty_stat"]
    try:
        # This is where we suspect the warning happens
        res = tracker._to_numpy(data, reduce=True)
        print("Result shape:", res.shape)
    except Exception as e:
        print("Caught exception:", e)


if __name__ == "__main__":
    reproduce()
