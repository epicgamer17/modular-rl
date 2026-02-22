import torch
import numpy as np
import matplotlib.pyplot as plt
from stats.stats import StatTracker, PlotType


def test_stat_tracker_new_viz():
    print("Testing StatTracker with new visualizations...")
    tracker = StatTracker(model_name="test_model")

    # 1. Test Policy Entropy (Rolling Avg)
    print("  Adding policy_entropy...")
    tracker.add_plot_types("policy_entropy", PlotType.ROLLING_AVG, rolling_window=5)
    for i in range(20):
        tracker.append("policy_entropy", 2.0 - 0.05 * i + np.random.normal(0, 0.1))

    # 2. Test Policy Improvement (BAR Comparison)
    print("  Adding policy_improvement (BAR)...")
    tracker.add_plot_types("policy_improvement", PlotType.BAR)
    num_actions = 100
    network_p = torch.zeros(num_actions)
    network_p[10] = 0.5
    network_p[20] = 0.5

    search_p = torch.zeros(num_actions)
    search_p[10] = 0.1
    search_p[20] = 0.9

    tracker.append("policy_improvement", network_p, subkey="network")
    tracker.append("policy_improvement", search_p, subkey="search")

    # 3. Test Value Diff
    print("  Adding value_diff...")
    tracker.add_plot_types("value_diff", PlotType.ROLLING_AVG)
    for i in range(20):
        tracker.append("value_diff", 0.5 / (i + 1))

    # 4. Test Plotting
    print("  Plotting graphs...")
    test_dir = "/tmp/test_graphs"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    tracker.plot_graphs(dir=test_dir)
    print("  Done. Check test_graphs directory.")


if __name__ == "__main__":
    test_stat_tracker_new_viz()
