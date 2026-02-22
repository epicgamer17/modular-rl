import torch
import numpy as np
import os
from stats.stats import StatTracker, PlotType


def test_robust_stats_shapes():
    print("Testing StatTracker robustness against multi-dimensional tensors...")
    tracker = StatTracker(name="robust_test")

    # 1. Test (10, 1, 32) shape which was failing before (reported as (10, 1, 32) after stack)
    # This means each append was (1, 32)
    print("  Testing (B, 1, D) shape reduction...")
    tracker.add_plot_types("chance_probs_3d", PlotType.BAR)
    for _ in range(10):
        tracker.append("chance_probs_3d", torch.randn(1, 32))

    # 2. Test deeply nested reduction
    print("  Testing reduction of (B, 4, 4, 1, 1) to scalar...")
    tracker.add_plot_types("deep_tensor", PlotType.ROLLING_AVG)
    for _ in range(10):
        tracker.append("deep_tensor", torch.randn(4, 4, 1, 1))

    # 3. Test explicit 2D Bar Plot (no reduction)
    print("  Testing BAR plot without reduction (B, D)...")
    tracker.add_plot_types("legit_bar", PlotType.BAR)
    for _ in range(5):
        tracker.append("legit_bar", torch.abs(torch.randn(10)))

    # 4. Test Plotting
    test_dir = "/tmp/test_graphs_robust"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    print("  Plotting graphs...")
    try:
        tracker.plot_graphs(dir=test_dir)
        print("  SUCCESS: Graphs plotted without ValueError.")
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback

        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    test_robust_stats_shapes()
