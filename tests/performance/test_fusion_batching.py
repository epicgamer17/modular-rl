import pytest
import torch
import time
import numpy as np
from core.graph import Graph, NODE_TYPE_SOURCE
from runtime.executor import execute, register_operator
from runtime.state import ReplayBuffer

def test_optimization_throughput():
    """
    Measures throughput improvement for:
    1. Operator Fusion
    2. Batching
    3. Replay Prefetching
    """
    device = "cpu"
    data_size = (1000, 1000)
    data = torch.randn(data_size, device=device)

    # --- 1. Operator Fusion Measurement ---
    def op_heavy_1(node, inputs):
        x = list(inputs.values())[0]
        for _ in range(10): x = torch.sin(x)
        return x

    def op_heavy_2(node, inputs):
        x = list(inputs.values())[0]
        for _ in range(10): x = torch.cos(x)
        return x

    def op_heavy_fused(node, inputs):
        x = list(inputs.values())[0]
        for _ in range(10): x = torch.sin(x)
        for _ in range(10): x = torch.cos(x)
        return x

    # Measure Separate
    start = time.time()
    for _ in range(100):
        y1 = op_heavy_1(None, {"in": data})
        y2 = op_heavy_2(None, {"in": y1})
    duration_separate = time.time() - start

    # Measure Fused
    start = time.time()
    for _ in range(100):
        y_fused = op_heavy_fused(None, {"in": data})
    duration_fused = time.time() - start

    print(f"\nFusion Optimization:")
    print(f"Separate Duration: {duration_separate:.4f}s")
    print(f"Fused Duration: {duration_fused:.4f}s")
    print(f"Speedup: {duration_separate / duration_fused:.2f}x")

    # --- 2. Batching Measurement ---
    model = torch.nn.Sequential(torch.nn.Linear(100, 100), torch.nn.ReLU())
    batch_data = torch.randn(64, 100)

    # Measure Serial
    start = time.time()
    for _ in range(100):
        for i in range(64):
            _ = model(batch_data[i].unsqueeze(0))
    duration_serial = time.time() - start

    # Measure Batched
    start = time.time()
    for _ in range(100):
        _ = model(batch_data)
    duration_batched = time.time() - start

    print(f"\nBatching Optimization:")
    print(f"Serial Duration: {duration_serial:.4f}s")
    print(f"Batched Duration: {duration_batched:.4f}s")
    print(f"Speedup: {duration_serial / duration_batched:.2f}x")

    # --- 3. Replay Prefetching Measurement ---
    rb = ReplayBuffer(capacity=10000)
    for i in range(1000):
        rb.add({"obs": torch.randn(100)})
    
    # Measure Sync Sampling
    start = time.time()
    for _ in range(100):
        _ = rb.sample(64)
    duration_sync = time.time() - start

    # Measure Prefetched Sampling
    rb.prefetch(batch_size=64, count=100)
    time.sleep(0.1) # Wait for prefetch to fill
    start = time.time()
    for _ in range(100):
        _ = rb.sample(64)
    duration_prefetch = time.time() - start

    print(f"\nPrefetching Optimization:")
    print(f"Sync Duration: {duration_sync:.4f}s")
    print(f"Prefetch Duration: {duration_prefetch:.4f}s")
    # Prefetch should be nearly instant (pop from list)
    print(f"Speedup: {duration_sync / duration_prefetch:.2f}x")

if __name__ == "__main__":
    test_optimization_throughput()
