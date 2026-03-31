import random
from typing import Dict, Any

import numpy as np
import pytest
import torch
from modules.backbones.mlp import MLPBackbone
from modules.backbones.resnet import ResNetBackbone
from modules.backbones.transformer import TransformerBackbone

# ----------------- CONSTANTS -----------------
SEED = 42
WARMUP_DEFAULT = 10
WARMUP_COMPILED = 20
# For mac compatibility, always use CPU for torch.compile as MPS is unsupported.
TARGET_DEVICE = "cpu"

# ----------------- PYTEST MARKERS -----------------
# Every single test file MUST declare a module-level pytest marker at the very top of the file.
pytestmark = pytest.mark.performance


def seed_all(seed: int = SEED) -> None:
    """Enforce strict determinism as per testing-standards.md rule 134."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ----------------- BENCHMARK CONFIGS -----------------
# Defining Small, Medium, and Large networks across different architectures.

BENCHMARK_CONFIGS: Dict[str, Dict[str, Any]] = {
    # --- MLP Architectures ---
    "MLP_Small": {
        "class": MLPBackbone,
        "kwargs": {"input_shape": (128,), "widths": [128, 128], "norm_type": "none"},
        "input_shape": (1, 128),
    },
    "MLP_Medium": {
        "class": MLPBackbone,
        "kwargs": {"input_shape": (512,), "widths": [512, 512, 512], "norm_type": "none"},
        "input_shape": (1, 512),
    },
    "MLP_Large": {
        "class": MLPBackbone,
        "kwargs": {
            "input_shape": (2048,),
            "widths": [2048, 2048, 2048, 2048],
            "norm_type": "none",
        },
        "input_shape": (1, 2048),
    },
    # --- ResNet Architectures ---
    "ResNet_Small": {
        "class": ResNetBackbone,
        "kwargs": {
            "input_shape": (3, 32, 32),
            "filters": [32, 32],
            "kernel_sizes": [3, 3],
            "strides": [1, 1],
            "norm_type": "none", # Changed from layer to none to avoid spatial dimension mismatch
        },
        "input_shape": (1, 3, 32, 32),
    },
    "ResNet_Medium": {
        "class": ResNetBackbone,
        "kwargs": {
            "input_shape": (3, 64, 64),
            "filters": [64, 64, 128, 128],
            "kernel_sizes": [3, 3, 3, 3],
            "strides": [1, 2, 1, 2],
            "norm_type": "none", # Changed from layer to none to avoid spatial dimension mismatch
        },
        "input_shape": (1, 3, 64, 64),
    },
    "ResNet_Large": {
        "class": ResNetBackbone,
        "kwargs": {
            "input_shape": (3, 128, 128),
            "filters": [128, 128, 256, 256, 512, 512],
            "kernel_sizes": [3, 3, 3, 3, 3, 3],
            "strides": [1, 2, 1, 2, 1, 2],
            "norm_type": "none", # Changed from layer to none to avoid spatial dimension mismatch
        },
        "input_shape": (1, 3, 128, 128),
    },
    # --- Transformer Architectures ---
    "Transformer_Small": {
        "class": TransformerBackbone,
        "kwargs": {
            "input_shape": (10, 64),
            "d_model": 64,
            "num_heads": 4,
            "d_ff": 128,
            "num_layers": 2,
        },
        "input_shape": (1, 10, 64),
    },
    "Transformer_Medium": {
        "class": TransformerBackbone,
        "kwargs": {
            "input_shape": (50, 256),
            "d_model": 256,
            "num_heads": 8,
            "d_ff": 512,
            "num_layers": 6,
        },
        "input_shape": (1, 50, 256),
    },
    "Transformer_Large": {
        "class": TransformerBackbone,
        "kwargs": {
            "input_shape": (100, 512),
            "d_model": 512,
            "num_heads": 16,
            "d_ff": 1024,
            "num_layers": 12,
        },
        "input_shape": (1, 100, 512),
    },
    # --- LSTM Architectures ---
    "LSTM_Small": {
        "class": "custom_lstm", # Use a helper since standard LSTM takes (B, T, D)
        "kwargs": {
            "input_shape": (10, 64),
            "hidden_size": 128,
            "num_layers": 1,
            "rnn_type": "lstm"
        },
        "input_shape": (1, 10, 64),
    },
    "LSTM_Medium": {
        "class": "custom_lstm",
        "kwargs": {
            "input_shape": (32, 256),
            "hidden_size": 256,
            "num_layers": 2,
            "rnn_type": "lstm"
        },
        "input_shape": (1, 32, 256),
    },
    "LSTM_Large": {
        "class": "custom_lstm",
        "kwargs": {
            "input_shape": (100, 1024),
            "hidden_size": 1024,
            "num_layers": 4,
            "rnn_type": "lstm"
        },
        "input_shape": (1, 100, 1024),
    },
}

def build_model(config_name: str, device: torch.device):
    config = BENCHMARK_CONFIGS[config_name]
    if config["class"] == "custom_lstm":
        from modules.backbones.recurrent import RecurrentBackbone
        return RecurrentBackbone(**config["kwargs"]).to(device)
    
    return config["class"](**config["kwargs"]).to(device)

@pytest.mark.parametrize("config_name", BENCHMARK_CONFIGS.keys())
@pytest.mark.parametrize("compile_enabled", [False, True])
def test_torch_compile_inference_latency(config_name: str, compile_enabled: bool, benchmark):
    """
    Benchmarks the inference latency speedup from torch.compile.
    
    This benchmark follows the 'Micro-Benchmarks' and 'The Warm-Up Rule' from the 
    project's performance testing standards. It provides speed comparisons for 
    Small, Medium, and Large networks across MLP, ResNet, Transformer, and LSTM architectures.
    """
    seed_all(SEED)
    
    # Selection of device: torch.compile is primarily optimized for CUDA/CPU.
    # On Mac platforms, we explicitly use CPU as MPS is not currently supported for compilation.
    device = torch.device(TARGET_DEVICE)
    
    # Reset dynamo cache to ensure isolation between benchmarks
    torch._dynamo.reset()
    
    # Initialize the specific model config
    config = BENCHMARK_CONFIGS[config_name]
    model = build_model(config_name, device)
    model.eval()
    
    if compile_enabled:
        # Applying torch.compile. 
        # Note: First inference will incur compilation overhead (handled by warmup).
        model = torch.compile(model)
        
    input_tensor = torch.randn(config["input_shape"], device=device)
    
    # --- The Warm-Up Rule (Rule 145) ---
    # We increase warmup iterations for compiled models to ensure the actual compilation is finished.
    # Inductor compilation can be slow, especially for 'Large' variants.
    warmup_iters = 50 if compile_enabled else WARMUP_DEFAULT
    with torch.inference_mode():
        for _ in range(warmup_iters):
            _ = model(input_tensor)
            
    # --- Benchmark Execution ---
    def inference_step():
        with torch.inference_mode():
            return model(input_tensor)
            
    # Metadata for better report identification
    benchmark.group = config_name
    benchmark.extra_info['compile_enabled'] = compile_enabled
    
    # Execution
    benchmark(inference_step)

    # Relative assertion logic can be added here if we want to enforce specific speedups in CI,
    # but as per 'Track, Don't Fail' rule, we usually just report.
