import torch
import torch.nn as nn
import os
import sys

# Fix path for imports
sys.path.append(os.getcwd())

from modules.backbones.mlp import NoisyLinear

def test_noisy_linear_init():
    # 1. Create a NoisyLinear layer
    in_features = 4
    out_features = 8
    layer = NoisyLinear(in_features, out_features)
    
    # 2. Check the noise buffers
    print(f"Epsilon weight buffer (first 5 elements): {layer.epsilon_weight.flatten()[:5]}")
    
    # 3. Check for extremely large values (uninitialized garbage)
    max_val = layer.epsilon_weight.abs().max().item()
    print(f"Max absolute value in noise buffer: {max_val}")
    
    if max_val > 10.0:
        print("ALERT: Uninitialized garbage detected in noise buffer!")
    else:
        print("Noise buffer seems reasonably small (but might still be uninitialized).")
        
    # 4. Try a forward pass
    x = torch.ones((1, in_features))
    y = layer(x)
    print(f"Output of forward pass: {y}")
    
    if torch.isnan(y).any() or torch.isinf(y).any():
        print("ALERT: Output contains NaN or Inf!")

if __name__ == "__main__":
    test_noisy_linear_init()
