import os
import sys
import torch
import torch.nn as nn

# Fix path for imports
sys.path.append(os.getcwd())

from agents.learner.target_builders import DistributionalTargetBuilder, SingleStepTargetPipeline
from agents.learner.losses.representations import C51Representation

class MockNetwork(nn.Module):
    def __init__(self, representation):
        super().__init__()
        self.components = {
            "behavior_heads": {
                "q_logits": nn.Module()
            }
        }
        self.components["behavior_heads"]["q_logits"].representation = representation

    def learner_inference(self, batch):
        # Return uniform logits (initial state)
        # shape [B, Actions, Atoms] or [B, 1, Actions, Atoms]
        B = batch["observations"].shape[0]
        Actions = 2
        Atoms = 51
        logits = torch.zeros((B, 1, Actions, Atoms))
        return {"q_logits": logits}

def test_rainbow_targets():
    # 1. Setup CartPole parameters
    vmin = 0.0
    vmax = 500.0
    atoms = 51
    gamma = 0.99
    n_step = 1
    
    repr = C51Representation(vmin, vmax, atoms)
    network = MockNetwork(repr)
    
    builder = SingleStepTargetPipeline([
        DistributionalTargetBuilder(target_network=network, gamma=gamma, n_step=n_step)
    ])
    
    # 2. Create a batch: reward=1, done=False
    batch = {
        "observations": torch.zeros((1, 4)),
        "next_observations": torch.zeros((1, 4)),
        "rewards": torch.tensor([1.0]),
        "dones": torch.tensor([False]),
        "actions": torch.tensor([0]),
    }
    
    # 3. Build targets
    current_targets = {}
    builder.build_targets(batch, {}, network, current_targets)
    
    # 4. Check results
    target_dist = current_targets["q_logits"] # [B, 1, Atoms]
    print(f"Target Distribution Shape: {target_dist.shape}")
    
    expected_val = repr.to_expected_value(target_dist)
    print(f"Initial expected value (uniform): {repr.to_expected_value(torch.zeros(1, 1, atoms)).item()}")
    print(f"Target expected value: {expected_val.item()}")
    
    # 5. Iterative check: what is the fixed point?
    current_v = expected_val.item()
    for i in range(100):
        # If the whole distribution was at current_v, after one step it should be 1 + 0.99 * current_v
        current_v = 1.0 + 0.99 * current_v
    print(f"Fixed point after 100 steps: {current_v}")

if __name__ == "__main__":
    test_rainbow_targets()
