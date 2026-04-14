import torch
from core.blackboard import Blackboard
from core.path_resolver import write_blackboard_path

bb = Blackboard()
total_loss = torch.tensor(1.0)
optimizer_key = "default"

# Simulate LossAggregatorComponent returns
updates = {
    f"losses.total_loss.{optimizer_key}": total_loss,
    "losses.total_loss": total_loss
}

for path, value in updates.items():
    print(f"Writing {path}")
    write_blackboard_path(bb, path, value)
    print(f"bb.losses['total_loss'] type: {type(bb.losses.get('total_loss'))}")

print(f"Final state: {bb.losses['total_loss']}")

try:
    bb.losses['total_loss'].items()
    print("items() worked")
except Exception as e:
    print(f"items() failed: {e}")
