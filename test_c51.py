import torch
import torch.nn.functional as F


class MockConfig:
    def __init__(self):
        self.discount_factor = 0.99
        self.n_step = 1
        self.v_min = 0.0
        self.v_max = 50.0
        self.atom_size = 51


class MockC51LossModule:
    def __init__(self):
        self.config = MockConfig()
        self.device = torch.device("cpu")
        self.support = torch.linspace(
            self.config.v_min,
            self.config.v_max,
            self.config.atom_size,
            device=self.device,
        )

    def _project_target_distribution(
        self,
        rewards: torch.Tensor,
        terminal_mask: torch.Tensor,
        next_probs: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = rewards.shape[0]
        discount = self.config.discount_factor**self.config.n_step
        delta_z = (self.config.v_max - self.config.v_min) / (self.config.atom_size - 1)

        tz = (
            rewards.view(-1, 1)
            + discount * (~terminal_mask).view(-1, 1) * self.support.view(1, -1)
        ).clamp(self.config.v_min, self.config.v_max)
        b = (tz - self.config.v_min) / delta_z
        lower = b.floor().long().clamp(0, self.config.atom_size - 1)
        upper = b.ceil().long().clamp(0, self.config.atom_size - 1)

        same = lower == upper
        upper[same & (upper < self.config.atom_size - 1)] += 1
        lower[same & (upper == self.config.atom_size - 1)] -= 1

        projected = torch.zeros((batch_size, self.config.atom_size), device=self.device)
        try:
            projected.scatter_add_(1, lower, next_probs * (upper.float() - b))
            projected.scatter_add_(1, upper, next_probs * (b - lower.float()))
            print("Scatter add successful.")
        except Exception as e:
            print("Error during scatter_add:", e)
        return projected


module = MockC51LossModule()
print("Testing b = 50.0 (v_max)")
rewards = torch.tensor([50.0])  # v_max
terminal_mask = torch.tensor([True])
next_probs = torch.zeros((1, 51))
next_probs[0, 50] = 1.0  # Prob 1 at the max atom
projected = module._project_target_distribution(rewards, terminal_mask, next_probs)
print("Projected sum:", projected.sum().item(), "Prob at 50:", projected[0, 50].item())

print("\\nTesting b = 2.0")
rewards = torch.tensor([2.0])
terminal_mask = torch.tensor([True])
next_probs = torch.zeros((1, 51))
next_probs[0, 2] = 1.0
projected = module._project_target_distribution(rewards, terminal_mask, next_probs)
print("Projected sum:", projected.sum().item(), "Prob at 2:", projected[0, 2].item())

print("\\nTesting b = 0.0")
rewards = torch.tensor([0.0])
terminal_mask = torch.tensor([True])
next_probs = torch.zeros((1, 51))
next_probs[0, 0] = 1.0
projected = module._project_target_distribution(rewards, terminal_mask, next_probs)
print("Projected sum:", projected.sum().item(), "Prob at 0:", projected[0, 0].item())
