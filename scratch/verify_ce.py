import torch
import torch.nn.functional as F

B, T, K = 2, 4, 5
preds = torch.randn(B, T, K)
targets = F.one_hot(torch.randint(0, K, (B, T)), num_classes=K).float()

flat_preds = preds.flatten(0, 1)
flat_targets = targets.flatten(0, 1)

raw_loss = F.cross_entropy(flat_preds, flat_targets, reduction="none")
print(f"raw_loss shape: {raw_loss.shape}")
print(f"raw_loss values: {raw_loss}")
print(f"raw_loss positive: {torch.all(raw_loss >= 0)}")
