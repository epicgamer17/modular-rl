from torch import nn
from torch import Tensor

from configs.base import Config


class Projector(nn.Module):
    def __init__(self, input_dim: int, config: Config):
        super().__init__()

        # Paper defaults (SimSiam/EfficientZero)
        # Hidden dim is typically 2048 for ResNet50, but for MuZero
        # it usually matches the representation size or slightly larger.
        proj_hidden_dim = config.projector_hidden_dim
        proj_output_dim = config.projector_output_dim
        pred_hidden_dim = config.predictor_hidden_dim
        pred_output_dim = config.predictor_output_dim
        self.projection = nn.Sequential(
            nn.Linear(input_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
            nn.BatchNorm1d(proj_output_dim),
        )
        self.projection_head = nn.Sequential(
            nn.Linear(proj_output_dim, pred_hidden_dim),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(),
            nn.Linear(pred_hidden_dim, pred_output_dim),
        )

    def forward(self, x):
        x = self.projection(x)
        return self.projection_head(x)
