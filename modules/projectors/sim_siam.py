from torch import nn
from torch import Tensor


class Projector(nn.Module):
    def __init__(
        self,
        input_dim: int,
        projector_hidden_dim: int,
        projector_output_dim: int,
        predictor_hidden_dim: int,
        predictor_output_dim: int,
    ):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, projector_hidden_dim),
            nn.BatchNorm1d(projector_hidden_dim),
            nn.ReLU(),
            nn.Linear(projector_hidden_dim, projector_hidden_dim),
            nn.BatchNorm1d(projector_hidden_dim),
            nn.ReLU(),
            nn.Linear(projector_hidden_dim, projector_output_dim),
            nn.BatchNorm1d(projector_output_dim),
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(projector_output_dim, predictor_hidden_dim),
            nn.BatchNorm1d(predictor_hidden_dim),
            nn.ReLU(),
            nn.Linear(predictor_hidden_dim, predictor_output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        return self.prediction_head(x)
