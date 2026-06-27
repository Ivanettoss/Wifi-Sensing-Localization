
#Baseline MLP classifier for CSI reference point classification.
# Input shape: (batch_size, num_antennas, num_subcarriers, window_size)
# For our current dataset: (batch_size, 3, 30, 30)
#First approach: matrix is flatten into a one size vector of size 3*30*30 = 2700

import torch
from torch import nn


class MLPClassifier(nn.Module):

    def __init__(
        self,
        input_dim: int = 2700,
        hidden_dim_1: int = 512,
        hidden_dim_2: int = 256,
        num_classes: int = 100,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim_2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute class logits from a batch of CSI windows.
        """

        logits = self.network(x)

        return logits


def count_trainable_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a PyTorch model.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )