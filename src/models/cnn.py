# 2-D CNN baseline for CSI reference point classification.
# Input shape:(batch_size, num_antennas, num_subcarriers, window_size)
#Second approach: treat the input as a 2-D image with 3 channels

import torch
from torch import nn


class CNN2DClassifier(nn.Module):

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 176,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute class logits from a batch of CSI windows.
        """

        features = self.feature_extractor(x)
        logits = self.classifier(features)

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