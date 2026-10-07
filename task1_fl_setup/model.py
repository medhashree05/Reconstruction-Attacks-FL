import torch.nn as nn


class LeNet(nn.Module):
    """Small CNN with Sigmoid activations.

    Sigmoid (not ReLU) is deliberate: DLG needs a twice-differentiable model
    to optimise through the gradient, and it converges far better with smooth
    activations. This is the same style of model used in the DLG paper.
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 12, 5, padding=2, stride=2), nn.Tanh(),   # 28 -> 14
            nn.Conv2d(12, 12, 5, padding=2, stride=2), nn.Tanh(),  # 14 -> 7
            nn.Conv2d(12, 12, 5, padding=2, stride=1), nn.Tanh(),  # 7 -> 7
        )
        self.fc = nn.Linear(12 * 7 * 7, num_classes)

    def forward(self, x):
        return self.fc(self.body(x).flatten(1))
