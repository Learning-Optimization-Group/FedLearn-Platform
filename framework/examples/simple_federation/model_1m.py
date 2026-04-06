import torch.nn as nn
import torch.nn.functional as F


class Model1M(nn.Module):
    """~1M parameter CNN for MNIST (1x28x28 -> 10 classes)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 3 * 3, 700)
        self.fc2 = nn.Linear(700, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))          # 28x28
        x = self.pool(F.relu(self.conv2(x)))  # 14x14
        x = self.pool(F.relu(self.conv3(x)))  # 7x7
        x = self.pool(F.relu(self.conv4(x)))  # 3x3
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_model():
    return Model1M()
