import torch.nn as nn
import torch.nn.functional as F


class Model100M(nn.Module):
    """~100M parameter wide MLP for MNIST (1x28x28 -> 10 classes).

    Uses a CNN feature extractor followed by very wide fully-connected layers
    to reach ~100M parameters.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 4096)
        self.fc4 = nn.Linear(4096, 4096)
        self.fc5 = nn.Linear(4096, 4096)
        self.fc6 = nn.Linear(4096, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 14x14
        x = self.pool(F.relu(self.conv2(x)))  # 7x7
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.dropout(F.relu(self.fc4(x)))
        x = self.dropout(F.relu(self.fc5(x)))
        x = self.fc6(x)
        return x


def get_model():
    return Model100M()
