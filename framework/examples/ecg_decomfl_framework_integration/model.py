# examples/decomfl/model.py
"""
Neural Network Model for ECG Classification
"""

import torch
import torch.nn as nn


class ECGModel(nn.Module):
    """
    Simple MLP for ECG binary classification.
    Used in DeComFL federated learning.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 2):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes
        """
        super(ECGModel, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(hidden_dim, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)

        return x


def create_model(input_dim: int, hidden_dim: int = 64, num_classes: int = 2, device: str = 'cpu'):
    """
    Factory function to create ECG model

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        num_classes: Number of output classes
        device: Device to place model on

    Returns:
        model: ECGModel instance
    """
    model = ECGModel(input_dim, hidden_dim, num_classes)
    model = model.to(device)
    return model