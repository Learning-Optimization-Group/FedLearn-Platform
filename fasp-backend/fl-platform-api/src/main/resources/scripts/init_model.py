import argparse
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from transformers import AutoModelForSequenceClassification
import os
from collections import OrderedDict

# ==============================================================================
# --- Model Definitions ---
# ==============================================================================
class CnnNet(nn.Module):
    """A simple CNN for CIFAR-10, identical to the one in client.py."""
    def __init__(self) -> None:
        super(CnnNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def get_model(model_type: str, model_name: str, device: str):
    """Returns a model instance based on the user's selection."""
    print(f"Loading model: {model_type} / {model_name}")
    model_type = model_type.upper()
    model_name = model_name.lower()

    if model_type == 'CNN':
        return CnnNet().to(device)

    elif model_type == 'TRANSFORMER':
        if model_name == 'opt-125m':
            print("Loading pre-trained 'facebook/opt-125m' for sequence classification.")
            return AutoModelForSequenceClassification.from_pretrained(
                "facebook/opt-125m", num_labels=3, use_safetensors=True
            ).to(device)
        else:
            raise ValueError(f"Unsupported Transformer model: {model_name}")

    else:
        raise ValueError(f"Unsupported model architecture: {model_type}")

# ==============================================================================
# --- Training Function ---
# ==============================================================================
def get_optimizer(optimizer_name: str, model_parameters):
    """Returns a PyTorch optimizer instance based on its name."""
    print(f"Using optimizer: {optimizer_name}")
    optimizer_name = optimizer_name.lower()
    if optimizer_name == 'adamw':
        return torch.optim.AdamW(model_parameters)
    elif optimizer_name == 'sgd':
        return torch.optim.SGD(model_parameters, lr=0.01, momentum=0.9)
    elif optimizer_name == 'rmsprop':
        return torch.optim.RMSprop(model_parameters)

    if optimizer_name != 'adam':
        print(f"Warning: Optimizer '{optimizer_name}' not recognized. Defaulting to Adam.")
    return torch.optim.Adam(model_parameters)

def train(net, trainloader, epochs: int, optimizer_name: str, device: str):
    """Train the network on the training set with a specified optimizer."""
    print(f"Starting CNN pre-training on device: {device}...")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = get_optimizer(optimizer_name, net.parameters())
    net.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in trainloader:
            images, labels = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}: pre-train loss {epoch_loss / len(trainloader):.4f}")

# ==============================================================================
# --- Main Execution Block ---
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Initialize and optionally pre-train a model.")
    parser.add_argument("--model-type", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--optimizer", type=str, required=True)
    parser.add_argument("--out", type=str, required=True, help="Output path for the .npz file")
    parser.add_argument("--pretrain-epochs", type=int, default=0)
    args = parser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    net = get_model(args.model_type, args.model_name, DEVICE)

    if args.model_type.lower() == "transformer":
        if args.pretrain_epochs > 0:
            print("--- WARNING: Server-side pre-training is not applicable for pre-trained LLMs. --pretrain-epochs ignored. ---")
        print("--- Initializing Transformer from public Hugging Face checkpoint. ---")

    elif args.model_type.lower() == "cnn":
        if args.pretrain_epochs > 0:
            print(f"--- Starting server-side pre-training for CNN for {args.pretrain_epochs} epochs... ---")
            pytorch_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            full_trainset = CIFAR10(root="./data", train=True, download=True, transform=pytorch_transforms)
            pretrain_subset = Subset(full_trainset, range(1000))
            pretrain_loader = DataLoader(pretrain_subset, batch_size=32, shuffle=True)
            train(net, pretrain_loader, epochs=args.pretrain_epochs, optimizer_name=args.optimizer, device=DEVICE)
            print("--- Pre-training complete. ---")
        else:
            print("--- No pre-training requested. Initializing CNN with random weights. ---")


    print(f"Saving initial model weights...")
    state_dict = net.state_dict()
    params_to_save = {
        key.replace('.', '__DOT__'): tensor.detach().cpu().numpy()
        for key, tensor in state_dict.items()
        if torch.is_tensor(tensor)
    }

    output_dir = os.path.dirname(args.out)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    np.savez(args.out, **params_to_save)
    print(f"Initial model weights with layer names saved to {args.out}")

if __name__ == "__main__":
    main()