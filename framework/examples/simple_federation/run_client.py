# examples/simple_federation/run_client.py
import argparse
import sys
import os
from collections import OrderedDict
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

import fedlearn as fl
from model import SimpleCNN
from data import get_mnist_loader


# 1. Define your custom client logic by inheriting from fl.Client
class MnistClient(fl.Client):
    def __init__(self, client_id: int):
        self.client_id = client_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = SimpleCNN().to(self.device)
        self.trainloader = get_mnist_loader(client_id, num_clients=10)  # Partition data

    def get_parameters(self) -> OrderedDict[str, torch.Tensor]:
        return self.net.state_dict()

    def fit(self, parameters: OrderedDict[str, torch.Tensor], config: dict):
        self.net.load_state_dict(parameters)

        # Training logic (similar to your old client.train method)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.net.parameters(), lr=0.01)
        self.net.train()
        for _ in range(2):  # 2 local epochs
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.net(images), labels)
                loss.backward()
                optimizer.step()

        return self.net.state_dict(), len(self.trainloader.dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedLearn Client")
    parser.add_argument("--server_address", type=str, default="localhost:50051", help="Server address")
    parser.add_argument("--id", type=int, required=True, help="Client ID")
    args = parser.parse_args()

    # 2. Instantiate your custom client
    client = MnistClient(client_id=args.id)

    # 3. Start the client
    fl.client.start_client(
        server_address=args.server_address,
        client=client,
        client_id=f"client_{args.id}"
    )