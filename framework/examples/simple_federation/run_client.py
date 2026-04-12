# examples/simple_federation/run_client.py
from data import get_mnist_loader
import fedlearn as fl
import argparse
import importlib
import sys
import os
from collections import OrderedDict
import torch

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../src')))


MODEL_REGISTRY = {
    "simple_cnn": ("model", "SimpleCNN"),
    "1m": ("model_1m", "Model1M"),
    "10m": ("model_10m", "Model10M"),
    "100m": ("model_100m", "Model100M"),
}


def load_model_class(model_key: str):
    module_name, class_name = MODEL_REGISTRY[model_key]
    mod = importlib.import_module(module_name)
    return getattr(mod, class_name)


class MnistClient(fl.Client):
    def __init__(self, client_id: int, model_cls, max_samples: int = 0):
        self.client_id = client_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = model_cls().to(self.device)
        loader = get_mnist_loader(client_id, num_clients=10)
        if max_samples > 0:
            from torch.utils.data import Subset, DataLoader
            sub = Subset(loader.dataset, list(
                range(min(max_samples, len(loader.dataset)))))
            self.trainloader = DataLoader(
                sub, batch_size=loader.batch_size, shuffle=True)
        else:
            self.trainloader = loader

    def get_parameters(self) -> OrderedDict[str, torch.Tensor]:
        return self.net.state_dict()

    def fit(self, parameters: OrderedDict[str, torch.Tensor], config: dict):
        self.net.load_state_dict(parameters)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.net.parameters(), lr=0.01)
        self.net.train()
        for _ in range(1):
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.net(images), labels)
                loss.backward()
                optimizer.step()

        return self.net.state_dict(), len(self.trainloader.dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedLearn Client")
    parser.add_argument("--server_address", type=str,
                        default="localhost:50051", help="Server address")
    parser.add_argument("--id", type=int, required=True, help="Client ID")
    parser.add_argument("--model", type=str, default="simple_cnn",
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Model to use (must match server)")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Max training samples per round (0 = use all)")
    args = parser.parse_args()

    model_cls = load_model_class(args.model)
    print(f"Using model: {args.model}")

    client = MnistClient(
        client_id=args.id, model_cls=model_cls, max_samples=args.max_samples)

    fl.client.start_client(
        server_address=args.server_address,
        client=client,
        client_id=f"client_{args.id}"
    )
