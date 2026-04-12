# examples/simple_federation/run_server.py
import argparse
import importlib
import sys
import os
from collections import OrderedDict
import torch

import fedlearn as fl
from data import get_test_loader

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


device = "cuda" if torch.cuda.is_available() else "cpu"


def make_evaluate_fn(model_cls):
    def server_side_evaluate(server_round: int, parameters: OrderedDict[str, torch.Tensor]):
        model = model_cls()
        model.load_state_dict(parameters)
        model.to(device)

        testloader = get_test_loader()
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0

        model.eval()
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = loss / len(testloader)
        accuracy = correct / total
        return avg_loss, {"accuracy": accuracy}

    return server_side_evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedLearn Server")
    parser.add_argument("--port", type=int, default=50051, help="Server port")
    parser.add_argument("--num_rounds", type=int, default=5,
                        help="Number of training rounds")
    parser.add_argument("--model", type=str, default="simple_cnn",
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Model to use (simple_cnn, 1m, 10m, 100m)")
    parser.add_argument("--min_clients", type=int, default=2,
                        help="Minimum clients required per round for aggregation (default: 2)")
    args = parser.parse_args()

    model_cls = load_model_class(args.model)
    net = model_cls()
    n_params = sum(p.numel() for p in net.parameters())
    print(
        f"Model: {args.model} ({n_params:,} params, {n_params * 4 / 1e6:.1f} MB)")
    print(f"Clients per round: {args.min_clients}")

    initial_parameters = net.state_dict()

    strategy = fl.FedAvg(
        initial_parameters=initial_parameters,
        evaluate_fn=make_evaluate_fn(model_cls),
        min_fit_clients=args.min_clients,
        clients_per_round=args.min_clients,
    )

    fl.server.start_server(
        server_address=f"0.0.0.0:{args.port}",
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )
