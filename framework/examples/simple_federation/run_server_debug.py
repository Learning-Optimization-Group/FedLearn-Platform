# examples/simple_federation/run_server.py
import argparse
print("DEBUG: After torch import")
import sys
import os
from collections import OrderedDict
import torch

# Adjust path to find the framework
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

import fedlearn as fl
from model import SimpleCNN  # Your shared model
from data import get_test_loader
print("DEBUG: After all imports")
# 1. Define the model and initial parameters
net = SimpleCNN()
print("DEBUG: Created SimpleCNN")
initial_parameters = net.state_dict()
print("DEBUG: Got state_dict")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"DEBUG: Device = {device}")


# 2. Define a server-side evaluation function
def server_side_evaluate(server_round: int, parameters: OrderedDict[str, torch.Tensor]):
    model = SimpleCNN()
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
    print("DEBUG: Defined evaluate function")



if __name__ == "__main__":
    print("DEBUG: Entered main block")
    
    parser = argparse.ArgumentParser(description="FedLearn Server")
    parser.add_argument("--port", type=int, default=50051, help="Server port")
    parser.add_argument("--num_rounds", type=int, default=5, help="Number of training rounds")
    args = parser.parse_args()
    print(f"DEBUG: Parsed args: port={args.port}, rounds={args.num_rounds}")
    
    # 3. Define the strategy
    strategy = fl.FedAvg(
        initial_parameters=initial_parameters,
        evaluate_fn=server_side_evaluate,
        min_fit_clients=2,
    )
    print("DEBUG: Created strategy")
    
    # 4. Start the server
    print("DEBUG: About to start server...")
    fl.server.start_server(
        server_address=f"0.0.0.0:{args.port}",
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )
    print("DEBUG: Server started (you won't see this)")