# =============================================================================
# run_client.py - Federated Learning Client for CB and SST-2
# =============================================================================

import argparse
from collections import OrderedDict
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
import fedlearn as fl
from transformers import AutoModelForSequenceClassification
from data import get_llm_loaders
from config import MODEL_NAME, DATASET_CONFIGS


class LLMClient(fl.Client):
    """
    Custom LLM client for sequence classification tasks.
    Supports both CB (3-class) and SST-2 (2-class) datasets.
    """

    def __init__(self, client_id: int, dataset_name: str, num_clients: int, data_fraction: float = 1.0):
        self.client_id = client_id
        self.dataset_name = dataset_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Get dataset configuration
        self.config = DATASET_CONFIGS[dataset_name]

        # Load model for sequence classification
        self.net = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=self.config.num_classes
        ).to(self.device)

        # Load data
        self.trainloader, _, _ = get_llm_loaders(
            dataset_name=dataset_name,
            client_id=client_id,
            num_clients=num_clients,
            data_fraction=data_fraction
        )

        print(f"Client {client_id} initialized:")
        print(f"  Dataset: {dataset_name}")
        print(f"  Device: {self.device}")
        print(f"  Training samples: {len(self.trainloader.dataset)}")
        print(f"  Num classes: {self.config.num_classes}")

    def get_parameters(self) -> OrderedDict[str, torch.Tensor]:
        return self.net.state_dict()

    def fit(self, parameters: OrderedDict[str, torch.Tensor], config: dict):
        """
        Train the model locally for one epoch (K=1 as per Professor Yang's setup).
        """
        self.net.load_state_dict(parameters)

        # Use AdamW optimizer with dataset-specific learning rate
        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.config.learning_rate
        )

        self.net.train()
        total_loss = 0.0
        num_batches = 0

        # K=1: One local epoch
        for epoch in range(self.config.local_epochs):
            for batch in self.trainloader:
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)


                labels = batch["labels"].to(self.device)

                # Forward pass
                outputs = self.net(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"  Client {self.client_id} - Avg Loss: {avg_loss:.4f}")

        return self.net.state_dict(), len(self.trainloader.dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedLearn LLM Client")
    parser.add_argument("--server_address", type=str, default="localhost:50051", help="Server address")
    parser.add_argument("--id", type=int, required=True, help="Client ID")
    parser.add_argument("--dataset", type=str, default="cb", choices=["cb", "sst2"], help="Dataset to use")
    parser.add_argument("--num_clients", type=int, default=8, help="Total number of clients")
    parser.add_argument("--data_fraction", type=float, default=1.0, help="Fraction of data to use (0.1 = 10%)")
    args = parser.parse_args()

    # Instantiate custom client
    client = LLMClient(
        client_id=args.id,
        dataset_name=args.dataset,
        num_clients=args.num_clients,
        data_fraction=args.data_fraction
    )

    # Start the client
    fl.client.start_client(
        server_address=args.server_address,
        client=client,
        client_id=f"client_{args.id}"
    )


# cd examples/llm_federation python run_client.py --server_address localhost:50051 --id 7 --dataset cb --num_clients 8