# FedLearn - Distributed Federated Learning Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

## Overview

FedLearn is a flexible and extensible federated learning framework designed for distributed machine learning across multiple clients. Built on top of Flower (flwr) with gRPC communication, it supports training both CNNs and Large Language Models with custom aggregation strategies and Byzantine-robust optimization techniques.

**Key Features:**
- 🌐 **Server-Client Architecture** - Efficient gRPC-based communication
- 🤖 **Multi-Model Support** - CNNs, Transformers, and LLMs (OPT, GPT-2, etc.)
- 🔄 **Custom Strategies** - FedAvg, DeComFL, and extensible strategy system
- 📊 **Non-IID Data** - Dirichlet-based data partitioning for realistic scenarios
- ⚡ **Optimized Training** - Mixed precision, gradient scaling, learning rate scheduling
- 🐳 **Docker Ready** - Easy client deployment and scaling

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Learning-Optimization-Group/FedLearn-Platform.git
cd FedLearn-Platform/framework

# Install dependencies
pip install -r requirements.txt

# Install framework in development mode
pip install -e .
```

**Requirements:** Python 3.10+, CUDA-capable GPU (optional but recommended for LLM training)

### 5-Minute Example

**Server:**
```python
import fedlearn as fl
import torch

# Initialize model
model = YourModel()
initial_parameters = model.state_dict()

# Define strategy
strategy = fl.FedAvg(
    initial_parameters=initial_parameters,
    min_fit_clients=2,
    clients_per_round=5
)

# Start server
fl.server.start_server(
    server_address="0.0.0.0:50051",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy
)
```

**Client:**
```python
import fedlearn as fl

# Create custom client
class MyClient(fl.Client):
    def get_parameters(self):
        return self.model.state_dict()
    
    def fit(self, parameters, config):
        # Local training logic
        return updated_parameters, num_samples

client = MyClient()

# Connect to server
fl.client.start_client(
    server_address="localhost:50051",
    client=client,
    client_id="client_0"
)
```

## Examples

### 1. Simple Federation (MNIST + CNN)
Basic federated learning with MNIST dataset and CNN model.

```bash
cd examples/simple_federation

# Terminal 1: Start server
python run_server.py --num_rounds 10 --num_clients 3

# Terminal 2-4: Start clients
python run_client.py --id 0 --server_address localhost:50051
python run_client.py --id 1 --server_address localhost:50051
python run_client.py --id 2 --server_address localhost:50051
```

**Expected Performance:** ~95% accuracy after 10 rounds

### 2. LLM Federation (OPT-125M)
Fine-tune OPT-125M on SuperGLUE CommitmentBank (CB) dataset.

```bash
cd examples/llm_federation

# Terminal 1: Start server
python run_server.py --dataset cb --num_rounds 5 --num_clients 3

# Terminal 2-4: Start clients
python run_client.py --id 0 --dataset cb --server_address localhost:50051
python run_client.py --id 1 --dataset cb --server_address localhost:50051
python run_client.py --id 2 --dataset cb --server_address localhost:50051
```

**Expected Performance:** ~83% accuracy on CB dataset after 5 rounds

### 3. ECG Classification (Transformer)
Binary ECG signal classification (Normal/Abnormal) using Transformer architecture.

```bash
cd examples/ecg_federation

# Terminal 1: Start server
python run_server.py --data_path ecg_data/ecg.csv --num_clients 5 --num_rounds 3

# Terminal 2-6: Start clients
python run_client.py --id 0 --data_path ecg_data/ecg.csv --server_address localhost:50051
python run_client.py --id 1 --data_path ecg_data/ecg.csv --server_address localhost:50051
# ... (start remaining clients)
```

**Expected Performance:** ~93.80% accuracy after 3 rounds

## Architecture

```
┌─────────┐  ┌─────────┐  ┌─────────┐
│Client 1 │  │Client 2 │  │Client 3 │  ...
└────┬────┘  └────┬────┘  └────┬────┘
     │            │            │
     └────────────┼────────────┘
                  │ gRPC
           ┌──────▼──────┐
           │   Server    │
           │  (Strategy) │
           └──────┬──────┘
                  │
          ┌───────▼────────┐
          │  Global Model  │
          └────────────────┘
```

FedLearn uses gRPC for efficient client-server communication and supports various aggregation strategies (FedAvg, DeComFL) for combining model updates from distributed clients.

## Documentation

📚 **Comprehensive Documentation:**

- **[Installation Guide](docs/installation.md)** - Detailed setup and troubleshooting
- **[Quick Start](docs/quickstart.md)** - Get started in 5 minutes
- **[API Reference](docs/api-reference/)** - Complete framework API
  - [Server API](docs/api-reference/server.md)
  - [Client API](docs/api-reference/client.md)
  - [Strategies](docs/api-reference/strategies.md)
  - [Core Modules](docs/api-reference/core-modules.md)
- **[Examples](docs/examples/)** - Detailed walkthroughs
  - [Simple Federation](docs/examples/simple-federation.md)
  - [LLM Federation](docs/examples/llm-federation.md)
  - [ECG Classification](docs/examples/ecg-federation.md)
- **[Advanced](docs/advanced/)** - Extend the framework
  - [Custom Strategies](docs/advanced/custom-strategies.md)
  - [Extending Framework](docs/advanced/extending-framework.md)

## Framework Structure

```
framework/
├── src/fedlearn/           # Core federated learning package
│   ├── client/            # Client implementations
│   ├── server/            # Server and coordination logic
│   ├── communication/     # gRPC and serialization
│   ├── core/             # Core utilities
│   ├── data/             # Data handling utilities
│   └── estimators/       # DeComFL estimators
├── examples/              # Ready-to-run examples
│   ├── simple_federation/ # MNIST + CNN
│   ├── llm_federation/    # OPT-125M fine-tuning
│   └── ecg_federation/    # ECG classification
└── docs/                  # Documentation
```

## Citation

If you use FedLearn in your research, please cite:

```bibtex
@article{yang2024decomfl,
  title={DeComFL: Decomposed Federated Learning},
  author={Yang, Haibo and [Co-authors]},
  journal={[Journal/Conference]},
  year={2024}
}
```

## Research

This framework implements algorithms from:
- **DeComFL** - Decomposed Federated Learning with Byzantine-robust aggregation
- Developed at Rochester Institute of Technology under Professor Haibo Yang

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- How to extend the framework
- Adding custom strategies
- Code style guidelines
- Testing requirements

## License

This project is licensed under the Apache License 2.0 - see [LICENSE](LICENSE) for details.

## Support

- 📖 **Documentation**: [docs/](docs/)
- 🐛 **Issues**: [GitHub Issues](https://github.com/Learning-Optimization-Group/FedLearn-Platform/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/Learning-Optimization-Group/FedLearn-Platform/discussions)

## Acknowledgments

Developed by the Learning Optimization Group at Rochester Institute of Technology.

**Principal Investigator:** Professor Haibo Yang

---

**Getting Started:** New to federated learning? Start with our [Quick Start Guide](docs/quickstart.md) and [Simple Federation Example](docs/examples/simple-federation.md).