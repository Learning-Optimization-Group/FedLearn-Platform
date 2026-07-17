# FedLearn - Distributed Federated Learning Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](../LICENSE)

## Overview

FedLearn is a flexible and extensible federated learning framework designed for distributed machine learning across multiple clients. **Built from scratch — this package carries no Flower / `flwr` dependency and borrows none of its semantics.** Communication is direct gRPC over a custom protobuf contract (`fedlearn.proto`, package `fedlearn.v2`); the framework supports CNNs, Transformers, and LLMs with pluggable aggregation strategies (FedAvg and DeComFL). The FL-server process supervisor in the Spring Boot backend is `FlServerManager` (package `com.federated.fl_platform_api.orchestration`) — it shells out to the `fl-runtime/` scripts and has nothing to do with Flower.

> **Scope note:** `flwr` is absent from `framework/requirements.txt`. Elsewhere in the repo (`backend/fl-platform-api/requirements.txt`, `client-docker/requirements.txt`) `flwr` / `flwr-datasets` are pulled in **for dataset partitioning only** — never for FL server/client/strategy semantics.

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

# Terminal 1: Start server (accepts --port and --num_rounds)
python run_server.py --num_rounds 10

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

# Terminal 1: Start server (cohort size is set by --clients_per_round / --min_fit_clients)
python run_server.py --dataset cb --num_rounds 5 --clients_per_round 3

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
├── src/fedlearn/            # Core federated learning package
│   ├── backbone/            # Frozen-backbone distribution
│   ├── bundle/              # Adapter-bundle packaging
│   ├── client/              # Client implementations
│   ├── communication/       # gRPC, safetensors serialization
│   ├── estimators/          # DeComFL estimators
│   ├── fot/                 # Federation over Text (research mode)
│   ├── privacy/             # Differential-privacy accounting
│   ├── security/            # TLS + connection-token plumbing
│   └── server/              # Server, coordinator, strategies
├── examples/                # Ready-to-run examples (full set in examples/)
│   ├── simple_federation/   # MNIST + CNN
│   ├── llm_federation/      # OPT-125M fine-tuning
│   ├── ecg_federation/      # ECG classification
│   └── fot_text_federation/ # Federation over Text, offline demo
└── docs/                    # Documentation
```

## Citation

If you use FedLearn in your research, please cite:

```bibtex
@inproceedings{li2025decomfl,
  title={Achieving Dimension-Free Communication in Federated Learning via Zeroth-Order Optimization},
  author={Li, Zhe and Ying, Bicheng and Liu, Zidong and Dong, Chaosheng and Yang, Haibo},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

## Research

This framework implements algorithms from:
- **DeComFL** — *Achieving Dimension-Free Communication in Federated Learning via Zeroth-Order Optimization* (ICLR 2025, [arXiv:2405.15861](https://arxiv.org/abs/2405.15861)); reference implementation [ZidongLiu/DeComFL](https://github.com/ZidongLiu/DeComFL) (Apache-2.0)
- **Federation over Text (FoT)** — an additive, local-LLM-only text-federation research mode, orthogonal to the gradient path ([arXiv:2604.16778](https://arxiv.org/abs/2604.16778)); see [`src/fedlearn/fot/`](src/fedlearn/fot/)
- Developed at Rochester Institute of Technology under Professor Haibo Yang

DeComFL paper-to-code mapping (Algorithms 2–4 → `server/decomfl_strategy.py`, `client/decomfl_client.py`, `estimators/`): [`wikis/framework/06_decomfl.md`](../wikis/framework/06_decomfl.md).

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- How to extend the framework
- Adding custom strategies
- Code style guidelines
- Testing requirements

## License

This project is licensed under the Apache License 2.0 - see [LICENSE](../LICENSE) for details.

## Support

- 📖 **Documentation**: [docs/](docs/)
- 🐛 **Issues**: [GitHub Issues](https://github.com/Learning-Optimization-Group/FedLearn-Platform/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/Learning-Optimization-Group/FedLearn-Platform/discussions)

## Acknowledgments

Developed by the Learning Optimization Group at Rochester Institute of Technology.

**Principal Investigator:** Professor Haibo Yang

---

**Getting Started:** New to federated learning? Start with our [Quick Start Guide](docs/quickstart.md) and [Simple Federation Example](docs/examples/simple-federation.md).