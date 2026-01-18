# Contributing to FedLearn

Thank you for your interest in contributing to FedLearn! This document provides guidelines for extending and modifying the framework.

## Table of Contents

- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Extending the Framework](#extending-the-framework)
- [Code Guidelines](#code-guidelines)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/FedLearn-Platform.git
cd FedLearn-Platform/framework
```

### 2. Install in Development Mode

```bash
# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install framework with dev dependencies
pip install -e ".[dev]"
```

### 3. Verify Installation

```bash
# Run tests
pytest tests/

# Check code style
ruff check src/
```

## Project Structure

```
framework/
├── src/fedlearn/              # Source code
│   ├── client/               # Client implementations
│   ├── server/               # Server coordination
│   ├── communication/        # gRPC and serialization
│   ├── core/                # Utilities
│   ├── data/                # Data handling
│   └── estimators/          # Gradient estimators
├── examples/                 # Example applications
│   ├── simple_federation/   # MNIST example
│   ├── llm_federation/      # LLM fine-tuning
│   └── ecg_federation/      # ECG classification
├── tests/                    # Unit tests
├── docs/                     # Documentation
└── setup.py                  # Package configuration
```

## Extending the Framework

### Adding a Custom Strategy

Create a new file `src/fedlearn/server/my_strategy.py`:

```python
from collections import OrderedDict
from typing import List, Tuple, Optional
import torch
from .strategy import Strategy

class MyStrategy(Strategy):
    """
    Custom aggregation strategy.
    
    This strategy implements [describe your approach].
    """
    
    def __init__(
        self,
        initial_parameters: OrderedDict[str, torch.Tensor],
        custom_param: float = 0.1,
    ):
        """
        Args:
            initial_parameters: Initial model parameters
            custom_param: Custom parameter for your strategy
        """
        self.parameters = initial_parameters
        self.custom_param = custom_param
    
    def aggregate_fit(
        self,
        results: List[Tuple[ClientProxy, FitRes]]
    ) -> Optional[OrderedDict[str, torch.Tensor]]:
        """
        Aggregate client updates.
        
        Args:
            results: List of (client, fit_result) tuples
            
        Returns:
            Aggregated parameters
        """
        # Extract parameters and weights
        parameters_list = []
        weights = []
        
        for _, fit_res in results:
            parameters_list.append(fit_res.parameters)
            weights.append(fit_res.num_examples)
        
        # Implement your aggregation logic
        aggregated = self._custom_aggregate(parameters_list, weights)
        
        return aggregated
    
    def _custom_aggregate(self, parameters_list, weights):
        """Your custom aggregation logic"""
        # Example: weighted average
        total_weight = sum(weights)
        
        aggregated_params = OrderedDict()
        for key in parameters_list[0].keys():
            weighted_sum = sum(
                params[key] * weight
                for params, weight in zip(parameters_list, weights)
            )
            aggregated_params[key] = weighted_sum / total_weight
        
        return aggregated_params
```

**Export your strategy** in `src/fedlearn/server/__init__.py`:

```python
from .my_strategy import MyStrategy

__all__ = ['MyStrategy', 'FedAvg', 'DeComFL']
```

**Use your strategy**:

```python
import fedlearn as fl

strategy = fl.MyStrategy(
    initial_parameters=model.state_dict(),
    custom_param=0.5
)

fl.server.start_server(
    server_address="0.0.0.0:50051",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy
)
```

### Adding a Custom Client

Create `src/fedlearn/client/my_client.py`:

```python
from collections import OrderedDict
import torch
from .client import Client

class MyClient(Client):
    """
    Custom client with specialized training loop.
    """
    
    def __init__(self, model, data_loader, special_config):
        self.model = model
        self.data_loader = data_loader
        self.special_config = special_config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    
    def get_parameters(self) -> OrderedDict[str, torch.Tensor]:
        """Return current model parameters"""
        return self.model.state_dict()
    
    def fit(
        self,
        parameters: OrderedDict[str, torch.Tensor],
        config: dict
    ) -> Tuple[OrderedDict[str, torch.Tensor], int]:
        """
        Custom training procedure.
        
        Args:
            parameters: Global model parameters
            config: Training configuration
            
        Returns:
            (updated_parameters, num_samples)
        """
        # Load global parameters
        self.model.load_state_dict(parameters)
        self.model.train()
        
        # Custom training loop
        optimizer = self._create_optimizer()
        
        for epoch in range(config.get('local_epochs', 5)):
            for X_batch, y_batch in self.data_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Your custom training step
                loss = self._training_step(X_batch, y_batch, optimizer)
        
        # Return updated parameters
        num_samples = len(self.data_loader.dataset)
        return self.model.state_dict(), num_samples
    
    def _create_optimizer(self):
        """Create custom optimizer"""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.special_config['learning_rate']
        )
    
    def _training_step(self, X, y, optimizer):
        """Custom training step"""
        optimizer.zero_grad()
        outputs = self.model(X)
        loss = torch.nn.functional.cross_entropy(outputs, y)
        loss.backward()
        optimizer.step()
        return loss.item()
```

### Adding Data Utilities

Add new functions to `src/fedlearn/data/utils.py`:

```python
import numpy as np
from typing import List, Tuple

def custom_partition(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    partition_type: str = 'custom'
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Custom data partitioning strategy.
    
    Args:
        X: Feature array
        y: Label array
        num_clients: Number of clients
        partition_type: Type of partitioning
        
    Returns:
        List of (X_client, y_client) for each client
    """
    # Implement your partitioning logic
    client_data = []
    
    # Example: split by some criterion
    samples_per_client = len(X) // num_clients
    
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client
        
        X_client = X[start_idx:end_idx]
        y_client = y[start_idx:end_idx]
        
        client_data.append((X_client, y_client))
    
    return client_data
```

### Adding Examples

Create a new example in `examples/my_example/`:

```
examples/my_example/
├── README.md              # Example-specific documentation
├── run_server.py         # Server script
├── run_client.py         # Client script
├── config.py             # Configuration
├── data.py               # Data loading utilities
└── model.py              # Model definition
```

**Example `run_server.py`**:

```python
import argparse
import sys
import os

# Add framework to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
import fedlearn as fl

from model import MyModel
from config import CONFIG

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--num_rounds", type=int, default=10)
    parser.add_argument("--num_clients", type=int, default=3)
    args = parser.parse_args()
    
    # Initialize model
    model = MyModel()
    initial_parameters = model.state_dict()
    
    # Create strategy
    strategy = fl.FedAvg(
        initial_parameters=initial_parameters,
        min_fit_clients=args.num_clients,
    )
    
    # Start server
    print(f"Starting server on 0.0.0.0:{args.port}")
    fl.server.start_server(
        server_address=f"0.0.0.0:{args.port}",
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
```

## Code Guidelines

### Style Guide

We follow PEP 8 with some modifications:

- **Line length**: 100 characters max
- **Imports**: Group into standard library, third-party, local
- **Type hints**: Required for all public APIs
- **Docstrings**: Google style for all public classes/functions

**Example**:

```python
from typing import List, Tuple, Optional
import torch
import numpy as np

from fedlearn.core import BaseClass

class MyClass(BaseClass):
    """
    One-line summary of the class.
    
    Longer description of what this class does and when to use it.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Attributes:
        attr1: Description of attr1
        attr2: Description of attr2
        
    Example:
        ```python
        obj = MyClass(param1=10, param2="test")
        result = obj.method()
        ```
    """
    
    def __init__(self, param1: int, param2: str):
        self.param1 = param1
        self.param2 = param2
    
    def method(self, arg: float) -> Tuple[int, str]:
        """
        Short description of method.
        
        Args:
            arg: Description of arg
            
        Returns:
            Tuple containing (result1, result2)
            
        Raises:
            ValueError: If arg is negative
        """
        if arg < 0:
            raise ValueError("arg must be non-negative")
        
        return self.param1, self.param2
```

### Code Quality Tools

```bash
# Format code
ruff format src/

# Lint code
ruff check src/

# Type checking (optional)
mypy src/fedlearn
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `FedAvgStrategy`)
- **Functions/Methods**: `snake_case` (e.g., `aggregate_parameters`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_TIMEOUT`)
- **Private**: Prefix with `_` (e.g., `_internal_method`)

## Testing

### Writing Tests

Create tests in `tests/` directory:

```python
# tests/test_my_strategy.py
import pytest
import torch
from collections import OrderedDict
from fedlearn.server import MyStrategy

class TestMyStrategy:
    """Tests for MyStrategy"""
    
    def test_initialization(self):
        """Test strategy initialization"""
        params = OrderedDict({'weight': torch.randn(10, 10)})
        strategy = MyStrategy(initial_parameters=params)
        assert strategy.parameters is not None
    
    def test_aggregate_fit(self):
        """Test aggregation logic"""
        params = OrderedDict({'weight': torch.randn(10, 10)})
        strategy = MyStrategy(initial_parameters=params)
        
        # Mock client results
        results = [
            (None, FitRes(parameters=params, num_examples=100)),
            (None, FitRes(parameters=params, num_examples=150)),
        ]
        
        aggregated = strategy.aggregate_fit(results)
        assert aggregated is not None
        assert 'weight' in aggregated
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_my_strategy.py

# Run with coverage
pytest --cov=src/fedlearn tests/

# Run specific test
pytest tests/test_my_strategy.py::TestMyStrategy::test_initialization
```

### Test Coverage

Aim for:
- **Core modules**: 80%+ coverage
- **Strategies**: 90%+ coverage
- **Utilities**: 70%+ coverage

## Submitting Changes

### 1. Create a Branch

```bash
git checkout -b feature/my-new-feature
```

Branch naming:
- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation
- `refactor/description` - Code refactoring

### 2. Make Changes

- Write clean, documented code
- Follow style guidelines
- Add tests for new functionality
- Update documentation

### 3. Test Locally

```bash
# Run tests
pytest tests/

# Check code style
ruff check src/

# Test examples
cd examples/simple_federation
python run_server.py --num_rounds 2 &
python run_client.py --id 0 --num_clients 1
```

### 4. Commit Changes

```bash
git add .
git commit -m "feat: add custom aggregation strategy

- Implement MyStrategy with weighted averaging
- Add tests for aggregation logic
- Update documentation"
```

Commit message format:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `test:` Tests
- `refactor:` Code refactoring

### 5. Push and Create PR

```bash
git push origin feature/my-new-feature
```

Then create a Pull Request on GitHub with:
- Clear description of changes
- Link to related issues
- Test results
- Documentation updates

## Pull Request Checklist

Before submitting:

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Examples work correctly
- [ ] No merge conflicts
- [ ] Commits are clean and descriptive

## Getting Help

- **Questions**: Open a [GitHub Discussion](https://github.com/Learning-Optimization-Group/FedLearn-Platform/discussions)
- **Bugs**: Create an [Issue](https://github.com/Learning-Optimization-Group/FedLearn-Platform/issues)
- **Documentation**: Check [docs/](docs/)

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Help newcomers
- Focus on collaboration

Thank you for contributing to FedLearn!