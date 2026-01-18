# Quick Start Guide

Get started with FedLearn in 5 minutes. This guide walks you through running your first federated learning experiment.

## Prerequisites

- FedLearn installed ([Installation Guide](installation.md))
- Python 3.10+
- Basic understanding of machine learning

## Your First Federated Learning Experiment

We'll train a simple CNN on MNIST across 3 clients in 10 rounds.

### Step 1: Run the Server

Open a terminal and start the server:

```bash
cd FedLearn-Platform/framework/examples/simple_federation
python run_server.py --num_rounds 10 --num_clients 3 --port 50051
```

You should see:
```
Starting server on 0.0.0.0:50051
Waiting for 3 clients to connect...
```

### Step 2: Run Clients

Open **three new terminals** and start each client:

**Terminal 2:**
```bash
cd FedLearn-Platform/framework/examples/simple_federation
python run_client.py --id 0 --server_address localhost:50051 --num_clients 3
```

**Terminal 3:**
```bash
python run_client.py --id 1 --server_address localhost:50051 --num_clients 3
```

**Terminal 4:**
```bash
python run_client.py --id 2 --server_address localhost:50051 --num_clients 3
```

### Step 3: Watch Training Progress

Back in the **server terminal**, you'll see:

```
Round 1/10
  Client 0 - Loss: 0.4523, Acc: 87.23%
  Client 1 - Loss: 0.4912, Acc: 85.67%
  Client 2 - Loss: 0.4701, Acc: 86.45%
  Global Model - Acc: 89.12%

Round 2/10
  Client 0 - Loss: 0.2341, Acc: 92.45%
  ...
```

After 10 rounds, you should see final accuracy around **95%**.

## Understanding What Happened

### Server Side
1. **Initialized** global model with random weights
2. **Sent** model to all 3 clients
3. **Received** trained updates from each client
4. **Aggregated** updates using FedAvg strategy
5. **Evaluated** global model on test set
6. Repeated for 10 rounds

### Client Side
Each client:
1. **Received** global model from server
2. **Loaded** its local data partition (MNIST subset)
3. **Trained** model locally for K epochs
4. **Sent** updated weights back to server

## Next Steps

### Experiment with Parameters

Try different configurations:

```bash
# More rounds
python run_server.py --num_rounds 20

# Different aggregation strategy
python run_server.py --strategy FedAvg

# Non-IID data distribution
python run_client.py --id 0 --alpha 0.1  # More heterogeneous
python run_client.py --id 0 --alpha 10.0 # More homogeneous
```

### Try Other Examples

**LLM Training:**
```bash
cd examples/llm_federation
python run_server.py --dataset cb --num_rounds 5
python run_client.py --id 0 --dataset cb --server_address localhost:50051
```

**ECG Classification:**
```bash
cd examples/ecg_federation
python run_server.py --data_path ecg_data/ecg.csv
python run_client.py --id 0 --data_path ecg_data/ecg.csv
```

## Build Your Own Federated Application

### 1. Define Your Client

```python
import fedlearn as fl
import torch

class MyCustomClient(fl.Client):
    def __init__(self, client_id, data_loader, model):
        self.client_id = client_id
        self.data_loader = data_loader
        self.model = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    
    def get_parameters(self):
        """Return current model parameters"""
        return self.model.state_dict()
    
    def fit(self, parameters, config):
        """Train model locally"""
        # Load server parameters
        self.model.load_state_dict(parameters)
        self.model.train()
        
        # Training loop
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        for epoch in range(5):  # Local epochs
            for X_batch, y_batch in self.data_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
        
        # Return updated parameters and dataset size
        return self.model.state_dict(), len(self.data_loader.dataset)
```

### 2. Create Server Script

```python
import fedlearn as fl

# Initialize your model
model = YourModel()
initial_parameters = model.state_dict()

# Define evaluation function (optional)
def evaluate_fn(round_num, parameters):
    model.load_state_dict(parameters)
    # Evaluate on test set
    accuracy = evaluate_model(model, test_loader)
    return 0.0, {"accuracy": accuracy}

# Configure strategy
strategy = fl.FedAvg(
    initial_parameters=initial_parameters,
    evaluate_fn=evaluate_fn,
    min_fit_clients=2,        # Minimum clients per round
    clients_per_round=5,      # Clients to sample each round
)

# Start server
fl.server.start_server(
    server_address="0.0.0.0:50051",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
)
```

### 3. Create Client Script

```python
import fedlearn as fl

# Load your data
train_loader = load_your_data()
model = YourModel()

# Create client
client = MyCustomClient(
    client_id=0,
    data_loader=train_loader,
    model=model
)

# Connect to server
fl.client.start_client(
    server_address="localhost:50051",
    client=client,
    client_id="client_0"
)
```

### 4. Run Your Application

```bash
# Terminal 1: Start server
python your_server.py

# Terminal 2+: Start clients
python your_client.py --id 0
python your_client.py --id 1
```

## Common Patterns

### Pattern 1: Data Partitioning

```python
from fedlearn.data import partition_data

# Non-IID split using Dirichlet distribution
client_data = partition_data(
    X=X_train,
    y=y_train,
    num_clients=5,
    alpha=0.5  # Lower = more heterogeneous
)
```

### Pattern 2: Custom Aggregation

```python
class MyStrategy(fl.Strategy):
    def aggregate_fit(self, results):
        # Custom aggregation logic
        weighted_params = weighted_average(results)
        return weighted_params
```

### Pattern 3: Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(X_batch)
    loss = criterion(outputs, y_batch)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## Tips for Success

### Performance Optimization
- Use **GPU** for faster training
- Enable **mixed precision** training
- Tune **batch size** for your hardware
- Use **learning rate scheduling**

### Debugging
- Start with **1-2 clients** first
- Use **verbose logging** during development
- Test server and client **separately**
- Check **port availability** (default: 50051)

### Production Deployment
- Use **Docker** for consistent environments
- Implement **error handling** in clients
- Add **checkpointing** for long training runs
- Monitor **resource usage** (CPU, GPU, memory)

## Troubleshooting

**Issue: "Address already in use"**
```bash
# Use different port
python run_server.py --port 50052
python run_client.py --server_address localhost:50052
```

**Issue: Clients not connecting**
- Check firewall settings
- Verify server IP address
- Ensure server started before clients

**Issue: Out of memory**
- Reduce batch size
- Enable gradient checkpointing
- Use smaller model

## Next Steps

Now that you've run your first experiment:

1. **Explore Examples** - Try [LLM Federation](examples/llm-federation.md) or [ECG Classification](examples/ecg-federation.md)
2. **Read API Docs** - Understand [Server API](api-reference/server.md) and [Client API](api-reference/client.md)
3. **Customize** - Create [Custom Strategies](advanced/custom-strategies.md)
4. **Deploy** - Learn about production deployment with Docker

## Resources

- **Examples Directory**: `framework/examples/`
- **API Reference**: [docs/api-reference/](api-reference/)
- **GitHub Issues**: Report bugs or ask questions

---

**Ready to dive deeper?** Check out the [Examples](examples/) or [API Reference](api-reference/) for more advanced usage.