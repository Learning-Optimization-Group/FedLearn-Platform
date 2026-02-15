# FedLearn Client - Docker Package

Pre-packaged Docker image for federated learning clients. This containerized solution includes the FedLearn framework and client scripts, allowing users to participate in FL training without installing dependencies locally.

## Overview

This Docker package enables easy deployment of FL clients by:
- **Zero Installation**: No need to install Python, PyTorch, or FedLearn framework
- **Reproducible Environment**: Consistent runtime across all machines
- **Quick Deployment**: Build once, deploy anywhere
- **Data Flexibility**: Mount local datasets into containers
- **Scalability**: Run multiple clients with different configurations

## Use Cases

### 1. Research & Development
- Test FL algorithms with multiple simulated clients on one machine
- Rapid prototyping without environment setup

### 2. Production Deployment
- Deploy clients across different organizations/locations
- Ensure consistent FL framework version across all clients

### 3. Client Application Integration
- Provide clients with ready-to-use Docker image
- Clients only need: Docker, dataset, and server address
- No Python/ML expertise required from client side

## Directory Structure

```
client-docker/
├── fedlearn/                   # FedLearn framework (copied from ../framework/src/fedlearn)
│   ├── client/
│   ├── server/
│   ├── communication/
│   └── ...
├── scripts/
│   ├── client.py              # Generic FL client script
│   └── models.py              # Model definitions
├── .dockerignore              # Files to exclude from Docker build
├── Dockerfile                 # Docker image definition
├── main.py                    # Entry point for container
├── requirements.txt           # Python dependencies
└── run-client.sh              # Helper script to run container
```

## Quick Start

### Prerequisites

- Docker installed ([Get Docker](https://docs.docker.com/get-docker/))
- Access to FL server (address and port)
- Dataset in supported format

### 1. Build Docker Image

```bash
cd client-docker

# Build the image
docker build -t fedlearn-client:latest .
```

**Build time**: ~5-10 minutes (depending on internet speed)

### 2. Run Client

```bash
# Basic usage
docker run -it \
  -v /path/to/your/dataset:/data \
  fedlearn-client:latest \
  --server-address <server-ip>:50051 \
  --client-id 0

# With custom configuration
docker run -it \
  -v /path/to/dataset:/data \
  fedlearn-client:latest \
  --server-address 192.168.1.100:50051 \
  --client-id 0 \
  --dataset-path /data/train.csv \
  --model cnn \
  --epochs 5
```

### 3. Using Helper Script

```bash
# Make script executable
chmod +x run-client.sh

# Run client
./run-client.sh \
  --server 192.168.1.100:50051 \
  --id 0 \
  --data /path/to/dataset
```

## Configuration

### Environment Variables

Set via `-e` flag in `docker run`:

```bash
docker run -it \
  -e SERVER_ADDRESS=192.168.1.100:50051 \
  -e CLIENT_ID=0 \
  -e DATASET_PATH=/data/train.csv \
  -e MODEL_TYPE=cnn \
  fedlearn-client:latest
```

**Available Variables**:
| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `SERVER_ADDRESS` | FL server address (host:port) | - | Yes |
| `CLIENT_ID` | Unique client identifier | 0 | No |
| `DATASET_PATH` | Path to dataset inside container | /data | No |
| `MODEL_TYPE` | Model architecture (cnn, transformer) | cnn | No |
| `LOCAL_EPOCHS` | Training epochs per round | 5 | No |
| `BATCH_SIZE` | Training batch size | 32 | No |

### Command Line Arguments

Passed directly to the container:

```bash
docker run fedlearn-client:latest [OPTIONS]

Options:
  --server-address TEXT    Server address (required)
  --client-id INTEGER      Client ID (default: 0)
  --dataset-path TEXT      Path to dataset (default: /data)
  --model TEXT             Model type (default: cnn)
  --epochs INTEGER         Local epochs (default: 5)
  --batch-size INTEGER     Batch size (default: 32)
```

## Dataset Mounting

### Volume Mounting

```bash
# Mount local directory to /data in container
docker run -v /local/path:/data fedlearn-client:latest ...
```

**Examples**:

**CSV Dataset**:
```bash
docker run -v ~/datasets/mnist:/data \
  fedlearn-client:latest \
  --server-address server:50051 \
  --dataset-path /data/mnist_train.csv
```

**Image Dataset**:
```bash
docker run -v ~/datasets/images:/data \
  fedlearn-client:latest \
  --server-address server:50051 \
  --dataset-path /data
```

### Supported Dataset Formats

- **CSV**: Structured data (e.g., tabular, time-series)
- **Images**: PNG, JPEG (organized in folders by class)
- **Text**: TXT files for NLP tasks
- **Custom**: Implement custom data loader in `scripts/client.py`

## Dockerfile Explained

The Dockerfile creates a containerized environment with all dependencies:

```dockerfile
# Base image with Python
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy framework and scripts
COPY fedlearn/ /app/fedlearn/
COPY scripts/ /app/scripts/
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entry point
COPY main.py /app/

# Create data directory
RUN mkdir -p /data

# Set entry point
ENTRYPOINT ["python", "main.py"]
```

**Key Components**:
1. **Base Image**: Python 3.10 (lightweight)
2. **Dependencies**: Installs from requirements.txt
3. **Framework**: Copies FedLearn framework
4. **Scripts**: Copies client script and models
5. **Data Directory**: `/data` for mounting datasets
6. **Entry Point**: `main.py` handles arguments

## Main Entry Point (main.py)

The `main.py` script:
1. Parses command-line arguments or environment variables
2. Loads dataset from mounted volume
3. Initializes client with FedLearn framework
4. Connects to server and starts training

**Workflow**:
```python
# Simplified workflow
import fedlearn as fl
from scripts.client import CustomClient

# Parse arguments
args = parse_args()

# Load data
train_data = load_dataset(args.dataset_path)

# Create client
client = CustomClient(
    client_id=args.client_id,
    data=train_data,
    model_type=args.model
)

# Connect to server
fl.client.start_client(
    server_address=args.server_address,
    client=client
)
```

## Multi-Client Deployment

### Local Testing (Multiple Clients on One Machine)

Use Docker Compose:

```yaml
# docker-compose.yml
version: '3.8'

services:
  client-0:
    image: fedlearn-client:latest
    volumes:
      - ./data/client0:/data
    environment:
      SERVER_ADDRESS: "server:50051"
      CLIENT_ID: 0
    networks:
      - fl-network

  client-1:
    image: fedlearn-client:latest
    volumes:
      - ./data/client1:/data
    environment:
      SERVER_ADDRESS: "server:50051"
      CLIENT_ID: 1
    networks:
      - fl-network

  client-2:
    image: fedlearn-client:latest
    volumes:
      - ./data/client2:/data
    environment:
      SERVER_ADDRESS: "server:50051"
      CLIENT_ID: 2
    networks:
      - fl-network

networks:
  fl-network:
    driver: bridge
```

**Run**:
```bash
docker-compose up
```

### Distributed Deployment (Different Machines)

**Machine 1** (Client 0):
```bash
docker run -v /data/client0:/data \
  fedlearn-client:latest \
  --server-address central-server.com:50051 \
  --client-id 0
```

**Machine 2** (Client 1):
```bash
docker run -v /data/client1:/data \
  fedlearn-client:latest \
  --server-address central-server.com:50051 \
  --client-id 1
```

**Machine 3** (Client 2):
```bash
docker run -v /data/client2:/data \
  fedlearn-client:latest \
  --server-address central-server.com:50051 \
  --client-id 2
```

## Building for Different Architectures

### ARM64 (Apple Silicon, Raspberry Pi)

```bash
docker build --platform linux/arm64 -t fedlearn-client:arm64 .
```

### AMD64 (x86_64)

```bash
docker build --platform linux/amd64 -t fedlearn-client:amd64 .
```

### Multi-Platform Build

```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t fedlearn-client:multi \
  --push \
  .
```

## Distribution

### Option 1: Docker Hub

```bash
# Tag image
docker tag fedlearn-client:latest username/fedlearn-client:latest

# Push to Docker Hub
docker push username/fedlearn-client:latest

# Users pull and run
docker pull username/fedlearn-client:latest
docker run -v /data:/data username/fedlearn-client:latest ...
```

### Option 2: Private Registry

```bash
# Tag for private registry
docker tag fedlearn-client:latest registry.company.com/fedlearn-client:latest

# Push
docker push registry.company.com/fedlearn-client:latest
```

### Option 3: Export/Import (Offline)

```bash
# Save image to file
docker save fedlearn-client:latest -o fedlearn-client.tar

# Transfer file to another machine
scp fedlearn-client.tar user@remote:/path/

# Load on another machine
docker load -i fedlearn-client.tar
```

## Client Application Integration

### For End Users (Non-Technical)

Provide users with:

1. **Docker Image** (via Docker Hub or file)
2. **Simple Run Script**:

```bash
#!/bin/bash
# run-my-client.sh

# User only needs to configure these
SERVER_ADDRESS="your-server.com:50051"
CLIENT_ID=0
DATA_PATH="/path/to/my/data"

docker run -it \
  -v $DATA_PATH:/data \
  fedlearn-client:latest \
  --server-address $SERVER_ADDRESS \
  --client-id $CLIENT_ID
```

3. **Instructions**:
   - Install Docker
   - Download dataset
   - Edit run script with server address
   - Run: `./run-my-client.sh`

### For Developers

Provide:
- Dockerfile (for customization)
- Client script (`scripts/client.py`) - modify for custom models
- Build instructions

## GPU Support

### Enable GPU in Docker

```bash
# Install nvidia-docker (one-time setup)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Run with GPU
docker run --gpus all \
  -v /data:/data \
  fedlearn-client:latest \
  --server-address server:50051
```

### Dockerfile for GPU

Add CUDA support:
```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y python3.10 python3-pip

# ... rest of Dockerfile
```

## Troubleshooting

### Issue: Cannot connect to server

**Check**:
1. Server is running and accessible
2. Firewall allows connection to server port
3. Correct server address format: `host:port`

```bash
# Test connectivity
docker run --rm alpine ping server-host
docker run --rm alpine nc -zv server-host 50051
```

### Issue: Dataset not found

**Check**:
1. Volume mount path is correct
2. Dataset exists in mounted directory
3. File permissions allow reading

```bash
# Verify mount
docker run -v /local/path:/data fedlearn-client:latest ls -la /data
```

### Issue: Out of memory

**Solution**: Reduce batch size or allocate more memory to Docker

```bash
# Reduce batch size
docker run ... fedlearn-client:latest --batch-size 16

# Increase Docker memory (Docker Desktop settings)
```

### Issue: Image build fails

**Solution**:
```bash
# Clear Docker cache
docker builder prune

# Rebuild
docker build --no-cache -t fedlearn-client:latest .
```

## Advanced Usage

### Custom Client Script

Modify `scripts/client.py` to implement custom training logic:

```python
import fedlearn as fl

class CustomClient(fl.Client):
    def __init__(self, client_id, data, model):
        self.client_id = client_id
        self.data = data
        self.model = model
    
    def fit(self, parameters, config):
        # Custom training logic
        # ...
        return updated_params, num_samples
```

Rebuild image:
```bash
docker build -t fedlearn-client:custom .
```

### Environment-Specific Configuration

Use `.env` file:

```bash
# .env
SERVER_ADDRESS=server.example.com:50051
CLIENT_ID=0
MODEL_TYPE=transformer
BATCH_SIZE=64
```

Run with:
```bash
docker run --env-file .env -v /data:/data fedlearn-client:latest
```

## Security Considerations

### 1. Network Security
- Use VPN or private network for server communication
- Consider TLS/SSL for gRPC connections

### 2. Data Privacy
- Data never leaves the container except as model updates
- Mount datasets as read-only: `-v /data:/data:ro`

### 3. Resource Limits
```bash
# Limit CPU and memory
docker run \
  --cpus="2.0" \
  --memory="4g" \
  -v /data:/data \
  fedlearn-client:latest ...
```

## Performance Optimization

### 1. Reduce Image Size

Use multi-stage builds:
```dockerfile
# Build stage
FROM python:3.10 AS builder
RUN pip install --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.10-slim
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
```

### 2. Layer Caching

Order Dockerfile commands from least to most frequently changing:
```dockerfile
COPY requirements.txt .
RUN pip install -r requirements.txt  # Cached if requirements unchanged
COPY . .  # This layer rebuilds when code changes
```

### 3. Parallel Builds

```bash
# Use BuildKit
DOCKER_BUILDKIT=1 docker build -t fedlearn-client:latest .
```

## Maintenance

### Update FedLearn Framework

```bash
# Copy new framework version
cp -r ../framework/src/fedlearn ./fedlearn/

# Rebuild image
docker build -t fedlearn-client:v2.0 .
```

### Version Tagging

```bash
# Tag releases
docker tag fedlearn-client:latest fedlearn-client:v1.0.0
docker tag fedlearn-client:latest fedlearn-client:v1.0
docker tag fedlearn-client:latest fedlearn-client:v1
```

## Future Enhancements

This Docker package is designed to support future client applications where:

1. **Web Interface**: Users upload datasets via web UI
2. **Automated Training**: Backend spawns Docker containers for each client
3. **Model Selection**: Users select pre-configured models from dropdown
4. **Zero Configuration**: Everything managed through UI

**Architecture**:
```
User Web UI → Backend API → Docker Container (this package) → FL Server
```

## Resources

- **Build Time**: ~5-10 minutes
- **Image Size**: ~2-3 GB (with PyTorch)
- **Runtime Memory**: 2-8 GB (depends on model/data)
- **CPU**: 2+ cores recommended

## Support

For issues or questions:
- Framework issues: See [framework/README.md](../framework/README.md)
- Client script customization: See [DEVELOPMENT.md](DEVELOPMENT.md)
- Docker-specific issues: Check Docker logs with `docker logs <container-id>`

---

**Quick Command Reference**:
```bash
# Build
docker build -t fedlearn-client:latest .

# Run
docker run -v /data:/data fedlearn-client:latest --server-address host:port --client-id 0

# Multi-client
docker-compose up

# Export
docker save fedlearn-client:latest -o fedlearn-client.tar
```