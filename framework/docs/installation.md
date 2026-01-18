# Installation Guide

This guide provides detailed instructions for installing FedLearn and its dependencies.

## System Requirements

### Minimum Requirements
- **Python**: 3.10 or higher
- **RAM**: 8GB (16GB+ recommended for LLM training)
- **Storage**: 5GB free space
- **OS**: Linux, macOS, or Windows

### Recommended for Production
- **GPU**: CUDA-capable GPU (NVIDIA)
- **CUDA**: 12.1 or higher
- **RAM**: 16GB+
- **Storage**: 20GB+ for model checkpoints

## Installation Methods

### Method 1: Quick Install (Recommended)

```bash
# Clone the repository
git clone https://github.com/Learning-Optimization-Group/FedLearn-Platform.git
cd FedLearn-Platform/framework

# Install PyTorch with CUDA support (if you have GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install FedLearn and dependencies
pip install -e .
```

### Method 2: CPU-Only Installation

```bash
# Clone repository
git clone https://github.com/Learning-Optimization-Group/FedLearn-Platform.git
cd FedLearn-Platform/framework

# Install PyTorch (CPU version)
pip install torch torchvision torchaudio

# Install FedLearn
pip install -e .
```

### Method 3: Development Installation

For contributors who want to modify the framework:

```bash
# Clone repository
git clone https://github.com/Learning-Optimization-Group/FedLearn-Platform.git
cd FedLearn-Platform/framework

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Verify installation
pytest tests/
```

## Verify Installation

After installation, verify everything works:

```python
import fedlearn as fl
import torch

print(f"FedLearn imported successfully")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

Expected output:
```
FedLearn imported successfully
PyTorch version: 2.7.0+cu121
CUDA available: True
CUDA version: 12.1
GPU: NVIDIA GeForce RTX 3090
```

## Component-by-Component Installation

### Core Dependencies

#### 1. PyTorch (Required)
```bash
# GPU version (CUDA 12.1)
pip install torch==2.7.0 torchvision==0.21.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu121

# CPU version
pip install torch==2.7.0 torchvision==0.21.0 torchaudio==2.7.0
```

#### 2. Flower Framework (Required)
```bash
pip install flwr==1.20.0 flwr-datasets==0.5.0
```

#### 3. Transformers (For LLM training)
```bash
pip install transformers>=4.57.0 datasets>=3.1.0 tokenizers>=0.21.4
```

#### 4. gRPC (Required for distributed training)
```bash
pip install grpcio>=1.75.1 grpcio-tools>=1.75.1 protobuf>=4.21.6
```

## Troubleshooting

### Issue: CUDA not available after installation

**Solution:**
```bash
# Verify CUDA toolkit installation
nvcc --version

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Issue: gRPC installation fails

**Solution:**
```bash
# Update pip and setuptools
pip install --upgrade pip setuptools wheel

# Install gRPC with pre-built wheels
pip install --upgrade grpcio grpcio-tools
```

### Issue: Memory errors during LLM training

**Solution:**
- Reduce batch size in config file
- Enable gradient checkpointing
- Use mixed precision training (enabled by default)
- Consider using gradient accumulation

### Issue: Import errors

**Solution:**
```bash
# Ensure you're in the framework directory
cd FedLearn-Platform/framework

# Reinstall in editable mode
pip install -e .

# Verify package location
pip show fedlearn
```

### Issue: Port already in use (Address already in use)

**Solution:**
```bash
# Find process using port 50051
lsof -i :50051

# Kill the process (replace PID)
kill -9 <PID>

# Or use a different port
python run_server.py --port 50052
```

## Virtual Environment Setup (Recommended)

Using a virtual environment isolates dependencies:

### Using venv
```bash
# Create virtual environment
python -m venv fedlearn-env

# Activate
source fedlearn-env/bin/activate  # Linux/Mac
fedlearn-env\Scripts\activate     # Windows

# Install FedLearn
cd FedLearn-Platform/framework
pip install -e .
```

### Using conda
```bash
# Create conda environment
conda create -n fedlearn python=3.10

# Activate
conda activate fedlearn

# Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install FedLearn
cd FedLearn-Platform/framework
pip install -e .
```

## Docker Installation

For containerized deployment:

```bash
# Build Docker image (from repository root)
docker build -t fedlearn:latest -f docker/Dockerfile .

# Run container
docker run -it --gpus all fedlearn:latest bash
```

## Platform-Specific Notes

### Linux
- No special requirements
- GPU support works out of the box with CUDA drivers

### macOS
- Apple Silicon (M1/M2): Use MPS backend for GPU acceleration
- Intel Macs: CPU-only installation recommended

```python
# Check MPS availability (Apple Silicon)
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
```

### Windows
- Use PowerShell or Command Prompt
- Ensure Microsoft Visual C++ Redistributable is installed
- GPU support requires NVIDIA drivers and CUDA toolkit

## Next Steps

After successful installation:

1. **Quick Start**: Run the [Quick Start Guide](quickstart.md)
2. **Examples**: Try the [Simple Federation Example](examples/simple-federation.md)
3. **Documentation**: Explore the [API Reference](api-reference/)

## Additional Resources

- **PyTorch Installation**: https://pytorch.org/get-started/locally/
- **CUDA Toolkit**: https://developer.nvidia.com/cuda-downloads
- **gRPC Documentation**: https://grpc.io/docs/languages/python/quickstart/

## Support

If you encounter issues not covered here:
- Check [GitHub Issues](https://github.com/Learning-Optimization-Group/FedLearn-Platform/issues)
- Create a new issue with detailed error messages
- Include Python version, OS, and CUDA version (if applicable)