# Use official NVIDIA Jetson PyTorch base image (CUDA 11.4 support)
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies (Ubuntu 20.04 uses python3 natively)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-venv \
    python3-dev \
    python3-pip \
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip using python3
RUN python3 -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install all requirements (--no-cache-dir to avoid stale pip cache)
# Note: torch/torchvision/torchaudio are pre-installed in the NVIDIA base image
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the fedlearn package
COPY fedlearn/ /app/fedlearn/

# Regenerate protobuf files to match installed protobuf runtime version
# (source .proto was compiled with protobuf 6.x, but container has 5.x)
RUN python3 -m grpc_tools.protoc \
    -I /app/fedlearn/communication/protos \
    --python_out=/app/fedlearn/communication/generated \
    --grpc_python_out=/app/fedlearn/communication/generated \
    /app/fedlearn/communication/protos/fedlearn.proto

# Copy scripts directory (includes models.py, client.py, etc.)
COPY scripts/ /app/scripts/

# Copy the entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Set Python path to include the app directory
ENV PYTHONPATH=/app:${PYTHONPATH}

# Set working directory to scripts
WORKDIR /app/scripts

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD []
