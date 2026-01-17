# Use NVIDIA CUDA runtime base image for GPU support with CPU fallback
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies including Python 3.10 (default for Ubuntu 22.04)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install PyTorch with CUDA 12.1 support from PyTorch index
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Install protobuf explicitly to ensure correct version
RUN pip install protobuf==6.32.1

# Install other Python dependencies
RUN pip install -r requirements.txt

# Copy the fedlearn package
COPY fedlearn/ /app/fedlearn/

# Copy scripts directory (includes models.py, client.py, etc.)
COPY scripts/ /app/scripts/

# Set Python path to include the app directory
ENV PYTHONPATH=/app:${PYTHONPATH}

# Set working directory to scripts
WORKDIR /app/scripts

# Create entry point script with support for --use-llm and --dataset flags
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Check if required arguments are provided\n\
if [ -z "$PROJECT_ID" ] || [ -z "$SERVER_ADDRESS" ] || [ -z "$PARTITION_ID" ]; then\n\
    echo "Error: Missing required environment variables or arguments"\n\
    echo "Usage: docker run --gpus all -e PROJECT_ID=<id> -e SERVER_ADDRESS=<address> -e PARTITION_ID=<id> <image> [--use-llm] [--dataset cb|sst2]"\n\
    echo "   OR: docker run --gpus all <image> --project-id <id> --server-address <address> --partition-id <id> [--use-llm] [--dataset cb|sst2]"\n\
    exit 1\n\
fi\n\
\n\
# Use environment variables\n\
PROJECT_ARG="${PROJECT_ID}"\n\
SERVER_ARG="${SERVER_ADDRESS}"\n\
PARTITION_ARG="${PARTITION_ID}"\n\
\n\
# Run the client\n\
exec python client.py \\\n\
    --project-id "$PROJECT_ARG" \\\n\
    --server-address "$SERVER_ARG" \\\n\
    --partition-id "$PARTITION_ARG" \\\n\
    "$@"\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command (can be overridden)
CMD []
