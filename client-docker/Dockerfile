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

# Install explicit protobuf
RUN pip3 install protobuf==6.32.1

# Install all other requirements
# Note: torch, torchvision, torchaudio are skipped because the NVIDIA base image already has optimized versions
RUN pip3 install -r requirements.txt

# Copy the fedlearn package
COPY fedlearn/ /app/fedlearn/

# Copy scripts directory (includes models.py, client.py, etc.)
COPY scripts/ /app/scripts/

# Set Python path to include the app directory
ENV PYTHONPATH=/app:${PYTHONPATH}

# Set working directory to scripts
WORKDIR /app/scripts

# Create entry point script (using python3)
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
if [ -z "$PROJECT_ID" ] || [ -z "$SERVER_ADDRESS" ] || [ -z "$PARTITION_ID" ]; then\n\
    echo "Error: Missing required environment variables"\n\
    exit 1\n\
fi\n\
\n\
exec python3 client.py \\\n\
    --project-id "$PROJECT_ID" \\\n\
    --server-address "$SERVER_ADDRESS" \\\n\
    --partition-id "$PARTITION_ID" \\\n\
    "$@"\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD []
