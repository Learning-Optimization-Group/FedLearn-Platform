# =============================================================================
# FedLearn Edge Client — Docker Image
# =============================================================================
# Default: standard x86 PyTorch image.
# Override for Jetson:
#   docker build --build-arg BASE_IMAGE=nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3 .
# =============================================================================
ARG BASE_IMAGE=pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-venv python3-dev python3-pip build-essential git wget curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# 1. Copy the core framework (Single Source of Truth)
COPY framework/ /app/framework/

# 2. Install the framework as a standard Python library
RUN pip3 install --no-cache-dir /app/framework/

# 3. Copy client-specific execution scripts and requirements
COPY client-docker/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
COPY client-docker/scripts/ /app/scripts/
COPY client-docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENV PYTHONPATH=/app:${PYTHONPATH}
WORKDIR /app/scripts
ENTRYPOINT ["/app/entrypoint.sh"]
CMD []
