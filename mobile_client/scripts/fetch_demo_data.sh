#!/usr/bin/env bash
# fetch_demo_data.sh — fetch the MNIST demo dataset at build time (fixes A6 §L5: do NOT commit
# dataset blobs to the repo). 15-LLD §13 task 18.
#
# VERIFY-BEFORE-USE: the mirror URL is pinned to a working public host; confirm availability.
set -euo pipefail

DATA_DIR="${DATA_DIR:-${PWD}/data/MNIST/raw}"
BASE_URL="${MNIST_BASE_URL:-https://ossci-datasets.s3.amazonaws.com/mnist}"

FILES=(
  train-images-idx3-ubyte.gz
  train-labels-idx1-ubyte.gz
  t10k-images-idx3-ubyte.gz
  t10k-labels-idx1-ubyte.gz
)

mkdir -p "${DATA_DIR}"
for f in "${FILES[@]}"; do
  if [[ -f "${DATA_DIR}/${f%.gz}" ]]; then
    echo "have ${f%.gz} (skip)"
    continue
  fi
  echo "fetching ${f}"
  curl -fSL "${BASE_URL}/${f}" -o "${DATA_DIR}/${f}"
  gunzip -kf "${DATA_DIR}/${f}"
done

echo "MNIST demo data ready in ${DATA_DIR} (raw blobs are git-ignored, not committed)"
