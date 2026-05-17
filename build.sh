#!/usr/bin/env bash
# Build script for Render — installs CPU-only PyTorch to save memory
set -e

# Install CPU-only PyTorch first (saves ~1.5GB vs full torch)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install -r requirements.txt
