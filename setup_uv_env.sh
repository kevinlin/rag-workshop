#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Create a virtual environment using uv
echo "Creating virtual environment..."
uv venv .venv

# Activate the virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

# Install development dependencies
echo "Installing development dependencies from requirements-dev.txt..."
uv pip install -r requirements-dev.txt

echo "Setup complete. Virtual environment '.venv' is ready and dependencies are installed."
