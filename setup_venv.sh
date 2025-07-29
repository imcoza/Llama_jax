#!/bin/bash

# Setup script for JAX virtual environment

echo "Setting up JAX virtual environment..."

# Create virtual environment
python3 -m venv llam_venv

# Activate virtual environment
source llam_venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

echo "Virtual environment setup complete!"
echo "To activate the environment, run: source llam_venv/bin/activate"
echo "To deactivate, run: deactivate" 