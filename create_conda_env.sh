#!/bin/bash

# Define variables
CONDA_ENV_NAME="raysam"
PYTHON_VERSION="3.11.9"
CUDA_VERSION="12.1"

echo "Creating Conda environment: $CONDA_ENV_NAME with Python $PYTHON_VERSION and required packages..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda not found! Please install Miniconda or Anaconda first."
    exit 1
fi

# Update Conda
conda update -n base -c defaults conda -y

# Create the Conda environment
conda create -n $CONDA_ENV_NAME python=$PYTHON_VERSION -y

# Activate the environment
source activate $CONDA_ENV_NAME

# Install packages
conda install -c anaconda pip -y
conda install -c pytorch pytorch torchvision torchaudio pytorch-cuda=$CUDA_VERSION -c nvidia -y
conda install -c conda-forge matplotlib natsort python-xxhash segment-anything python-elf kornia zarr pooch pandas numpy pillow scikit-learn scikit-image pyyaml -y

# Clean Conda cache
conda clean --all -y

# Install additional Python packages using pip
pip install --upgrade pip
pip install \
    jupyter jupyterlab \
    h5py torchsummary timm tensorboard \
    einops torch-tb-profiler pyclean \
    "ray[data,train,tune,serve]==2.24.0"

echo "Conda environment $CONDA_ENV_NAME created successfully!"
echo "To activate the environment, use: conda activate $CONDA_ENV_NAME"
