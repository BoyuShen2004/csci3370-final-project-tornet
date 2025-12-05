#!/bin/bash
# Create/refresh the "tornet2" conda environment and install all code_submission deps.
# Usage: bash create_tornet2_env.sh

set -euo pipefail

CONDA_ROOT="/projects/weilab/shenb/miniconda3"
ENV_NAME="tornet2"
REPO_ROOT="/projects/weilab/shenb/csci3370/code_submission"

if [ ! -d "$CONDA_ROOT" ]; then
  echo "Conda root not found at $CONDA_ROOT" >&2
  exit 1
fi

source "$CONDA_ROOT/etc/profile.d/conda.sh"

echo ">>> Creating/updating env: $ENV_NAME"
conda create -y -n "$ENV_NAME" python=3.11
conda activate "$ENV_NAME"

echo ">>> Installing Python deps from $REPO_ROOT/requirements.txt"
pip install --upgrade pip
pip install -r "$REPO_ROOT/requirements.txt"

echo ">>> Installing NetCDF/HDF native libs via conda-forge"
conda install -y -c conda-forge netcdf4 h5netcdf libnetcdf hdf5

echo ">>> Done. Activate with: conda activate $ENV_NAME"

