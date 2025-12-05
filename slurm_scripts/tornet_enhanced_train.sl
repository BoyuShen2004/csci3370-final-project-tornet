#!/bin/bash
#SBATCH --job-name=tornet-enhanced-train
#SBATCH --output=/projects/weilab/shenb/csci3370/code_submission/slurm_scripts/logs/%x_%j.out
#SBATCH --error=/projects/weilab/shenb/csci3370/code_submission/slurm_scripts/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=32GB
#SBATCH --time=48:00:00
#SBATCH --partition=medium
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=shenb@bc.edu

# Enhanced TorNet Training Script - Building on Baseline Success

echo ">>> Starting Enhanced TorNet Training (Building on Baseline Success)..."
echo ">>> Job ID: $SLURM_JOB_ID"
echo ">>> Start time: $(date)"

# Set up environment
source /projects/weilab/shenb/miniconda3/etc/profile.d/conda.sh
conda activate tornet2

# Set working directory (enhanced copy)
cd /projects/weilab/shenb/csci3370/code_submission/tornet_enhanced
export PYTHONPATH=/projects/weilab/shenb/csci3370/code_submission/tornet_enhanced:$PYTHONPATH

# Set environment variables
export TORNET_ROOT=/projects/weilab/shenb/csci3370/data
export KERAS_BACKEND=torch
# Force outputs under code_submission/outputs
export EXP_DIR=/projects/weilab/shenb/csci3370/code_submission/outputs/tornado_enhanced_$(date +%Y%m%d%H%M%S)-$SLURM_JOB_ID-None

# Paths
CONFIG_YAML=/projects/weilab/shenb/csci3370/code_submission/tornet_enhanced/configs/params_enhanced_tornet.yaml

# Create experiment directory and convert YAML -> JSON for the trainer
mkdir -p "$EXP_DIR"
python - <<PY
import yaml, json, sys, os
src = "$CONFIG_YAML"
dst = os.path.join("$EXP_DIR", "params_enhanced_tornet.json")
with open(src, "r") as f:
    data = yaml.safe_load(f)
with open(dst, "w") as f:
    json.dump(data, f)
print(f"Wrote {dst}")
PY

echo ">>> TORNET_ROOT: $TORNET_ROOT"
echo ">>> KERAS_BACKEND: $KERAS_BACKEND"
echo ">>> EXP_DIR: $EXP_DIR"
echo ">>> CONFIG: $CONFIG_YAML"

echo ">>> Starting enhanced TorNet training ..."
python scripts/tornado_detection/train_enhanced_tornet_keras.py \
       $EXP_DIR/params_enhanced_tornet.json

echo ">>> Training completed"
echo ">>> End time: $(date)"


