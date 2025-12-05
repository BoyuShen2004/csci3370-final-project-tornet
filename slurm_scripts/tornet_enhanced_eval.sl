#!/bin/bash
#SBATCH --job-name=tornet-enhanced-eval
#SBATCH --output=/projects/weilab/shenb/csci3370/code_submission/slurm_scripts/logs/%x_%j.out
#SBATCH --error=/projects/weilab/shenb/csci3370/code_submission/slurm_scripts/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --partition=short
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=shenb@bc.edu

echo ">>> Starting enhanced TorNet evaluation..."
echo ">>> Job ID: $SLURM_JOB_ID"
echo ">>> Start time: $(date)"

# Set up environment
source /projects/weilab/shenb/miniconda3/etc/profile.d/conda.sh
conda activate tornet2

# Set working directory
cd /projects/weilab/shenb/csci3370/code_submission/tornet_enhanced
export PYTHONPATH=/projects/weilab/shenb/csci3370/code_submission/tornet_enhanced:$PYTHONPATH

# Set environment variables
export TORNET_ROOT=/projects/weilab/shenb/csci3370/data
export KERAS_BACKEND=torch

# Config (YAML) with model path (override with first arg)
CONFIG_YAML="${1:-/projects/weilab/shenb/csci3370/code_submission/tornet_enhanced/configs/eval.yaml}"

# Read model path from YAML
MODEL_PATH=$(python - <<PY
import yaml,sys
cfg = yaml.safe_load(open("$CONFIG_YAML"))
print(cfg.get("model_path"))
PY
)

echo ">>> TORNET_ROOT: $TORNET_ROOT"
echo ">>> CONFIG:     $CONFIG_YAML"
echo ">>> MODEL_PATH: $MODEL_PATH"
echo ">>> GPU info:"
nvidia-smi || true
echo "=========================================="

python scripts/tornado_detection/test_enhanced_tornet_keras.py "$MODEL_PATH"

echo ">>> Evaluation completed at $(date)"
echo ">>> Results saved alongside: $MODEL_PATH"


