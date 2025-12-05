#!/bin/bash
#SBATCH --job-name=dinov3-supervised-pretrain
#SBATCH --output=/projects/weilab/shenb/csci3370/code_submission/slurm_scripts/logs/%x_%j.out
#SBATCH --error=/projects/weilab/shenb/csci3370/code_submission/slurm_scripts/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=120:00:00
#SBATCH --partition=long
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=shenb@bc.edu

echo ">>> Starting DINOv3 supervised pretraining..."
echo ">>> Job ID: $SLURM_JOB_ID"
echo ">>> Start time: $(date)"

# Set up environment
source /projects/weilab/shenb/miniconda3/etc/profile.d/conda.sh
conda activate tornet2

# Set working directory
cd /projects/weilab/shenb/csci3370/code_submission/dinov3_vit

# Set environment variables
export TORNET_ROOT=/projects/weilab/shenb/csci3370/data
export KERAS_BACKEND=torch

# Optional override for encoder checkpoint via first arg
export DINOV3_PRETRAINED="${1:-/projects/weilab/shenb/csci3370/code_submission/dinov3_vit/pretrained/dinov3_vits16_pretrain_lvd1689m-08c60483.pth}"

echo ">>> TORNET_ROOT: $TORNET_ROOT"
echo ">>> ENCODER:     $DINOV3_PRETRAINED"
echo ">>> GPU info:"
nvidia-smi || true
echo "=========================================="

python supervised_pretrain/train_supervised_classification.py --config supervised_pretrain/config.yaml

echo ">>> Supervised pretraining completed at $(date)"


