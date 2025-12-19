# TorNet Tornado Detection

This repository contains the code and experiments for improving tornado detection on the TorNet benchmark using an enhanced residual CNN and evaluating transfer learning with DINOv3 Vision Transformers.

The enhanced CNN improves PR AUC, ROC AUC, and CSI over the MIT Lincoln Laboratory baseline, while DINOv3 underperforms due to domain mismatch between natural images and polarimetric radar data.

See the final report for full methodology, experiments, and results.

--------------------------------------------------
Runnable scripts / commands (HPC / SLURM)
--------------------------------------------------

These commands are intended for an HPC cluster setup similar to Boston College’s Andromeda (SLURM-managed GPU nodes). Edit SBATCH headers, paths, and environment/module loads inside each `.sl` script to match your cluster.

From the repo root:

cd slurm_scripts

Train + evaluate Enhanced CNN:

```
sbatch tornet_enhanced_train.sl
sbatch tornet_enhanced_eval.sl
```

DINOv3 pipeline (supervised adaptation + fine-tune + inference):

```
sbatch dinov3_supervised_pretrain.sl
sbatch dinov3_finetune.sl
sbatch dinov3_inference.sl
```

Logs are written under:

ls -lah logs/

--------------------------------------------------
Contribution
--------------------------------------------------

Joint Work (core pipeline + results reproduction)
- End-to-end enhanced CNN training + evaluation workflow (integration + final config): `tornet_enhanced/`, `tornet_enhanced/configs/`, `tornet_enhanced/scripts/tornado_detection/`, `slurm_scripts/tornet_enhanced_train.sl`, `slurm_scripts/tornet_enhanced_eval.sl`
- Metrics, reporting, and baseline comparison used in the final writeup: `tornet_enhanced/tornet/metrics/`, `tornet_enhanced/tornet/models/keras/imbalanced_metrics.py`, `outputs/`, `baseline_results.png`, `class_composition.png`
- Reproducible cluster runs + logging outputs from SLURM experiments: `slurm_scripts/`, `slurm_scripts/logs/`, `outputs/`
- Repo-level documentation and final deliverables packaging: `ReadMe.md`, `tornet_enhanced/ReadMe.md`, `CSCI3370_Final_Report.pdf`

Boyu (Ethan) Shen (data + modeling + DINOv3 track)
- DINOv3 transfer-learning implementation (data loading, model head/projection, focal/imbalanced losses, training stages, inference): `dinov3_vit/classification_model.py`, `dinov3_vit/data_loader_tornet.py`, `dinov3_vit/imbalanced_losses.py`, `dinov3_vit/supervised_pretrain/`, `dinov3_vit/finetune/`, `dinov3_vit/inference/`, `slurm_scripts/dinov3_supervised_pretrain.sl`, `slurm_scripts/dinov3_finetune.sl`, `slurm_scripts/dinov3_inference.sl`
- Environment setup and dependency wiring for the repo: `create_tornet2_env.sh`, `requirements.txt`, `requirements/`
- Experiment artifacts and run bookkeeping for model training: `outputs/` (both `tornado_enhanced_*` and `dino_supervised_*` runs), plus associated configs/logs in those folders

Brendan Keller (evaluation + baseline reproduction tooling)
- Evaluation utilities for ROC AUC / PR AUC / CSI / HSS and confusion-matrix based reporting used across experiments: components under `tornet_enhanced/tornet/metrics/` and evaluation-side logic invoked by `tornet_enhanced/scripts/tornado_detection/test_enhanced_tornet_keras.py` and `slurm_scripts/tornet_enhanced_eval.sl`
- Baseline metric reproduction and comparison outputs used to generate/verify reported numbers: evaluation scripts + output verification under `tornet_enhanced/` and `outputs/`, including the final comparison figures (`baseline_results.png`) as referenced in the report