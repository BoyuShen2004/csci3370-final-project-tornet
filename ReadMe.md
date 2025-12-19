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

sbatch tornet_enhanced_train.sl
sbatch tornet_enhanced_eval.sl

DINOv3 pipeline (supervised adaptation + fine-tune + inference):

sbatch dinov3_supervised_pretrain.sl
sbatch dinov3_finetune.sl
sbatch dinov3_inference.sl

Logs are written under:

ls -lah logs/

--------------------------------------------------
Contribution
--------------------------------------------------

Boyu (Ethan) Shen
- Implemented the TorNet data ingestion pipeline, including radar preprocessing, normalization, channel assembly, and visualization of representative samples.
- Designed and wrote the Method section, including the enhanced CNN architecture, residual-block formulation, loss functions, sampling strategy, and the full DINOv3 transfer-learning pipeline.
- Executed and documented the full Experiments section: HPC job scheduling, training runs, compute accounting, metric reporting, and ablation commentary.

Brendan Keller
- Implemented evaluation utilities for ROC AUC, PR AUC, CSI, HSS, and confusion matrices, and reproduced the MIT LL baseline metrics for comparison.
- Wrote the Introduction, Related Work, and Conclusion sections, framing the technical context, prior literature, and the discussion and implications of the experimental findings.

Joint Work
- Co-designed and implemented the enhanced CNN training procedure in PyTorch, including optimizer configuration, cosine warmup scheduling, and focal-loss tuning.
- Conducted hyperparameter tuning (learning rate, batch size, augmentation strength, focal-loss γ, oversampling ratio) and jointly selected the final model configuration.
- Performed literature review on radar-based tornado detection, class-imbalance strategies, CNN inductive biases, and vision-transformer learning.
- Managed GitHub documentation, milestone updates, and TA/professor check-ins.
- Delivered the final presentation.