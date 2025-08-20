# RhoFold: Open-Source RNA 3D Structure Pipeline

## Overview
Open-source, resource-efficient RNA tertiary-structure pipeline that combines a RoPE-enhanced Transformer encoder with an EGNN refinement stage. Built to run accurate RNA 3D predictions on a single GPU—via careful memory/compute optimizations (attention approximations, slimmer MLPs, residual MLP refinements). Benchmarked with reproducible scripts and small pre-trained weights; shows lower RMSD vs. lightweight baselines. Submitted to NeurIPS 2025 (Workshop); preprint available.

## Run Instructions
1. Install dependencies:
   ```bash
   pip install torch transformers egnn-pytorch rotary-embedding-torch tqdm scikit-learn matplotlib pandas kagglehub
   ```
2. Prepare data:
   - Place required CSVs in `ESMFold_v1/kaggle-data/` and `RibonanzaNet/kaggle-data/`.
   - Use `data-conversion.py` or `data-conversion-kaggle.py` to preprocess/split data.
3. Train or evaluate:
   - Run main training scripts:
     - `ESMFold_v1/training-esm-tx-egnn-v1a.py` (ESMFold pipeline)
     - `RibonanzaNet/training-v5a.py`, `training-mod-arch-v6a.py`, `ryanmehra-rna3dfolding-v4a.py`, `ryanmehra-rna3dfolding-v5a.py` (RibonanzaNet variants)
   - Use `download_models.py` to fetch pre-trained weights.
   - Use `scores_chart.py` to visualize benchmarking results.

## Code File Summary
- `training-esm-tx-egnn-v1a.py`: Main ESMFold training pipeline (Transformer + EGNN).
- `training-v5a.py`, `training-mod-arch-v6a.py`, `ryanmehra-rna3dfolding-v4a.py`, `ryanmehra-rna3dfolding-v5a.py`: RibonanzaNet training scripts (various architectures).
- `data-conversion.py`, `data-conversion-kaggle.py`: Data preprocessing and splitting utilities.
- `download_models.py`: Downloads pre-trained weights and datasets.
- `scores_chart.py`: Plots RMSD/dRMAE benchmarking results.
- `example.py`: Example for loading/inspecting RNA datasets.

## License & Attribution
All code is owned by Ryan Mehra and licensed for free use.
