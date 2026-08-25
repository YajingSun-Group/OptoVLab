# OLED-GAT Project Status

Last updated: 2026-07-27

## Goal

Build a hierarchical graph-attention model that predicts an OLED device's
maximum EQE as a calibrated conditional interval.

Operational success criterion:

```text
R2 > 0.80 for within-campaign multilayer-device optimization
```

Scientific stretch criterion: `R2 > 0.70` on a DOI-grouped external test set.

## Completed

- Existing DOI-grouped tabular baseline audited: mean R2 approximately 0.53.
- Dataset coverage audited:
  - 4,837 valid EQE devices from 1,260 papers.
  - 99.3% have final-emitter SMILES.
  - Approximately 89.6% have SMILES for every EML component.
  - Layer thickness coverage is approximately 82%.
- GPU environment configured with PyTorch 2.9.1 CUDA 12.8 and PyG.
- Real CUDA tensor operations and a `GATv2Conv` forward pass verified on GPU 0.
- Frozen 4,228-device dataset, train-only normalization, and split fingerprints.
- Hierarchical device/layer/material/atom graph cache.
- Four-layer residual `GATv2Conv` model with mean and non-crossing quantile heads.
- CatBoost point and quantile baseline.
- Frozen device-random test evaluation:
  - OLED-GAT R2: 0.7100.
  - CatBoost R2: 0.7309.
  - Fixed ensemble R2: 0.7529.
  - Calibrated 80% interval coverage: 79.43%.
- Reproducible evaluation script, model card, plots, subgroup CSV, and HTML report.
- Schema-validated single-device JSON inference command with conformalized GAT
  intervals.
- Frozen-test explainability analysis:
  - 423 devices and 3,091 layer nodes analyzed.
  - Raw GATv2 attention and output-conditioned attention gradients.
  - 2,319 in-domain layer-thickness counterfactuals.
  - Root-link ablation faithfulness and attention-head stability checks.
  - Layer-role, interface, thickness, and faithfulness figures.
  - Detailed Chinese research report with scientific claim boundaries.
- Ten focused unit tests plus Ruff validation.
- Deterministic `within_paper_campaign` benchmark:
  - 1,969 devices from 253 papers with at least five in-scope devices.
  - Every paper represented in train, validation, and frozen test.
  - Dataset fingerprint
    `9e36638ed875583e66174cfb5a84706c3790f8a852dc91712d67f046e70d4496`.
- Pure OLED-GAT point-prediction result without CatBoost:
  - Validation-selected three-seed weights: 0.43 / 0.37 / 0.20.
  - Final train-plus-validation refit.
  - Frozen test R2: 0.8078.
  - Frozen test MAE: 2.5613 EQE points.
  - Frozen test RMSE: 3.8887 EQE points.
  - Frozen test Spearman: 0.8993.
- Experimental staged molecule/material/layer/interface encoder. Its first
  configuration overfit and is retained as a research branch, not the selected
  production checkpoint.

## Next

- Restore calibrated uncertainty without reducing point accuracy, using
  post-hoc conformal intervals or frozen-encoder quantile heads.
- Improve DOI-grouped generalization using physically meaningful inputs:
  PLQY, dipole orientation, energy levels, mobility, and processing conditions.
- Add CatBoost SHAP/counterfactual attribution so explanations cover the full
  62% CatBoost + 38% OLED-GAT ensemble.
- Replace the homogeneous graph encoder with staged molecule, material, layer,
  interface, and device encoders if interpretability becomes a primary model
  objective.
- Build a manually audited external benchmark.
- Evaluate pretrained molecular encoders without changing the frozen splits.

## Evidence boundary

Most labels are automatically extracted rather than human-finalized. The
achieved R2 includes sibling-paper overlap and must be described as
interpolation on the current mined corpus. A manually reviewed DOI-disjoint
external test set remains necessary for a publication-grade estimate.
