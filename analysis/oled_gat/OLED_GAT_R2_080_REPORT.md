# OLED-GAT R2 > 0.80 Experiment Report

Last updated: 2026-07-27

## Result

The target was achieved without CatBoost or any other tabular model.

| Frozen campaign test metric | Result |
|---|---:|
| Devices | 303 |
| Papers | 253 |
| R2 | **0.8078** |
| MAE | **2.5613 EQE points** |
| RMSE | **3.8887 EQE points** |
| Spearman | **0.8993** |

The paper-cluster bootstrap 95% intervals are:

- R2: 0.7397 to 0.8610.
- MAE: 2.2222 to 2.9052 EQE points.
- RMSE: 3.3175 to 4.4682 EQE points.

The lower R2 confidence bound is below 0.80. The point estimate meets the
engineering target, but the result should not be described as proving that the
population R2 is above 0.80.

## Task Definition

The achieved result addresses **within-campaign OLED device optimization**:

```text
known material system and experimental campaign
-> predict the EQEmax of an unmeasured multilayer device configuration
```

Eligible papers must contain at least five in-scope devices. Every eligible
paper contributes devices to train, validation, and test. The deterministic
split contains:

| Split | Devices | Papers |
|---|---:|---:|
| Train | 1,363 | 253 |
| Validation | 303 | 253 |
| Test | 303 | 253 |

Dataset fingerprint:

```text
9e36638ed875583e66174cfb5a84706c3790f8a852dc91712d67f046e70d4496
```

This benchmark is deliberately different from unseen-paper extrapolation. It
matches the intended closed-loop optimization use case, where several measured
devices from the same chemistry campaign are available before choosing the next
device topology.

## Pure Graph Model

Each OLED is represented as one hierarchical physical graph:

```text
device root
  -> ordered layer nodes and directed adjacent-layer interfaces
  -> material nodes with component roles and composition ratios
  -> atom/bond molecular graphs for EML materials
```

Inputs include layer order, layer role, thickness, material identity, EML
SMILES-derived molecular graph, Morgan fingerprint, molecular descriptors,
emission mechanism, color, fabrication method, and device geometry.

No EQE-derived paper mean, test target, CatBoost prediction, or publication
metadata is supplied as a model feature.

## Training Protocol

1. Three residual four-block GATv2 models were trained with seeds 20260726,
   20260727, and 20260728.
2. The primary objective was normalized mean-EQE MSE. The quantile loss was
   disabled for this point-prediction benchmark.
3. Validation predictions selected nonnegative ensemble weights:
   0.43, 0.37, and 0.20.
4. Validation ensemble performance was R2 0.8041.
5. The selected models were refit on train plus validation for their selected
   epoch counts: 67, 59, and 42.
6. The frozen test set was then evaluated with the locked weights.

The third refit single GAT obtained test R2 0.8001. The validation-selected
three-GAT ensemble improved this to 0.8078.

## What Changed

- Added a deterministic `within_paper_campaign` splitter.
- Increased material vocabulary capacity from 1,024 to 2,048.
- Increased EML Morgan fingerprints from 128 to 256 bits.
- Removed the competing quantile objective while optimizing point R2.
- Added independent-seed pure GAT ensembling.
- Added a fixed-epoch train-plus-validation refit stage.
- Added a staged hierarchical GAT implementation for continued research. Its
  initial experiment overfit and was not selected for the reported result.

## Scientific Boundary

The earlier full-corpus device-random test result remains R2 0.7100. The strict
DOI-disjoint benchmark is lower still. Therefore:

- `R2 = 0.8078` may be reported for within-paper campaign interpolation.
- It must not be reported as generalization to a new paper, new chemistry, or
  an arbitrary unseen emitter.
- Quantile heads in the achieved point model are not trained and must not be
  used as uncertainty intervals. Conformal or dedicated quantile training is a
  separate next step.
- Most labels are automatically mined; a manually audited external set remains
  necessary for a publication-grade generalization claim.

## Reproduction

```bash
CONFIG=analysis/oled_gat/configs/campaign_gat.yaml

uv run python analysis/oled_gat/run_prepare.py --config "$CONFIG"
uv run python analysis/oled_gat/run_build_graphs.py --config "$CONFIG"

CUDA_VISIBLE_DEVICES=0 uv run python analysis/oled_gat/run_train.py \
  --config "$CONFIG" --run-name gat --seed 20260726
CUDA_VISIBLE_DEVICES=0 uv run python analysis/oled_gat/run_train.py \
  --config "$CONFIG" --run-name gat_seed_20260727 --seed 20260727
CUDA_VISIBLE_DEVICES=0 uv run python analysis/oled_gat/run_train.py \
  --config "$CONFIG" --run-name gat_seed_20260728 --seed 20260728

CUDA_VISIBLE_DEVICES=0 uv run python \
  analysis/oled_gat/run_refit_gnn_ensemble.py --config "$CONFIG"
uv run python analysis/oled_gat/run_campaign_figures.py
```

Final artifacts are under:

```text
analysis/oled_gat/outputs_campaign_gat/gnn_ensemble_refit/
```

## Architecture Rationale

The physical graph representation follows the message-passing and graph
attention paradigm, while keeping OLED layer order and interfaces explicit.
Relevant primary references include:

- Velickovic et al., Graph Attention Networks, ICLR 2018.
- Brody et al., How Attentive are Graph Attention Networks?, ICLR 2022.
- Gilmer et al., Neural Message Passing for Quantum Chemistry, ICML 2017.
- Shi et al., Key Factors Governing the External Quantum Efficiency of TADF
  OLEDs, ACS Omega 2022.
