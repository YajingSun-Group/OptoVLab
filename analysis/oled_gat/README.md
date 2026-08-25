# OLED-GAT

`OLED-GAT` predicts maximum external quantum efficiency (`EQEmax`) from an
OLED's ordered multilayer structure and the molecular graphs of its emissive
layer materials.

The current pure-GNN result for within-campaign device optimization is:

```text
frozen test R2 = 0.8078
MAE = 2.5613 EQE points
RMSE = 3.8887 EQE points
```

It uses three validation-weighted OLED-GAT seeds and no CatBoost. See
[`OLED_GAT_R2_080_REPORT.md`](OLED_GAT_R2_080_REPORT.md).

The project keeps two evaluation protocols because they answer different
questions:

```text
paper_grouped: generalization to unseen papers
device_random: interpolation among literature devices
```

The full-corpus device-random OLED-GAT remains `R2 = 0.7100`; the strict
DOI-grouped benchmark is lower. These protocols are not interchangeable with
the campaign result.

## V1 scope

- Strict organic-small-molecule emitter scope inherited from
  `oled_device_discovery`.
- Valid, plausible `EQEmax`.
- Every EML component has a valid SMILES.
- Single-junction, non-white OLEDs.
- No explicit outcoupling or capping layer.
- Quantiles 0.10, 0.50, and 0.90 are predicted jointly without crossing.

The first version uses a hierarchical heterogeneous graph:

```text
device root
  -> ordered layer nodes
  -> material nodes
  -> atom/bond molecular graphs for EML materials
```

Non-EML material identities are categorical embeddings. This is mathematically
equivalent to applying a learned linear projection to a sparse one-hot vector,
without materializing a large dense one-hot matrix.

## Commands

Prepare the frozen dataset:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python \
  analysis/oled_gat/run_prepare.py
```

Build graph cache and train the strict DOI-grouped model:

```bash
uv run python analysis/oled_gat/run_build_graphs.py
CUDA_VISIBLE_DEVICES=0 uv run python \
  analysis/oled_gat/run_train.py
```

Reproduce the achieved device-random benchmark:

```bash
CONFIG=analysis/oled_gat/configs/device_random.yaml
uv run python analysis/oled_gat/run_prepare.py --config "$CONFIG"
uv run python analysis/oled_gat/run_build_graphs.py --config "$CONFIG"
CUDA_VISIBLE_DEVICES=0 uv run python analysis/oled_gat/run_train.py --config "$CONFIG"
CUDA_VISIBLE_DEVICES=0 uv run python analysis/oled_gat/run_baseline.py --config "$CONFIG"
CUDA_VISIBLE_DEVICES=0 uv run python analysis/oled_gat/run_evaluate_frozen.py --config "$CONFIG"
uv run python analysis/oled_gat/run_report.py --config "$CONFIG"
```

Reproduce the pure-GNN campaign benchmark:

```bash
CONFIG=analysis/oled_gat/configs/campaign_gat.yaml
uv run python analysis/oled_gat/run_prepare.py --config "$CONFIG"
uv run python analysis/oled_gat/run_build_graphs.py --config "$CONFIG"

for SEED in 20260726 20260727 20260728; do
  CUDA_VISIBLE_DEVICES=0 uv run python analysis/oled_gat/run_train.py \
    --config "$CONFIG" --run-name "gat_seed_${SEED}" --seed "$SEED"
done

CUDA_VISIBLE_DEVICES=0 uv run python \
  analysis/oled_gat/run_refit_gnn_ensemble.py --config "$CONFIG" \
  --runs gat_seed_20260726 gat_seed_20260727 gat_seed_20260728
uv run python analysis/oled_gat/run_campaign_figures.py
```

Predict one schema-complete device with the achieved GAT checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python analysis/oled_gat/run_predict.py \
  analysis/oled_gat/examples/device_4czipn.json
```

Generate publication figures for regression and model architecture:

```bash
uv run python analysis/oled_gat/run_publication_figures.py
```

Run the frozen-test explainability analysis:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python \
  analysis/oled_gat/run_explainability.py
```

Regenerate its figures without repeating attribution inference:

```bash
uv run python analysis/oled_gat/run_explainability.py --plots-only
```

The analysis combines GATv2 attention, attention gradients, physical-unit
thickness gradients, local thickness counterfactuals, root-link ablations, and
attention-head stability checks. See
[`OLED_GAT_EXPLAINABILITY_REPORT.md`](OLED_GAT_EXPLAINABILITY_REPORT.md).
Outputs are stored under
`outputs_device_random/explainability/`.

Every EML component must include a valid `canonical_smiles`. Known materials
should include their platform `global_material_id`; unseen non-EML identities
fall back to the out-of-vocabulary embedding.

Outputs are written under `analysis/oled_gat/outputs/` for DOI-grouped
experiments and `analysis/oled_gat/outputs_device_random/` for the achieved
benchmark.

## Scientific safeguards

- Data normalization and vocabularies are fitted on training papers only.
- The final test split is frozen before model development.
- Validation drives early stopping and hyperparameter selection.
- Test performance is reported once for the frozen achieved configuration.
- Mean and median R2, MAE, RMSE, Spearman correlation, interval coverage,
  interval width, and pinball loss are all reported.
- Device-random and DOI-grouped scores are never presented as equivalent.

## References

- Velickovic et al., Graph Attention Networks, ICLR 2018.
- Brody et al., How Attentive are Graph Attention Networks?, ICLR 2022.
- Koenker and Bassett, Regression Quantiles, Econometrica 1978.
- PyTorch Geometric `GATv2Conv` documentation.
