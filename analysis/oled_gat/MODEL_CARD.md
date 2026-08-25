# OLED-GAT Model Card

## Intended use

Predict the maximum external quantum efficiency (`EQEmax`, percent) of an OLED
device from:

- its ordered multilayer stack;
- layer roles, material identities, and reported thicknesses;
- EML component roles and reported composition ratios;
- atom/bond graphs, RDKit descriptors, and Morgan fingerprints for EML
  materials;
- device-level mechanism, color, fabrication, and emission geometry fields.

The model outputs a conditional mean and non-crossing q10/q50/q90 estimates.

## Dataset

The frozen V1 corpus contains 4,228 devices from 1,144 papers. It includes
single-junction, non-white OLEDs with organic small-molecule emitters, plausible
reported `EQEmax`, and valid SMILES for every EML component.

Dataset fingerprint:

```text
8e36123fb67d584adcdd801151db9201b44edbf59f0c25ca36945e59114cea93
```

The device-random benchmark contains the same records with a different frozen
split:

```text
b320d0547644b0cf43fe20938649eb750c5ecada3f0f36b004018b8c945b3936
```

The within-paper campaign benchmark contains 1,969 devices from 253 papers,
each with at least five in-scope devices:

```text
9e36638ed875583e66174cfb5a84706c3790f8a852dc91712d67f046e70d4496
```

## Model

The hierarchy is:

```text
device root
  -> ordered layer nodes
  -> material nodes
  -> atom/bond nodes for EML molecular structures
```

The encoder uses four residual `GATv2Conv` layers. Readout separately pools all
layers, all materials, the EML layer, and EML materials. A shared head predicts
the conditional mean plus a median and positive lower/upper deltas, which
guarantees `q10 <= q50 <= q90`.

## Frozen results

Within-campaign test set, 303 devices from 253 papers:

| Model | R2 | MAE | RMSE | Spearman |
|---|---:|---:|---:|---:|
| Best single refit OLED-GAT | 0.8001 | 2.7407 | 3.9657 | 0.8859 |
| Validation-weighted 3x OLED-GAT | **0.8078** | **2.5613** | **3.8887** | **0.8993** |

The achieved model is a point-prediction model. Its quantile loss is disabled,
so its q10/q50/q90 outputs must not be used.

Device-random test set, 423 devices:

| Model | R2 | MAE | RMSE | Spearman |
|---|---:|---:|---:|---:|
| OLED-GAT | 0.7100 | 3.4518 | 5.0806 | 0.8430 |
| CatBoost | 0.7309 | 3.5052 | 4.8942 | 0.8597 |
| Fixed ensemble | 0.7529 | 3.2954 | 4.6898 | 0.8690 |

The 80% conformalized ensemble interval has 79.43% test coverage and mean width
14.33 EQE percentage points.

The strict DOI-grouped validation benchmark is substantially lower (current
ensemble R2 approximately 0.59). Therefore the random-split result must not be
presented as performance on unseen papers.

## Limitations

- Most records are automatically extracted.
- The input lacks PLQY, transition-dipole orientation, HOMO/LUMO alignment,
  mobility, morphology, and detailed processing conditions.
- Sibling devices from the same paper occur across splits in the achieved
  device-random benchmark.
- Every paper occurs across splits in the campaign benchmark by design. The
  campaign result estimates topology interpolation within a known experimental
  campaign, not unseen-paper extrapolation.
- Applicability is limited to the V1 organic-small-molecule scope.
- A manually audited, DOI-disjoint external test set is required before
  experimental deployment claims.
- The bundled 4CzIPN/PPF example predicts substantially below the user's
  reported 26.5% experiment. This case demonstrates that acceptable aggregate
  random-split R2 does not guarantee reliable architecture optimization for a
  specific chemistry.
