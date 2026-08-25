# Architecture

## Research loop

```text
Scientific PDFs
  -> Data Mining Agent
     -> MinerU document blocks
     -> schema-constrained device extraction
     -> material identity search and cross-validation
     -> eligible molecular-image OCSR
     -> evidence-linked human review
  -> reviewed organic-optoelectronic database
  -> Device Modeling Agent
     -> ordered layer graph
     -> directed adjacent-layer interfaces
     -> quantile EQE prediction and attribution audit
  -> Experimental Design Agent
     -> retrieval-grounded hypothesis
     -> Scientific Critic
     -> researcher approval and experiment
  -> measured result returned to the corpus
```

## Components

### Mining platform

`src/evolab_local/mining_platform/` owns papers, parsed blocks, candidate
records, evidence anchors, material structures, batch execution, review events,
and finalization. Deterministic services expose typed operations; LLM/VLM calls
do not write directly to final records.

The first maintained domain template is
`config/mining_platform/domains/oled_device_v1.yaml`. Device output is nested as
`devices[] -> layers[] -> components[]` plus performance records, because OLED
layer counts and reported working points are variable.

### OptoVLab agent layer

`src/evolab_local/optovlab/` stores conversations, tool events, analysis
artifacts, retrieval indexes, and controlled modeling jobs. It references the
mining platform rather than duplicating its scientific records.

Three specialists are exposed in `apps/web/`:

1. Data Mining Agent for plan-driven PDF extraction and review.
2. Device Modeling Agent for dataset inspection, OLED-GAT, and controlled HPC
   job preparation.
3. Experimental Design Agent for evidence-grounded device recommendations.

### OLED-GAT

`analysis/oled_gat/` contains data preparation, directed device-graph models,
quantile training, evaluation, and explainability exports. Layer order is part
of the graph topology. Attention is retained as an importance signal for
hypothesis generation; it is not treated as proof of a physical mechanism.

### User interfaces

- `apps/web/`: live agent workbench backed by FastAPI.
- `apps/database-web/`: static OLED/OFET/OPV browser backed by compressed JSON.
- `site/`: dependency-free publication demonstration deployed to GitHub Pages.

## Persistence and provenance

Runtime state is kept under `runtime/` and excluded from Git. The mining SQLite
database is authoritative for candidate values, evidence, reviews, and final
records. Every human modification is represented as a review event so the raw
model output remains recoverable.

## Safety boundaries

- Final scientific records require an explicit review transition.
- Missing evidence stays unresolved; it is not filled from model intuition.
- Model training can be prepared through the UI but submission requires an
  explicit confirmation.
- Browser automation does not solve CAPTCHA, bypass paywalls, or enter account
  credentials.
- The full web application has no built-in authentication in v0.1.0.
