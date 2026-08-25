# Data and Reproducibility

## What is included

The release contains only small demonstration records:

- `examples/data/oled_demo.json`: four manuscript demonstration devices;
- `examples/data/ofet_demo.json`: two synthetic UI records;
- `examples/data/opv_demo.json`: two synthetic UI records;
- matching gzip files under `apps/database-web/public/data/`.

The catalog labels each dataset as a demonstration and records its count,
compressed size, and SHA-256 digest. These files validate schemas and user
interfaces; they are not suitable for scientific benchmarking.

## What is not included

- publisher PDFs and supplementary information;
- the complete mined OLED corpus;
- private OFET source workbooks;
- third-party OPV datasets;
- API keys, runtime databases, review queues, or browser profiles;
- trained model checkpoints and large generated outputs.

This boundary prevents accidental redistribution and keeps model claims
separate from runnable UI examples.

## Rebuilding the database browser

After obtaining inputs under their applicable licenses, run:

```bash
uv run python scripts/build_database_site_data.py \
  --oled-database /path/to/reviewed-platform.sqlite \
  --ofet-workbook /path/to/ofet.xlsx \
  --opv-json /path/to/opv.json \
  --output-directory apps/database-web/public/data
```

The exporter reads the SQLite snapshot in read-only mode, distinguishes human
finalized from auto-reviewed OLED records, and writes compressed JSON plus a
catalog manifest.

## Reproducing model results

The manuscript reports a specific frozen data snapshot and split. The code in
`analysis/oled_gat/` exposes preparation, training, evaluation, quantile
prediction, and attribution routines, but the numerical result cannot be
independently reproduced from the demonstration records. A complete archival
release should include, subject to rights review:

1. a DOI-free or appropriately licensed model-ready feature table;
2. immutable train/validation/test membership identifiers;
3. preprocessing configuration and random seeds;
4. checkpoint hashes and environment lock information;
5. evaluation outputs for MAE, R2, interval coverage, and calibration.

Until that archive is available, treat the website metrics as manuscript
results rather than a benchmark reproduced by this public repository.
