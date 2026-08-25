# Contributing

OptoVLab is a research codebase. Changes should preserve scientific provenance,
human-review boundaries, and reproducibility.

## Development setup

```bash
uv sync --group dev
npm --prefix apps/web ci
npm --prefix apps/database-web ci
```

Create a branch, keep changes scoped, and run:

```bash
uv run ruff check .
uv run pytest tests/mining_platform tests/optovlab tests/database_site
uv run python scripts/check_public_release.py
npm --prefix apps/web run build
npm --prefix apps/database-web run build
```

OLED-GAT changes also require:

```bash
uv sync --extra modeling --group dev
uv run pytest tests/oled_gat
```

## Scientific changes

- Preserve source evidence and review history; never silently replace an
  extracted value.
- Add a regression test when changing a schema, resolver, validator, or review
  transition.
- Label synthetic examples and schematic curves explicitly.
- Report the data split, random seed, target definition, and evaluation metric
  for modeling changes.
- Do not describe attention weights or correlations as causal evidence.

## Data and secrets

Do not commit publisher PDFs, browser profiles, runtime SQLite databases,
private datasets, model checkpoints, API keys, email credentials, or absolute
server paths. Use `.env.example` and small synthetic fixtures instead.

## Pull requests

Describe the behavior changed, the scientific or engineering rationale, the
commands used for verification, and any data or deployment migration required.
