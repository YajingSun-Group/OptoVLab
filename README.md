<p align="center">
  <img src="site/assets/optovlab_logo.svg" alt="OptoVLab" width="360">
</p>

<p align="center">
  <strong>A self-improving agentic virtual laboratory for organic optoelectronic research.</strong>
</p>

<p align="center">
  <a href="https://yajingsun-group.github.io/OptoVLab/">Online demo</a> ·
  <a href="docs/ARCHITECTURE.md">Architecture</a> ·
  <a href="docs/DEPLOYMENT.md">Deployment</a> ·
  <a href="docs/DATA_AND_REPRODUCIBILITY.md">Data and reproducibility</a>
</p>

<p align="center">
  
[![CI](https://github.com/YajingSun-Group/OptoVLab/actions/workflows/ci.yml/badge.svg)](https://github.com/YajingSun-Group/OptoVLab/actions/workflows/ci.yml)
[![Pages](https://github.com/YajingSun-Group/OptoVLab/actions/workflows/pages.yml/badge.svg)](https://github.com/YajingSun-Group/OptoVLab/actions/workflows/pages.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-137f78.svg)](LICENSE)
</p>
OptoVLab connects three specialist agents in an auditable research loop:

- **Data Mining Agent:** turns scientific PDFs into device-grouped records with
  evidence anchors, material identity resolution, and human review.
- **Device Modeling Agent:** represents ordered OLED layer stacks as directed
  graphs and predicts maximum EQE with quantile-regression OLED-GAT models.
- **Experimental Design Agent:** retrieves device precedents, proposes
  physically motivated experiments, and keeps the researcher in control.

The repository accompanies the manuscript *A Self-improving Agentic Virtual
Laboratory for Organic Optoelectronic Research*. It contains the platform code,
OLED mining schema, OLED-GAT implementation, reproducible demo records, and a
static publication website. The paper corpus, publisher PDFs, private API keys,
runtime databases, and trained checkpoints are deliberately excluded.

## Online Demo

The [GitHub Pages site](https://yajingsun-group.github.io/OptoVLab/) is a
dependency-free, deterministic walkthrough of the three-agent workflow. It
does not call commercial models or expose the research server. Quantitative
values shown there are manuscript results; animated tool activity and example
records are explicitly presented as demonstrations.

## Repository Layout

| Path | Purpose |
|---|---|
| `site/` | Public GitHub Pages demonstration |
| `apps/web/` | Full React agent workbench |
| `apps/database-web/` | Static OLED/OFET/OPV database browser |
| `src/evolab_local/mining_platform/` | Mining, evidence, materials, review, and batch APIs |
| `src/evolab_local/optovlab/` | Agent sessions, analysis, RAG, and modeling orchestration |
| `analysis/oled_gat/` | Directed device-graph model, training, evaluation, and explainability |
| `config/` | Versioned, secret-free configuration and OLED schema |
| `examples/data/` | Small demonstration datasets only |
| `tests/` | Unit and API regression tests |

## Quick Start

Prerequisites: Python 3.11+, [uv](https://docs.astral.sh/uv/), Node.js 22+, and
npm.

```bash
git clone https://github.com/YajingSun-Group/OptoVLab.git
cd OptoVLab

uv sync --group dev
npm --prefix apps/web ci
npm --prefix apps/database-web ci
cp .env.example .env
```

Start the complete local demonstration stack:

```bash
./scripts/dev.sh
```

Open:

- Agent workbench: `http://127.0.0.1:5175`
- Database browser: `http://127.0.0.1:3000`
- FastAPI documentation: `http://127.0.0.1:8000/docs`

The bundled demo data are sufficient for browsing, deterministic analysis, and
API/UI development. Real PDF extraction additionally requires the optional
services and credentials listed in [Deployment](docs/DEPLOYMENT.md).

## Verification

```bash
uv run ruff check .
uv run pytest tests/mining_platform tests/optovlab tests/database_site
uv run python scripts/check_public_release.py
npm --prefix apps/web run build
npm --prefix apps/database-web run build
```

OLED-GAT tests require the optional modeling environment:

```bash
uv sync --extra modeling --group dev
uv run pytest tests/oled_gat
```

For the publication site:

```bash
uv run python -m http.server 4173 --directory site
uv run python scripts/verify_public_site.py --base-url http://127.0.0.1:4173
```

## Data Boundary

This code release does **not** redistribute publisher PDFs or the complete
19,175-device research corpus. The small JSON files under `examples/data/` and
`apps/database-web/public/data/` exist only to make the interfaces runnable.
See [Data and Reproducibility](docs/DATA_AND_REPRODUCIBILITY.md) before using
demo values for analysis.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). Until the
article receives its final bibliographic record, cite this software release and
the accompanying manuscript title.

## Contributing and Security

Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) before proposing changes. Do
not open a public issue containing unpublished PDFs, API keys, credentials, or
private datasets; follow [`SECURITY.md`](SECURITY.md) instead.

## License

Source code is released under the [MIT License](LICENSE). Third-party datasets
and services retain their own terms; no third-party research dataset is bundled
in this release.
