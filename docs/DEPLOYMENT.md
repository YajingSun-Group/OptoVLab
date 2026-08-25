# Deployment

## Public static site

The `site/` directory is self-contained. Test it locally with:

```bash
python -m http.server 4173 --directory site
```

`.github/workflows/pages.yml` publishes this directory after every successful
push to `main`. In the GitHub repository, select **Settings -> Pages -> Source:
GitHub Actions** once. The expected URL is:

```text
https://yajingsun-group.github.io/OptoVLab/
```

The public site is an interactive demonstration, not a hosted inference API.

## Full local platform

### Prerequisites

- Python 3.11 or newer
- uv
- Node.js 22 or newer and npm
- Linux is recommended for optional Slurm and GPU integration

Install dependencies:

```bash
uv sync --group dev
npm --prefix apps/web ci
npm --prefix apps/database-web ci
cp .env.example .env
```

Start all three development services:

```bash
./scripts/dev.sh
```

| Service | Default address | Purpose |
|---|---|---|
| FastAPI | `127.0.0.1:8000` | Mining and OptoVLab APIs |
| Agent workbench | `127.0.0.1:5175` | Three-agent UI |
| Database browser | `127.0.0.1:3000` | OLED/OFET/OPV explorer |

Stop the stack with `Ctrl+C`. Logs are written to
`runtime/optovlab/dev-logs/`.

### Optional scientific services

The checked-in demo works without paid APIs. Real extraction can use:

- DeepSeek-compatible text extraction;
- Qwen-compatible vision and identity checks;
- MinerU document parsing;
- DECIMER segmentation and image-to-SMILES;
- PubChem, OPSIN, OpenAlex, and an optional web-search provider.

Set URLs and credentials in `.env`; never add secrets to YAML or source files.
Service defaults are loopback addresses and can be replaced by remote internal
endpoints.

Initialize the mining runtime and inspect the CLI:

```bash
uv run mining-platform init-runtime
uv run mining-platform --help
```

### Model development

Install the optional graph-model stack separately:

```bash
uv sync --extra modeling --group dev
uv run pytest tests/oled_gat
```

Training inputs and checkpoints are intentionally absent. Update
`analysis/oled_gat/configs/*.yaml` to reference a prepared local dataset before
starting a run.

## Production notes

Build the frontends with `npm run build`, serve the Vite output and Next.js
static export behind HTTPS, and reverse proxy `/api` to FastAPI. Add
authentication and authorization before exposing the full workbench outside a
trusted group network. Restrict filesystem access and Slurm submission to a
dedicated service account.

After any backend, frontend, or configuration change, restart the affected
service and verify its live endpoint before treating the deployment as updated.
