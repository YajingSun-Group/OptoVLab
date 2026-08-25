# OptoVLab static database browser

`apps/database-web` is a standalone static browser for three organic
optoelectronic device datasets:

- OLED: exported from a reviewed `mining_platform` SQLite snapshot.
- OFET: exported from a user-supplied workbook.
- OPV: imported from a separately obtained, appropriately licensed JSON source.

The public repository bundles only tiny demonstration records. The site is
independent of the review UI and FastAPI. Full data files can be generated
as gzip-compressed JSON and loaded only when the corresponding OLED, OFET, or
OPV tab is opened.

## Generate data

From the repository root:

```bash
uv run python scripts/build_database_site_data.py \
  --oled-database /path/to/platform.sqlite \
  --ofet-workbook /path/to/OFET-summary.xlsx \
  --opv-json /path/to/OPV2D/docs/opv_data.json
```

If the OPV2D reference checkout does not exist:

```bash
git clone --depth 1 https://github.com/sunyrain/OPV2D.git /tmp/OPV2D-reference
```

## Run locally

```bash
cd apps/database-web
npm ci
npm run dev
```

Open `http://127.0.0.1:3000`. For a production static export:

```bash
npm run build
```

The deployable files are written to `out/`.

## OLED quality tiers

- `Human final`: the paper was explicitly finalized with `Confirm Paper`; the
  displayed nested JSON comes from `candidate_final_records`.
- `Auto reviewed`: all device-used materials were completed by the mining
  platform, but the paper was not explicitly frozen with `Confirm Paper`.

These tiers are intentionally kept distinct in the public browser.

## Data boundary

The committed `public/data/*.json.gz` files are demonstration data, not the
paper dataset. Generated full exports must be reviewed for publisher rights,
participant privacy, source attribution, and repository size before release.
