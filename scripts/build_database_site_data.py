from __future__ import annotations

import json
from pathlib import Path

import typer

from evolab_local.database_site import export_static_database_data


ROOT = Path(__file__).resolve().parents[1]
app = typer.Typer(no_args_is_help=False)


@app.command()
def main(
    oled_database: Path = typer.Option(
        ...,
        "--oled-database",
        exists=True,
        dir_okay=False,
        help="Read-only OLED mining-platform SQLite snapshot.",
    ),
    ofet_workbook: Path = typer.Option(
        ...,
        "--ofet-workbook",
        exists=True,
        dir_okay=False,
        help="OFET source workbook.",
    ),
    opv_json: Path = typer.Option(
        ...,
        "--opv-json",
        exists=True,
        dir_okay=False,
        help="OPV2D browser JSON source.",
    ),
    output_directory: Path = typer.Option(
        ROOT / "apps/database-web/public/data",
        "--output-directory",
        file_okay=False,
        help="Static website data directory.",
    ),
) -> None:
    manifest = export_static_database_data(
        oled_database=oled_database.resolve(),
        ofet_workbook=ofet_workbook.resolve(),
        opv_json=opv_json.resolve(),
        output_directory=output_directory.resolve(),
    )
    typer.echo(
        json.dumps(
            {
                "generated_at": manifest.generated_at,
                "datasets": [
                    {
                        "key": dataset.key,
                        "records": dataset.record_count,
                        "papers": dataset.paper_count,
                        "compressed_bytes": dataset.compressed_bytes,
                    }
                    for dataset in manifest.datasets
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    app()
