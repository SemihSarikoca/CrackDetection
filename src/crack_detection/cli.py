"""Command line entry points for the crack detection pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .config import DatasetConfig, PipelineConfig
from .pipeline import CrackDetectionPipeline

app = typer.Typer(help="Run crack detection pipelines over image datasets.")


@app.command()
def run(
    image_root: Path = typer.Option(..., help="Directory containing crack images to process."),
    output_dir: Optional[Path] = typer.Option(None, help="Directory where visualizations will be written."),
    sample_limit: Optional[int] = typer.Option(None, help="Limit the number of samples processed."),
    visualize: bool = typer.Option(False, help="Persist visualization artifacts for each image."),
) -> None:
    """Execute the default crack detection pipeline."""

    config = PipelineConfig(
        dataset=DatasetConfig(image_root=image_root),
        output_dir=output_dir,
        sample_limit=sample_limit,
        visualize=visualize,
    )
    pipeline = CrackDetectionPipeline(config)
    pipeline.run()


def main() -> None:  # pragma: no cover - Typer handles invocation
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
