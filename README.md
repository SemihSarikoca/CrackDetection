# CrackDetection
A modular, scalable crack detection toolkit built in Python. The project exposes a configurable pipeline that reads imagery from disk, applies reproducible preprocessing, detects crack candidates, and optionally emits visual summaries. The CLI is powered by Typer, so the module can be used either as a library or as a command-line tool.

## Features
- Config-driven pipeline with dataclass-based configuration objects
- Canny edge detector baseline with connected-component filtering
- Optional visualization artifacts saved as multi-panel figures
- Pytest-ready structure with a starter test suite and metadata in `pyproject.toml`

## Getting Started
1. Create and activate your Python environment (Python \>= 3.10).
2. Install the package in editable mode along with development extras:
	```sh
	python -m venv .venv
	source .venv/bin/activate
	pip install -e '.[development]'
	```
3. Explore the CLI help:
	```sh
	python -m crack_detection.cli --help
	```
4. Run the pipeline:
	```sh
	python -m crack_detection.cli run --image-root path/to/images --visualize --output-dir outputs
	```

## Project Layout
```
src/
  crack_detection/
	 cli.py             # Typer-based entry points
	 config.py          # Dataclass configs for the pipeline
	 data.py            # Dataset streaming helpers
	 detector.py        # Canny-based crack detector implementation
	 pipeline.py        # Pipeline orchestration logic
	 preprocessing.py   # Image preprocessing utilities
	 visualization.py   # Matplotlib-based artifact writers
tests/
  test_pipeline_config.py
pyproject.toml
```

## Next Steps
- Replace the baseline detector with a learning-based approach when training data is available.
- Extend the pipeline with additional metrics (length estimation, severity classification).
- Integrate experiment tracking (Weights & Biases, MLflow) for reproducible research.
