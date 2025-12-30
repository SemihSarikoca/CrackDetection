# CrackDetection
A modular, scalable crack detection toolkit built in Python. The project exposes a configurable pipeline that reads imagery from disk, applies reproducible preprocessing, detects crack candidates, and optionally emits visual summaries. The CLI is powered by Typer, so the module can be used either as a library or as a command-line tool.

## Features
- Config-driven pipeline with dataclass-based configuration objects
- Hand-crafted, multi-scale Canny edge detector (bilateral denoising, dual Gaussian passes, Sobel gradients, NMS, hysteresis) layered with connected-component analytics, grid suppression, and thin-crack rescue
- Optional visualization artifacts saved as multi-panel figures
- Pytest-ready structure with a starter test suite and metadata in `pyproject.toml`

## Technical Notes
- All processing stages (grayscale conversion, bilateral smoothing, Gaussian smoothing, Sobel gradients, Non-Maximum Suppression, hysteresis thresholding, and component analytics) are implemented manually with NumPy to comply with the "no off-the-shelf ML/CV ops" constraint.
- A multi-stage filter blends density checks, directional kernels, perimeter/area heuristics, local variance maps, and multi-scale Canny outputs to suppress grout/hex tiles while re-introducing thin diagonal cracks detected at lighter blur scales.
- When paired `image/` and `label/` directories exist (e.g., `train/image` vs `train/label`), the loader automatically operates on `image/` to avoid accidentally re-processing ground-truth masks; override with `DatasetConfig.image_subdir` if you intentionally need a different branch.
- Pillow is used purely for decoding images from disk; Matplotlib is imported lazily so you can run the pipeline headless unless visualizations are requested.

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
	python -m crack_detection.cli --image-root path/to/images --visualize --output-dir outputs
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
