# CLAUDE.md

## Project Overview

dattri is a PyTorch library for developing, benchmarking, and deploying efficient data attribution algorithms. Data attribution quantifies how much each training sample influences model predictions. Published at NeurIPS 2024.

## Build & Development

```bash
# Install (development)
pip install -e .[test]

# Install with CUDA acceleration
pip install -e .[test,sjlt]

# Recommended environment
conda create -n dattri python=3.10
conda activate dattri
pip install -e .[test]
```

## Commands

```bash
# Run all checks (lint + tests)
make test

# Run tests only
pytest -v test/

# Run tests excluding GPU tests
pytest -v -m "not gpu" test/

# Run specific test file
pytest -v test/dattri/algorithm/test_influence_function.py

# Lint
ruff check

# Format
ruff format

# Lint with auto-fix
ruff check --fix
```

## Code Style

- **Formatter/Linter**: Ruff (preview mode, line length 88)
- **Docstrings**: Google-style
- **Type hints**: Required on all function parameters and return types
- **Naming**: PascalCase for classes, snake_case for functions, UPPER_SNAKE_CASE for constants
- **Imports**: `from __future__ import annotations` at top; use `TYPE_CHECKING` for expensive imports
- **Import order**: stdlib, third-party, local (`dattri.*`)
- **Python version**: 3.8+ compatible (no modern syntax like `X | Y` unions)

## Architecture

- **`dattri/algorithm/`**: Attribution algorithms (IF, TracIn, TRAK, RPS, Shapley, DVEmb, etc.)
  - All inherit from `BaseAttributor` or `BaseInnerProductAttributor`
  - Two-phase pattern: `cache()` precomputes, `attribute()` scores
- **`dattri/func/`**: Low-level utilities (HVP/IHVP, Fisher, random projection)
- **`dattri/benchmark/`**: Datasets and models for standardized evaluation
- **`dattri/metric/`**: Evaluation metrics (LOO, LDS, AUC, brittleness)
- **`dattri/model_util/`**: Model utilities (retraining, dropout, hooks)
- **`dattri/task.py`**: `AttributionTask` abstraction decoupling algorithms from tasks

## Testing

- Framework: pytest
- Tests mirror source structure under `test/dattri/`
- GPU tests marked with `@pytest.mark.gpu`
- Tests use synthetic data (TensorDataset) for speed
