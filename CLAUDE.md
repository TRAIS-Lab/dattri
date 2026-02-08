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

## Contributing 

### Project Structure

- **`dattri/`**: `AttributionTask` handles the ML model, training loss functions, and target function for attribution. Keep Attributors cleanly separated from `AttributionTask` to maintain generalizability across models, loss functions, and target functions. When implementing a new Attributor, inherit from an existing Attributor class to maximally reuse code. If the new Attributor is completely different from existing ones, inherit `BaseAttributor` to ensure API consistency.
- **`examples/`**: Bite-size examples, preferably a single script without a README. Examples should not be computationally heavy; heavy workloads belong in `experiments/` instead.
- **`experiments/`**: Large experiments/benchmarks requiring multiple files. One folder per experiment, preferably with a `README.md`.

### CI/CD

- **Unit tests** (`test/`, `.github/workflows/pytest.yml`): Run-through tests and sanity checks. The test directory (almost) mirrors the structure of `dattri/`. A new unit test should be included whenever non-trivial changes are made in `dattri/` (new feature or bug fix). GPU tests are omitted by default in GitHub Actions and can be manually triggered by commenting "run gpu test" on the PR.
- **Example tests** (`.github/workflows/examples_test.yml`): A new example test should be added whenever a new example is added in `examples/`. An example test is a line in the workflow file that runs the example script.

### Pull Requests

- For non-trivial PRs (new feature, bug fix, significant refactoring), start with a new Issue outlining the planned changes for maintainer approval. This is **required** if the changes involve major API design changes.
- PRs can start as `[WIP]` without all tests passing to get early feedback.
- Code review focuses on correctness, sufficient and clear comments/docstrings, and whether unit tests or example tests are properly added.
- Manually trigger "run gpu test" if the changes involve existing GPU tests.
- Always use **squash and merge** to merge a PR.
- The document test runs after the PR is merged. If it fails, fix it promptly.
- Close the related issues after the document test passes.
