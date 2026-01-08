# Self-Supervised Laplace Approximation

This repository contains code to reproduce the experimental results for **self-supervised laplace approximation** across multiple settings, including:
- conjugate prior toy models,
- heteroscedastic regression, and
- real-world benchmarks (UCI datasets).

## Getting started

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate  # Windows PowerShell
```

Install dependencies (Poetry):

```bash
pip install poetry
poetry install
```

## Project structure

- `experiments/` — runnable experiment suites and their outputs (tables/plots)
- `src/` — core implementations (models, utilities, and SSLA/ASSLA code)
- `scripts/` — convenience entry points to run experiments from the command line

### Note on reproducibility

Some experiment scripts write their outputs into the corresponding `experiments/**/` subfolders. Running an experiment may overwrite existing outputs. Where applicable, experiments are seeded so outputs should be reproducible.
