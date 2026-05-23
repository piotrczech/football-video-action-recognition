# Project Guidelines

This file defines lightweight team rules for this repository. The goal is to keep local and cluster work consistent from the first implementation branches.

## 1. Environment Rules (Local vs Cluster)
- All training and preprocessing should be runnable via Python scripts (`.py`), not notebook-only flows.
- Never hardcode user-specific absolute paths.
- All paths must be provided through script arguments or config files.
- Cluster-specific settings must stay in config/CLI values, not in reusable pipeline logic.
- Project-wide paths and inference defaults live in `configs/project.yaml` (loaded by `murawa.settings`).
- Training hyperparameters live in `configs/train.full.yaml` and `configs/train.quick.yaml` only.

## 2. Run Naming and Artifact Standards
Use a predictable run identifier:

`<model>_<datasetVariant>_<YYYYMMDD-HHMM>_<tag>`

Each training run should persist at least:
- model weights,
- resolved configuration,
- class mapping,
- basic training metadata,
- metrics summary,
- dataset variant metadata.

## 3. Reproducibility Rules
- Use fixed random seeds where applicable.
- Log the final config used for each run.
- Record key metadata needed to rerun or compare experiments.
- When changing workflow assumptions, document them in the related issue and update repository docs.

## 4. Decision and Documentation Discipline
- Work should be issue-driven (`issue-first`).
- Use one branch per issue (`feature/<issue-id>-<short-name>`).
- Merge through pull requests.
- Keep implementation decisions written in repository context files, not only in chat messages.

## 5. Scope Boundaries for Baseline Branch

The historical **baseline** branch contained only documentation and repository hygiene. The current **main** branch includes the full semester implementation (data pipeline, model adapters, inference services, Streamlit UI).

Regardless of branch, keep these rules:
- paths and inference defaults in `configs/project.yaml` via `murawa.settings`,
- training hyperparameters in `configs/train.full.yaml` and `configs/train.quick.yaml` only,
- no user-specific absolute paths in reusable code.
