# Project Guidelines

This file defines lightweight team rules for this repository. The goal is to keep local and cluster work consistent from the first implementation branches.

## 1. Environment Rules (Local vs Cluster)
- All training and preprocessing should be runnable via Python scripts (`.py`), not notebook-only flows.
- Never hardcode user-specific absolute paths.
- All paths must be provided through script arguments or config files.
- Cluster-specific settings must stay in config/CLI values, not in reusable pipeline logic.

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
This baseline branch is intentionally limited to:
- documentation and project conventions,
- dependency/bootstrap metadata,
- repository hygiene (`.gitignore`, standards).

The following are out of scope here:
- final directory scaffolding,
- production pipeline modules,
- model adapter implementation,
- Streamlit feature implementation.
