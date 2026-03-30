# Stage 3. Model Training and Extended Data Format

Stage 3 moves the project from prepared data and a shared input flow to actual model experiments. This is where the shared architecture from Stage 1 and the prepared data from Stage 2 begin to work in practice.

The goal of this stage is to reach a point where:
- both models are connected to the same high-level project interface,
- their training can be launched in a consistent way through `.py` scripts,
- training can be executed on the university cluster without manual code fixes,
- models save their outputs in a predictable format,
- trained checkpoints can later be used locally for inference and integration with the downstream pipeline.

This stage is critical, because this is where the first real models for comparison are created. If it is done poorly, then even with good data the final comparisons will be inconsistent, and integration with tracking and Streamlit will become unnecessarily difficult.

The shared standards for this stage are:
- training is launched exclusively through `.py` scripts,
- both models should be integrated as strongly as possible into the shared project interface,
- every experiment should be reproducible and clearly linked to a specific data variant,
- checkpoint and metadata saving must be standardized,
- training code should not contain machine-specific or person-specific local workarounds,
- trained models should be ready for later local use without manual rewriting of paths and formats.

This file provides the common context for all issues assigned to Stage 3.