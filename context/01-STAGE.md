# Stage 1. Initial Setup and Shared Interface

Stage 1 establishes the technical and organizational foundation for all further work. This is not yet the model-building stage. Instead, it is the stage where the team defines rules, interfaces, and integration points between parallel workstreams.

The goal of this stage is to reach a point where:
- it is clear what exactly will be compared,
- it is clear what the data should look like after preprocessing,
- it is clear how training and prediction are invoked,
- it is clear what a trained model artifact looks like,
- it is clear how a trained model will later be connected to the pipeline and Streamlit.

A well-executed Stage 1 unlocks all later stages. A poorly executed Stage 1 leads to duplicated work, inconsistent data formats, inconsistent model execution flows, and painful manual integration later on.

The shared standards for this stage are:
- the project is developed around `.py` scripts rather than notebooks,
- solutions should be shared between YOLO and RF-DETR wherever possible,
- all technical decisions should be written down in the repository,
- interfaces should be simple and usable both on the cluster and locally,
- data, models, and outputs must have a predictable directory and naming structure,
- everything created at this stage should make later integration with tracking, the minimap, and Streamlit easier.

This file provides the common context for all issues assigned to Stage 1.