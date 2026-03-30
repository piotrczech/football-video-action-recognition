# Stage 2. Data Work and Basic Model Flow

Stage 2 moves the project from architecture-level planning to real data and the first shared input flow for the models. This is where raw datasets begin to turn into material that can actually be used for training and experiments.

The goal of this stage is to reach a point where:
- data from different sources has been collected and organized,
- all used datasets share a common format,
- the number of frames and the basic preprocessing steps can be controlled,
- experiment-ready dataset variants are available,
- both models can use the same data loader and the same basic input flow.

This stage is critical, because without it there is no sensible way to start training either YOLO or RF-DETR. If the data is poorly prepared, even well-written model adapters will not save the quality of the experiments.

The shared standards for this stage are:
- every preprocessing step should be executed through `.py` scripts,
- processed data must follow the common format defined in Stage 1,
- all decisions related to data reduction and transformations must be reproducible,
- dataset variants must be comparable and clearly named,
- no model should require a separate manual data preparation path if that can be avoided.

This file provides the common context for all issues assigned to Stage 2.