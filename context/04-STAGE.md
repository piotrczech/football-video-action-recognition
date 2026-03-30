# Stage 4. Integrating the Existing Work into Streamlit

Stage 4 is the point where the project begins to function as one coherent system from the user’s perspective. After preparing the architecture, the data, and the models, the next step is to build a flow from an input video file to the final output shown in the Streamlit application.

The goal of this stage is to reach a point where:
- the user can upload a video file,
- the file is split into frames and prepared for downstream processing,
- the prediction pipeline can use the selected model on consecutive frames,
- the outputs are extended with tracking and team-related information,
- a simplified minimap and an output video are generated,
- the Streamlit application shows both the final result and selected intermediate outputs.

This stage is practical and integration-oriented. The goal is no longer to train models further, but to assemble them into a usable demonstration system. Some tasks in this stage may be developed independently of the final model and can rely on dummy data, mocks, or saved example outputs, which means that work on the UI and pipeline does not have to wait until the very end of the model stage.

The shared standards for this stage are:
- Streamlit should use one shared pipeline rather than separate flows for different models,
- some views and modules may be developed earlier on dummy data in order to speed up integration,
- solutions should be as simple and stable as possible, without adding new unnecessary scope,
- the final output should be runnable locally,
- application logic should be separated from model and data-processing logic,
- each module in this stage should have clear inputs and outputs so that it can be tested or replaced easily.

This file provides the common context for all issues assigned to Stage 4.