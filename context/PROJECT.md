# Project Context

This project focuses on football match video analysis using machine learning and computer vision. The target system should accept a video file, detect players, goalkeepers, referees, and the ball, maintain object identity across frames, assign players to teams, and then generate a simplified minimap together with an output video available through a Streamlit interface.

In the experimental part, two detection models will be compared: YOLO and RF-DETR. In addition, the impact of different data variants will be evaluated: the base SoccerNet dataset, SoccerNet extended with an additional ball-focused dataset, and variants that include image transformations and preprocessing. Model training will be performed through `.py` scripts on the university cluster, while trained models will later be loaded locally for prediction and integration with the application.

The project is practical in nature and must be completed within a single semester. For that reason, the scope has been intentionally limited. The goal is not jersey number recognition or identification of specific players, but rather stable object detection, model comparison, integration with tracking, and generation of a clear final visualization.

The most important project assumptions are:
- a shared pipeline for both models,
- shared training and prediction scripts wherever possible,
- model training on the cluster and local use of saved artifacts,
- final integration through a Streamlit application,
- comparison of both model quality and data variant impact.

This file provides the common context for all issues in the project.