# Football Video Action Recognition

Semester project focused on football video analysis with a shared pipeline approach developed for the **Machine Learning for Data Analysis** course during Applied Mathematics Master’s program.

## Project Goal
The final system should:
- accept a football match video,
- detect players, goalkeepers, referees, and the ball,
- track identities across frames,
- assign players to teams,
- generate a minimap and output video,
- expose the flow through a Streamlit app.

The experimental track compares YOLO and RF-DETR, including data-variant impact analysis.

## Current Status (Stages 1-5)
- Stage 1: foundation, interfaces, and standards ([context](context/01-STAGE.md))
- Stage 2: data preparation and shared input flow ([context](context/02-STAGE.md))
- Stage 3: model training and artifact standardization ([context](context/03-STAGE.md))
- Stage 4: Streamlit and end-to-end integration ([context](context/04-STAGE.md))
- Stage 5: final reports and project wrap-up ([context](context/05-STAGE.md))

## Environment setup (pyenv)

Below is an example of how to configure the environment using `pyenv` and a virtual environment:

```bash
# install a chosen Python version (e.g. 3.13.x)
pyenv install 3.13.2

# create a virtual environment for this project
pyenv virtualenv 3.13.2 football

# associate the virtual environment with this project directory
pyenv local football 
pyenv activate football 

# upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```