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

# install project in editable mode (enables package imports from scripts/)
pip install -e .
```

## SoccerNet raw data (COCO flow)

Download should be done via the SoccerNet Python library.

1. Set SoccerNet password in `.env`:

```bash
SOCCERNET_PASSWORD=CHANGEME
```

2. For player and ball recognition, use tracking task data (not action spotting labels):

```bash
python scripts/download_soccernet_dataset.py --mode task --task tracking-2023 --splits train --no-extract
```

To download all available splits for a task (can be very large on disk):

```bash
python scripts/download_soccernet_dataset.py --mode task --task tracking-2023 --splits train test challenge --source HuggingFace
```

Local lightweight smoke test (1-2 matches) in `games` mode below is only for quick checks and can use action-spotting labels:

```bash
python scripts/download_soccernet_dataset.py --mode games --splits train --max-games 2 --files 1_224p.mkv 2_224p.mkv Labels-v2.json
```

To download all available matches from selected splits in `games` mode, skip `--max-games` (or set it to `0`):

```bash
python scripts/download_soccernet_dataset.py --mode games --splits train test --files 1_224p.mkv 2_224p.mkv 1_player_boundingbox_maskrcnn.json 2_player_boundingbox_maskrcnn.json
```

3. Inspect extracted data and validate COCO keys (`images`, `annotations`, `categories`):

```bash
python scripts/inspect_soccernet_dataset.py --dataset-root data/raw/soccernet/soccernet_tracking_2023_coco
```

This stage keeps data as raw source only. Conversion to shared training format is handled in a later issue.

### Download parameters

Main mode and task settings:
- `--mode {task,games}`: `task` downloads task packages; `games` downloads selected files per match.
- `--task`: SoccerNet task name for `task` mode (default: `tracking-2023`).
- `--splits`: splits to download (`train`, `valid`, `test`, `challenge` depending on task availability).
- `--task-version`: optional task version (example: `224p`).
- `--source`: source backend passed to SoccerNet downloader (default: `HuggingFace`).

Games mode settings:
- `--games-task`: game list provider used by SoccerNet `getListGames` (default: `spotting`).
- `--files`: files to download for each selected match.
- `--max-games`: max matches per split; `0` means no limit (all available).
- `--randomized`: randomize match order before selecting first `--max-games`.

Output and auth settings:
- `--output-root`: local root for raw downloads (default: `data/raw/soccernet`).
- `--extract-root`: extraction target for zip archives in `task` mode.
- `--password`: override password from CLI.
- `.env`: if `--password` is not provided, the script reads `SOCCERNET_PASSWORD` / `SOCCERNET_DATA_PASSWORD`.
- `--no-extract`: keep downloaded archives without extraction in `task` mode.

For the full up-to-date list run:

```bash
python scripts/download_soccernet_dataset.py --help
```

## Quick start (Stage 1 mock flow)

This repository now contains a simple mock integration flow for two model names:
- `yolo`
- `rf` (alias for `rfdetr`)

Model implementations are organized in `src/murawa/models/`:
- `yolo.py`
- `rfdetr.py`
- `factory.py` (shared model selection)

### 1) Run mock training

```bash
python scripts/train.py --model yolo --dataset-variant base-format
python scripts/train.py --model rf --dataset-variant base-format
```

This creates:
- checkpoint files in `models/checkpoints/<run_name>/`,
- metadata in `models/metadata/<run_name>/`.

### 2) Run mock prediction from CLI

```bash
# frame-style analysis
python scripts/predict.py --model yolo --dataset-variant base-format --mode image

# match-style analysis
python scripts/predict.py --model rf --dataset-variant base-format --mode video
```

Outputs are written to:
- `outputs/predictions/<run_name>/prediction_summary.json`
- `outputs/predictions/<run_name>/*_prediction.txt`

### 3) Run Streamlit skeleton

```bash
streamlit run app/streamlit_app.py
```

Minimal Streamlit structure:
- `app/streamlit_app.py` - main entry + navigation
- `app/pages/frame_page.py` - page: `Analizuj klatkę`
- `app/pages/match_page.py` - page: `Analizuj mecz`
- `app/ui_common.py` - shared UI helpers (upload/result/common selects)

Both views support:
- optional file upload,
- model selection,
- dataset variant selection,
- running mock processing and previewing the result.
