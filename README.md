# Football Video Action Recognition

Semester project focused on football video analysis with a shared pipeline approach developed for the **Machine Learning for Data Analysis** course during Applied Mathematics Master's program.

## Project Goal
The final system should:
- accept a football match video,
- detect players, goalkeepers, referees, and the ball,
- track identities across frames,
- assign players to teams,
- generate a minimap and output video,
- expose the flow through a Streamlit app.

The experimental track compares YOLO and RF-DETR, including data-variant impact analysis.

## Project setup

### Install

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

### Prepare raw data

To start working with this repository, the required datasets must be available locally.  

Run:

```bash
python scripts/prepare_raw_data.py
```

The script will report missing datasets and print the next required step.
The most important setup details are:

- a `.env` file containing `SOCCERNET_PASSWORD` is required only when using `python scripts/prepare_raw_data.py --download-soccernet`
- the downloaded ball-extra dataset must be extracted into `data/raw/ball-extra` (<a href="https://universe.roboflow.com/footballvisionai/playersdetection-2-vfmys/dataset/2" target="_blank" rel="noopener noreferrer">link</a>)
- SoccerNet data can be downloaded into `data/raw/soccernet` with `python scripts/prepare_raw_data.py --download-soccernet`

If everything is correct, the validation output should look like this:
```bash
python scripts/prepare_raw_data.py
Note: validating SoccerNet split=train only. Use `--all` for train,test,challenge.
Verification files
- Dataset ball-extra: OK
- Dataset soccernet: OK
```

### Convert raw data to useable data format

Training data variants are expected in: `data/ready/<variant>/<split>/`

Each split must contain:
- `_annotations.coco.json` (COCO format with `images`, `annotations`, `categories`)
- image files referenced by `file_name` in COCO (path relative to the split directory)

To generate everything use:

```bash
# build all three variants
python scripts/bootstrap_base_variant.py --force

# build one selected variant
python scripts/bootstrap_base_variant.py --variant extended --force
```

Then we can extend this dataset by additional tranforming images:

```bash
python scripts/build_ready_variants.py --force
```

(or simple just one variant)
```bash
python scripts/build_ready_variants.py --variant extended-transformed --force
```

Then inspect prepared data in the app:

```bash
streamlit run app/streamlit_app.py
```

### Run model training

Train a model by running

```bash
python scripts/train.py --model yolo --dataset-variant base --profile quick
python scripts/train.py --model yolo --dataset-variant base --profile full
python scripts/train.py --model yolo --dataset-variant base --profile full --name yolo--supercomp--epochs300
python scripts/train.py --model yolo --dataset-variant base --profile quick --no-amp
python scripts/train.py --model yolo --dataset-variant base --profile quick --force-cpu
python scripts/train.py --model rf --dataset-variant base --profile rf --name rfdetr--cluster-test
```

so for local dev, propably just
```bash
python scripts/train.py --model yolo --dataset-variant base --profile quick
python scripts/train.py --model rf --dataset-variant base --profile quick
```

Notes:
- `yolo` uses the real Ultralytics backend by default.
- `rf` (`rfdetr`) uses the real Roboflow RF-DETR backend.
- Training artifacts always use `models/checkpoints/<run_name>/` and `models/metadata/<run_name>/`.
- If you pass `--name`, that value becomes the run name after sanitization. Without `--name`, an automatic name based on model, dataset variant, and timestamp is used.
- Device is selected by backend/config (`GPU` when available, otherwise `CPU` where supported).
- Use `--no-amp` to disable AMP for YOLO (recommended when debugging ROCm instability/segfaults).
- Use `--force-cpu` to force CPU execution regardless of profile config.
- To force developer fallback mock for YOLO, set `MURAWA_YOLO_MOCK=1` before running scripts.
- Static training profiles are stored in `configs/train.quick.yaml`, `configs/train.full.yaml`, and `configs/train.rf.yaml`.
- `quick` uses deterministic representative subsampling inside the requested split (`train`/`valid`) based on the config seed; it does not take the first `N` images and never consults `test`.

This creates:
- checkpoint files in `models/checkpoints/<run_name>/`,
- metadata in `models/metadata/<run_name>/`.

### Predict data

```bash
# frame-style analysis
python scripts/predict.py --model yolo --dataset-variant base --mode frame

# match-style analysis
python scripts/predict.py --model yolo --dataset-variant base --mode match --input-path /path/to/match.mp4
```

Note: `mode=match` requires a video input. If no fallback video is found in `data/ready/<variant>/test`, pass `--input-path`.

Outputs are written to:
- `outputs/predictions/<run_name>/prediction_summary.json`
- `outputs/predictions/<run_name>/*_prediction.txt`
- `outputs/predictions/<run_name>/preview/*.jpg` (mini preview frames/images)

### Use internet APP by streamlit GUI

```bash
streamlit run app/streamlit_app.py
```
