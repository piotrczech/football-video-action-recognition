# Additional dataset for ball detection

## Source
Roboflow dataset: PlayersDetection 2

## Purpose
This dataset is used as an additional source focused on improving the hardest class in the project, especially ball detection.

## Local raw path
`data/raw/ball_extra/playersdetection_v2_coco`

## Download format
COCO

## Current structure
The dataset is stored locally with three splits:
- `train/`
- `valid/`
- `test/`

Each split contains images and COCO annotations in JSON format.

## Validation
The dataset structure is checked with:
`scripts/inspect_extra_dataset.py`

The inspection verifies:
- split presence,
- image counts,
- annotation file presence,
- annotation counts,
- category list.

## Notes
This dataset is kept as a raw source for now.
It is not yet converted to the common training format.
Conversion will be handled in the next data-format issue.