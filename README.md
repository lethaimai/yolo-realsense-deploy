# YOLO RealSense Deploy

This folder contains the minimal files needed to run live YOLO object detection on an Intel RealSense camera.

## Included

- `models/best.pt`: trained YOLO weights
- `requirements.txt`: Python dependencies
- `how_to_run`: notes from local development

## Recommended Files

For a complete deployable repository, also add:

- `realtime_components_realsense.py`

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

If `realtime_components_realsense.py` is in this folder, run:

```bash
python realtime_components_realsense.py
```

If it is one level above this folder, run:

```bash
python ../realtime_components_realsense.py
```

## Notes

- This deploy setup is for live detection on the image frame.
- It does not require retraining.
- It does not require ROS or robot calibration unless you want robot-frame coordinates.
- The Intel RealSense camera and its system drivers must already be installed on the target machine.
