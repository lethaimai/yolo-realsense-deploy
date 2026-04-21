# YOLO RealSense Deploy

This repository is for two tasks:

- live YOLO object detection on an Intel RealSense camera
- training a YOLO detector on the same component classes

## Included Files

- `realtime_components_realsense.py`: live RealSense inference
- `train_components_yolo.py`: YOLO training script for standard bounding boxes
- `models/best.pt`: trained YOLO weights
- `requirements.txt`: Python dependencies
- `how_to_run`: local notes from development

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

## RealSense Requirement

The target machine must already have:

- an Intel RealSense camera connected by USB
- Intel RealSense system drivers / SDK (`librealsense`) installed

Useful checks:

```bash
realsense-viewer
```

or

```bash
rs-enumerate-devices
```

## Run Live Detection

Run:

```bash
python realtime_components_realsense.py
```

This is the simplest mode. It shows detected objects directly on the image frame.

## Dataset Preparation For Training

The training script expects a YOLO dataset inside a folder named `dataset` with this structure:

```text
dataset/
  data.yaml
  images/
    train/
    val/
  labels/
    train/
    val/
```

Each image in `images/train` or `images/val` must have a matching YOLO label file with the same base name in the corresponding labels folder.

Example:

```text
dataset/images/train/img_001.png
dataset/labels/train/img_001.txt
```

YOLO label format for each line:

```text
class_id x_center y_center width height
```

These values must be normalized to the image size and usually stay between `0` and `1`.

## Classes

The training script uses these 13 classes:

```text
0  ear_1
1  ear_2
2  black_surface
3  slider
4  hatch_handle
5  probe_long_handle
6  probe_short_handle
7  blue_button
8  red_button
9  red_comp
10 black_hole
11 red_hole
12 red_comp_button
```

## data.yaml

If `dataset/data.yaml` does not exist, `train_components_yolo.py` creates it automatically.

Expected content:

```yaml
path: .
train: images/train
val: images/val

names:
  0: ear_1
  1: ear_2
  2: black_surface
  3: slider
  4: hatch_handle
  5: probe_long_handle
  6: probe_short_handle
  7: blue_button
  8: red_button
  9: red_comp
  10: black_hole
  11: red_hole
  12: red_comp_button
```

## Train A Model

After preparing the dataset, run:

```bash
python train_components_yolo.py
```

The current training script uses:

- image size `640`
- `100` epochs
- batch size `16`
- base model `yolo11n.pt`

Training results are saved under:

```text
runs/components_13cls/
```

The trained weights will usually be here:

```text
runs/components_13cls/weights/best.pt
```

## Notes

- `realtime_components_realsense.py` is for standard bounding-box detection.
- `train_components_yolo.py` is also for standard bounding-box detection.
- ROS and robot calibration are not required for simple live image detections.
- If you want robot-frame coordinates, that is a separate setup step and needs calibration on the target robot.
