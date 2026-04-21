#!/usr/bin/env python3
"""
Real time YOLO detection on Intel RealSense D435.

Features:
    - use detection model runs/components_13cls3/weights/best.pt
    - for each class keep only the highest confidence detection
    - choose between full detections or center-only display
    - draw center pixel coordinates and depth at each detection center
    - print centers per frame to terminal
    - visualize live video from the D435 color + depth stream

Press q to exit.
"""

import os
from collections import deque

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import pyrealsense2 as rs


SMOOTH_WINDOW = 5


def ask_center_only_mode() -> bool:
    """Ask whether to show only center points instead of full boxes/labels."""
    while True:
        ans = input("Show only center points? (y/n): ").strip().lower()
        if ans in {"y", "n"}:
            return ans == "y"
        print("Please type y or n.")


def ask_show_camera_xyz() -> bool:
    """Ask whether to display camera-frame XYZ coordinates."""
    while True:
        ans = input("Show camera-frame XYZ too? (y/n): ").strip().lower()
        if ans in {"y", "n"}:
            return ans == "y"
        print("Please type y or n.")


def load_camera_intrinsics(project_root: str):
    """Load camera intrinsics, preferring the OpenCV YAML if present."""
    yaml_path = os.path.join(
        project_root, "handeye_calib", "config", "realsense_info.yaml"
    )
    if os.path.exists(yaml_path):
        fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
        if fs.isOpened():
            K = fs.getNode("K").mat()
            dist = fs.getNode("D").mat()
            fs.release()
            if K is not None and dist is not None:
                return K, dist, yaml_path
        else:
            fs.release()

    npz_path = os.path.join(
        project_root, "handeye_calib", "config", "realsense_calibration.npz"
    )
    if os.path.exists(npz_path):
        calib = np.load(npz_path)
        return calib["K"], calib["dist"], npz_path

    raise FileNotFoundError(
        "No intrinsics found in handeye_calib/config/realsense_info.yaml "
        "or handeye_calib/config/realsense_calibration.npz"
    )


def find_det_weights(project_root: str) -> str:
    """
    Find detection best.pt.
    Prefer runs/components_13cls3/weights/best.pt.
    If it is not found, fall back to newest best.pt in runs.
    """
    preferred = os.path.join(
        project_root, "runs", "components_13cls3", "weights", "best.pt"
    )
    if os.path.exists(preferred):
        print(f"Using det weights: {preferred}")
        return preferred

    pattern = os.path.join(project_root, "runs")
    best_paths = []
    for root, dirs, files in os.walk(pattern):
        if "best.pt" in files:
            best_paths.append(os.path.join(root, "best.pt"))

    if not best_paths:
        raise FileNotFoundError(f"No best.pt found under {pattern}")

    best_paths.sort(key=os.path.getmtime)
    latest = best_paths[-1]
    print(f"[WARN] components_13cls3 not found, using latest: {latest}")
    return latest


def filter_boxes_per_class(result):
    """
    From one Ultralytics detection result, keep only the highest
    confidence box per class.

    Returns:
        xyxy_keep: [M, 4]
        cls_keep:  [M]
        conf_keep: [M]
    """
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None, None, None

    cls = boxes.cls.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy()
    xyxy = boxes.xyxy.detach().cpu().numpy()

    keep_indices = []
    for c in np.unique(cls.astype(int)):
        idxs = np.where(cls == c)[0]
        best_idx = idxs[conf[idxs].argmax()]
        keep_indices.append(best_idx)

    keep_indices = np.array(keep_indices, dtype=int)
    return xyxy[keep_indices], cls[keep_indices], conf[keep_indices]


def get_center_depth_mm(depth_img, u, v, depth_scale, patch_radius=2):
    """
    Read a robust depth estimate at pixel (u, v) from a small local patch.
    Returns (depth_mm, valid_count) where depth_mm is None if no valid depth exists.
    """
    h, w = depth_img.shape[:2]
    u0 = max(0, u - patch_radius)
    u1 = min(w, u + patch_radius + 1)
    v0 = max(0, v - patch_radius)
    v1 = min(h, v + patch_radius + 1)

    patch = depth_img[v0:v1, u0:u1]
    valid = patch[patch > 0]
    if valid.size == 0:
        return None, 0

    depth_raw = float(np.median(valid))
    depth_mm = depth_raw * depth_scale * 1000.0
    return depth_mm, int(valid.size)


def pixel_depth_to_cam(u, v, depth_mm, K):
    """Project pixel and depth into camera-frame coordinates in mm."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X = (u - cx) * depth_mm / fx
    Y = (v - cy) * depth_mm / fy
    Z = depth_mm
    return np.array([X, Y, Z], dtype=float)


def update_smoothed_measurement(history, class_id, u, v, depth_mm, cam_point):
    """Update per-class history and return median-smoothed values."""
    if class_id not in history:
        history[class_id] = {
            "u": deque(maxlen=SMOOTH_WINDOW),
            "v": deque(maxlen=SMOOTH_WINDOW),
            "depth": deque(maxlen=SMOOTH_WINDOW),
            "x": deque(maxlen=SMOOTH_WINDOW),
            "y": deque(maxlen=SMOOTH_WINDOW),
            "z": deque(maxlen=SMOOTH_WINDOW),
        }

    item = history[class_id]
    item["u"].append(float(u))
    item["v"].append(float(v))

    if depth_mm is not None:
        item["depth"].append(float(depth_mm))
    if cam_point is not None:
        item["x"].append(float(cam_point[0]))
        item["y"].append(float(cam_point[1]))
        item["z"].append(float(cam_point[2]))

    smoothed_u = int(round(np.median(item["u"])))
    smoothed_v = int(round(np.median(item["v"])))

    smoothed_depth = None
    if item["depth"]:
        smoothed_depth = float(np.median(item["depth"]))

    smoothed_cam = None
    if item["x"] and item["y"] and item["z"]:
        smoothed_cam = np.array(
            [
                np.median(item["x"]),
                np.median(item["y"]),
                np.median(item["z"]),
            ],
            dtype=float,
        )

    return smoothed_u, smoothed_v, smoothed_depth, smoothed_cam


def draw_boxes_and_centers(
    img,
    xyxy,
    cls,
    conf,
    names,
    center_info=None,
    show_labels=True,
    center_only=False,
    show_camera_xyz=False,
):
    """
    Draw filtered boxes and center points on the image.
    Optionally show the class name and confidence.
    """
    if xyxy is None:
        return img

    img_out = img.copy()

    if center_info is None:
        center_info = [None] * len(xyxy)

    for box, c, p, info in zip(xyxy, cls, conf, center_info):
        x1, y1, x2, y2 = box.astype(int)
        c_int = int(c)

        # center in pixels
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        if info is not None and info.get("center_px") is not None:
            cx, cy = info["center_px"]

        # unique color per class
        color = (
            int(37 * (c_int + 1) % 255),
            int(17 * (c_int + 1) % 255),
            int(73 * (c_int + 1) % 255),
        )

        # box
        if not center_only:
            cv2.rectangle(img_out, (x1, y1), (x2, y2), color, 2)

        # center point
        cv2.circle(img_out, (cx, cy), 4, color, thickness=-1)

        coord_text = f"({cx}, {cy}, ?)"
        if info is not None:
            depth_label = info.get("depth_label", "?")
            coord_text = f"({cx}, {cy}, {depth_label})"

        coord_size, coord_baseline = cv2.getTextSize(
            coord_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        )
        coord_tx = min(max(0, cx + 6), max(0, img_out.shape[1] - coord_size[0]))
        coord_ty = min(
            max(coord_size[1] + coord_baseline, cy - 6),
            img_out.shape[0] - 1,
        )
        cv2.rectangle(
            img_out,
            (coord_tx, coord_ty - coord_size[1] - coord_baseline),
            (coord_tx + coord_size[0], coord_ty + coord_baseline),
            color,
            thickness=-1,
        )
        cv2.putText(
            img_out,
            coord_text,
            (coord_tx, coord_ty - coord_baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            lineType=cv2.LINE_AA,
        )

        if show_camera_xyz and info is not None and info.get("cam_text"):
            cam_text = info["cam_text"]
            cam_size, cam_baseline = cv2.getTextSize(
                cam_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
            )
            cam_tx = coord_tx
            cam_top = min(
                img_out.shape[0] - cam_size[1] - cam_baseline,
                coord_ty + cam_baseline + 4,
            )
            cv2.rectangle(
                img_out,
                (cam_tx, cam_top),
                (cam_tx + cam_size[0], cam_top + cam_size[1] + cam_baseline),
                color,
                thickness=-1,
            )
            cv2.putText(
                img_out,
                cam_text,
                (cam_tx, cam_top + cam_size[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                lineType=cv2.LINE_AA,
            )

        if show_labels and not center_only:
            label = f"{names[c_int]} {p:.2f}"
            text_size, baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            tx, ty = x1, max(0, y1 - 4)

            # label background
            cv2.rectangle(
                img_out,
                (tx, ty - text_size[1] - baseline),
                (tx + text_size[0], ty + baseline),
                color,
                thickness=-1,
            )

            # label text
            cv2.putText(
                img_out,
                label,
                (tx, ty - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                lineType=cv2.LINE_AA,
            )

    return img_out


def setup_realsense():
    """
    Configure RealSense pipeline for color + depth stream.
    Resolution and fps can be adjusted if needed.
    """
    pipeline = rs.pipeline()
    config = rs.config()

    # color + depth streams 640x480 at 30 fps
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    print("Starting RealSense pipeline with depth...")
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    align = rs.align(rs.stream.color)
    print(f"Depth scale: {depth_scale} m per unit")
    return pipeline, depth_scale, align


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))

    # find model weights
    weights_path = find_det_weights(project_root)

    # load YOLO model
    print(f"Loading model from {weights_path}")
    model = YOLO(weights_path)

    center_only = ask_center_only_mode()
    show_camera_xyz = ask_show_camera_xyz()

    K = None
    if show_camera_xyz:
        try:
            K, _dist, calib_path = load_camera_intrinsics(project_root)
            print(f"Loaded intrinsics from {calib_path}")
        except FileNotFoundError as exc:
            print(f"[WARN] {exc}")
            print("[WARN] Camera-frame XYZ overlay disabled.")
            show_camera_xyz = False

    # choose device
    if torch.cuda.is_available():
        device = 0
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = "cpu"
        print("CUDA not available, using CPU")

    # RealSense
    try:
        pipeline, depth_scale, align = setup_realsense()
    except Exception as e:
        print("[ERROR] Could not start RealSense pipeline:", e)
        return

    print("Real time detection started. Press q to quit.\n")

    measurement_history = {}

    try:
        while True:
            # wait for next frame
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            # frame as numpy array (BGR)
            frame = np.asanyarray(color_frame.get_data())
            depth_img = np.asanyarray(depth_frame.get_data())
            h, w = frame.shape[:2]

            # run YOLO detection on this frame
            results = model.predict(
                source=frame,
                imgsz=640,
                conf=0.25,
                device=device,
                verbose=False,
            )

            result = results[0]
            xyxy, cls, conf = filter_boxes_per_class(result)

            centers_text = []
            center_info = []

            if xyxy is not None:
                for box, c, p in zip(xyxy, cls, conf):
                    x1, y1, x2, y2 = box
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    cx_norm = cx / w
                    cy_norm = cy / h
                    c_int = int(c)
                    class_name = model.names[c_int]
                    u = int(np.clip(round(cx), 0, w - 1))
                    v = int(np.clip(round(cy), 0, h - 1))
                    depth_mm, valid_count = get_center_depth_mm(
                        depth_img, u, v, depth_scale, patch_radius=2
                    )

                    if depth_mm is None:
                        depth_text = "depth=invalid"
                        depth_label = "?"
                        cam_point = None
                        cam_text = None
                    else:
                        depth_text = f"depth={depth_mm:.1f}mm"
                        depth_label = f"{int(round(depth_mm))}"
                        cam_point = None if K is None else pixel_depth_to_cam(u, v, depth_mm, K)
                        cam_text = None
                        if cam_point is not None:
                            cam_text = (
                                f"cam=({int(round(cam_point[0]))},"
                                f"{int(round(cam_point[1]))},"
                                f"{int(round(cam_point[2]))})"
                            )

                    smooth_u, smooth_v, smooth_depth_mm, smooth_cam_point = update_smoothed_measurement(
                        measurement_history, c_int, u, v, depth_mm, cam_point
                    )

                    if smooth_depth_mm is None:
                        depth_text = "depth=invalid"
                        depth_label = "?"
                    else:
                        depth_text = f"depth={smooth_depth_mm:.1f}mm"
                        depth_label = f"{int(round(smooth_depth_mm))}"

                    cam_text = None
                    if smooth_cam_point is not None:
                        cam_text = (
                            f"cam=({int(round(smooth_cam_point[0]))},"
                            f"{int(round(smooth_cam_point[1]))},"
                            f"{int(round(smooth_cam_point[2]))})"
                        )

                    center_info.append(
                        {
                            "center_px": (smooth_u, smooth_v),
                            "depth_mm": None if smooth_depth_mm is None else float(smooth_depth_mm),
                            "depth_text": depth_text,
                            "depth_label": depth_label,
                            "depth_valid_count": valid_count,
                            "cam_point": None if smooth_cam_point is None else smooth_cam_point.tolist(),
                            "cam_text": cam_text,
                        }
                    )

                    center_line = (
                        f"{class_name}: center=({smooth_u}, {smooth_v}) "
                        f"norm=({cx_norm:.3f}, {cy_norm:.3f}) {depth_text}"
                    )
                    if cam_text is not None:
                        center_line += f" {cam_text}"
                    center_line += f" conf={p:.2f}"
                    centers_text.append(center_line)

            if centers_text:
                print("\n".join(centers_text))
                print("-" * 40)

            # draw visualization
            vis = draw_boxes_and_centers(
                frame,
                xyxy,
                cls,
                conf,
                model.names,
                center_info=center_info,
                show_labels=True,
                center_only=center_only,
                show_camera_xyz=show_camera_xyz,
            )

            cv2.imshow("RealSense YOLO components", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Stopped RealSense pipeline.")


if __name__ == "__main__":
    main()
