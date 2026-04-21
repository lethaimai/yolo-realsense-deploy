import os
from ultralytics import YOLO


CLASS_NAMES = {
    0: "ear_1",
    1: "ear_2",
    2: "black_surface",
    3: "slider",
    4: "hatch_handle",
    5: "probe_long_handle",
    6: "probe_short_handle",
    7: "blue_button",
    8: "red_button",
    9: "red_comp",
    10: "black_hole",
    11: "red_hole",
    12: "red_comp_button",
}


def ensure_data_yaml(path: str) -> None:
    """
    Create a data.yaml for this project if it does not exist yet.
    """
    if os.path.exists(path):
        print(f"Using existing data config: {path}")
        return

    print(f"Creating data config at {path}")

    lines = [
        "path: .",
        "train: images/train",
        "val: images/val",
        "",
        "names:",
    ]
    for idx, name in CLASS_NAMES.items():
        lines.append(f"  {idx}: {name}")

    content = "\n".join(lines) + "\n"

    # path is dataset/data.yaml, so make sure dataset exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    print("Wrote default data.yaml with 13 classes")


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_yaml = os.path.join(project_root, "dataset", "data.yaml")

    ensure_data_yaml(data_yaml)

    model_name = "yolo11n.pt"  # small and fast to start
    print(f"Loading model {model_name}")
    model = YOLO(model_name)

    print("Starting training")
    model.train(
        data=data_yaml,
        imgsz=640,
        epochs=100,
        batch=16,
        project="runs",
        name="components_13cls",
        device=0,        # set to "cpu" if you have no GPU
        workers=4,
    )

    print("Training finished. Check runs/components_13cls for results.")


if __name__ == "__main__":
    main()