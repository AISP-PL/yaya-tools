import yaml


def create_yolo_yaml_str(
    train_path: str, val_path: str, test_path: str, classes: list[str], annotation_format: str = "coco"
) -> str:
    """Creates a YOLO dataset YAML as string."""
    data = {
        "train": train_path,
        "val": val_path,
        "test": test_path,
        "nc": len(classes),
        "names": {i: name for i, name in enumerate(classes)},
        "annotation_format": annotation_format,
    }
    return yaml.dump(data, default_flow_style=False, sort_keys=False)
