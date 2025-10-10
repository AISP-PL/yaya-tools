import logging
from pathlib import Path

import cv2
import supervision as sv
from tqdm import tqdm


def convert_rgba_to_rgb(dataset_path: str):
    """Converts all RGBA images in a dataset to RGB format.

    This function scans a dataset directory for images (JPG, JPEG, PNG) and converts
    any PNG images with an alpha (transparency) channel to standard 3-channel RGB format.
    The converted images are saved back into the same dataset directory.

    Args:
        dataset_path (str): Path to the dataset directory containing image files.
    """
    # list image paths
    image_paths = sv.list_files_with_extensions(
        directory=dataset_path, extensions=["jpg", "jpeg", "png"]
    )

    # dataset cannot contain images with alpha channel -> convert RGBA to RGB
    with sv.ImageSink(target_dir_path=dataset_path) as sink:
        for image_path in tqdm(
            image_paths, desc="Converting RGBA to RGB", unit="image"
        ):
            image_path = Path(image_path)

            # only png images can have alpha channel, skip the rest
            if not image_path.suffix == ".png":
                continue

            image = cv2.imread(filename=str(image_path), flags=cv2.IMREAD_UNCHANGED)
            if image is None:
                logging.warning(f"Skipping unreadable: {image_path}")
                continue

            # convert RGBA to RGB
            if image.ndim == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

            sink.save_image(image=image, image_name=image_path.name)


def convert_dataset_to_coco_format(
    dataset_path: str, data_yaml_path: str
) -> sv.DetectionDataset:
    """Converts a YOLO-formatted dataset to COCO format.

    Args:
        dataset_path (str): Path to the directory containing YOLO-format images and labels.
        data_yaml_path (str): Path to the YOLO data YAML configuration file.

    Returns:
        sv.DetectionDataset: A `supervision` DetectionDataset object in COCO format.
    """
    return sv.DetectionDataset.from_yolo(
        images_directory_path=dataset_path,
        annotations_directory_path=dataset_path,
        data_yaml_path=data_yaml_path,
    )


def train_valid_test_split(
    dataset: sv.DetectionDataset,
) -> tuple[sv.DetectionDataset, sv.DetectionDataset, sv.DetectionDataset]:
    """Splits a dataset into training [80%], validation [10%], and test [10%] datasets.

    Args:
        dataset (sv.DetectionDataset): A `supervision` DetectionDataset object.

    Returns:
        tuple[sv.DetectionDataset, sv.DetectionDataset, sv.DetectionDataset]:
        A tuple containing training validation and test datasets
    """
    coco_dataset_train, coco_dataset_valid_test = dataset.split(
        split_ratio=0.8, shuffle=True
    )
    coco_dataset_valid, coco_dataset_test = coco_dataset_valid_test.split(
        split_ratio=0.5, shuffle=True
    )
    return coco_dataset_train, coco_dataset_valid, coco_dataset_test


def save_dataset_coco_format(
    dataset: sv.DetectionDataset, images_path: str, annotations_path: str
) -> None:
    """Saves a `supervision` DetectionDataset in COCO format.

    Args:
        dataset (sv.DetectionDataset): The dataset to save.
        images_path (str): Path where the image files should be saved.
        annotations_path (str): Path to save the COCO annotation JSON file.
    """
    dataset.as_coco(
        images_directory_path=images_path, annotations_path=annotations_path
    )
