"""
Helper functions for image processing.
"""

import logging
import os
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

import cv2  # types: ignore
import numpy as np
import supervision as sv
from tqdm import tqdm

from yaya_tools.helpers.annotations import annotations_sv_to_yolo_file
from yaya_tools.helpers.augmentations import Augumentation
from yaya_tools.helpers.hashing import get_random_sha1

logger = logging.getLogger(__name__)


def threaded_resize_image(args: tuple[str, str, str, int, int, int]) -> tuple[str, bool]:
    image_name, source_directory, target_directory, new_width, new_height, interpolation = args
    """Threded Resize image"""
    try:
        image = cv2.imread(f"{source_directory}/{image_name}")
        if image is None:
            logger.error(f"Could not read image {image_name}")
            return image_name, False
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
        cv2.imwrite(f"{target_directory}/{image_name}", resized_image)
        return image_name, True

    except Exception as e:
        logger.error(f"Error resizing image {image_name}: {e}")
        return image_name, False


def multiprocess_resize(
    source_directory: str,
    target_directory: str,
    images_names: list[str],
    new_width: int = 640,
    new_height: int = 640,
    pool_size: int = 5,
    interpolation: int = cv2.INTER_NEAREST,
) -> tuple[list[str], list[str]]:
    """
    Using multiprocessing pool resize simultaneously multiple images
    from source_directory to new files in target_directory.

    Parameters
    ----------
    source_directory : str
        Source directory with images to resize.
    target_directory : str
        Target directory to save resized images.
    images_names : list[str]
        List of image filenames to resize.

    Returns
    -------
    tuple[list[str], list[str]]
        List of successfully processed files and list of errors.
    """

    # Initialize lists
    sucess_files: list[str] = []
    failed_files: list[str] = []

    # Pool : Start a pool of workers
    with Pool(pool_size) as pool:
        results = list(
            tqdm(
                pool.imap(
                    threaded_resize_image,
                    [
                        (image_name, source_directory, target_directory, new_width, new_height, interpolation)
                        for image_name in images_names
                    ],
                ),
                total=len(images_names),
            )
        )

    # Results : Get results
    for image_name, success in results:
        if success:
            sucess_files.append(image_name)
        else:
            failed_files.append(image_name)

    return sucess_files, failed_files


def xyxy_to_xywh(xyxy: np.ndarray) -> np.ndarray:
    """
    Converts bounding box coordinates from `(x_min, y_min, x_max, y_max)`
    format to `(x, y, width, height)` format.

    Args:
        xyxy (np.ndarray): A numpy array of shape `(N, 4)` where each row
            corresponds to a bounding box in the format `(x_min, y_min, x_max, y_max)`.

    Returns:
        np.ndarray: A numpy array of shape `(N, 4)` where each row corresponds
            to a bounding box in the format `(x, y, width, height)`.

    Examples:
        ```python
        import numpy as np
        import supervision as sv

        xyxy = np.array([
            [10, 20, 40, 60],
            [15, 25, 50, 70]
        ])

        sv.xyxy_to_xywh(xyxy=xyxy)
        # array([
        #     [10, 20, 30, 40],
        #     [15, 25, 35, 45]
        # ])
        ```
    """
    xywh = xyxy.copy()
    xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]
    xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]
    return xywh


def multiprocess_augment(
    dataset_path: str,
    selected_detections: sv.Detections,
    selected_negatives: list[str],
    all_detections: sv.Detections,
    iterations: int,
    augumentation: Augumentation,
) -> None:
    """Multiprocess list of images and detections"""

    # Files : Create single files list
    all_detections_files = all_detections.data.get("filepaths", np.array([], dtype=str))
    detections_files = selected_detections.data.get("filepaths", np.array([], dtype=str))
    files_unique = np.unique(detections_files)
    files_possible = np.concatenate([files_unique, selected_negatives])

    # Random : Shuffle and limit to N
    np.random.shuffle(files_possible)
    files_to_augment = files_possible[:iterations]

    # Output path : mkdir
    Path(os.path.join(dataset_path, "generated")).mkdir(parents=True, exist_ok=True)

    # Augmentation : Augment files [iterativly first]
    for filename in tqdm(files_to_augment, desc="Augmenting images", unit="images"):
        # Image : Load
        filepath = os.path.join(dataset_path, filename)
        image = cv2.imread(filepath)
        if image is None:
            logger.error(f"Could not read image {filepath}!")
            continue

        # File annotations : Default empty list
        annotations_xyxy_class: list[float] = []
        # File annotations : Select
        file_annotations: sv.Detections = all_detections[all_detections_files == filename]  # type: ignore
        # Annotation : Extract if possible, transform to list
        # of [ [xyxy, class_id], ...] using numpy operations and reshaping
        if (file_annotations != sv.Detections.empty()) and (file_annotations.class_id is not None):
            xyxy = file_annotations.xyxy
            annotations_xyxy_class = np.concatenate([xyxy, file_annotations.class_id[:, None]], axis=1).tolist()

        # Augmentation : Apply
        new_yolo_annotations: Optional[sv.Detections] = None
        if augumentation.is_bboxes:
            augmented = augumentation.transform(image=image, bboxes=annotations_xyxy_class)
            new_albumentation_boxes = np.array(augmented["bboxes"]).reshape(-1, 5)
            new_yolo_annotations = sv.Detections(
                xyxy=new_albumentation_boxes[:, :4],
                class_id=new_albumentation_boxes[:, 4].astype(int),
            )
        else:
            augmented = augumentation.transform(image=image)
            new_yolo_annotations = file_annotations

        # Output image : Save
        output_name = f"{get_random_sha1()}.jpeg"
        output_path = os.path.join(dataset_path, "generated", output_name)
        cv2.imwrite(output_path, augmented["image"])

        # Output annotations : Save
        if new_yolo_annotations is not None:
            output_txt_path = output_path.replace(".jpeg", ".txt")
            annotations_sv_to_yolo_file(output_txt_path, new_yolo_annotations)
