import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import supervision as sv  # type: ignore
import tqdm
from scipy.optimize import linear_sum_assignment as linear_assignment
from supervision.dataset.formats.yolo import (
    detections_to_yolo_annotations,
    yolo_annotations_to_detections,
)
from supervision.utils.file import read_txt_file

from yaya_tools.classifiers.classifier_orientation import (
    DetectionsOrientation,
    get_detections_orientation,
)
from yaya_tools.helpers.np_boxes import xyxy_to_xywh

logger = logging.getLogger(__name__)


def file_annotations_to_sv(imagepath: str) -> Optional[sv.Detections]:
    """
    Load the annotations from the dataset folder

    Arguments
    ----------
    filename : str
        Filename of the annotations
    dataset_path : str
        Path to the dataset folder

    Returns
    -------
    Optional[sv.Detections]
        Annotations as sv.Detections object
    """
    # Imagepath to .txt filepath
    annotation_path = Path(imagepath).with_suffix(".txt")

    # Annotations : Load the annotations
    lines = read_txt_file(file_path=annotation_path, skip_empty=True)

    # sv.Detections : Create
    file_detections = yolo_annotations_to_detections(
        lines=lines,
        resolution_wh=(1, 1),
        with_masks=False,
        is_obb=False,
    )

    # Image name : Add the file path
    file_detections.data["filepaths"] = np.array([imagepath] * len(file_detections.xyxy))

    return file_detections


def annotations_sv_to_yolo_file(imagepath: str, annotations_sv: sv.Detections) -> None:
    """
    Save the annotations to the dataset folder

    Arguments
    ----------
    filename : str
        Filename of the annotations
    dataset_path : str
        Path to the dataset folder
    annotations_sv : sv.Detections
        Annotations as sv.Detections object
    """
    # Check : Class_id is empty
    if annotations_sv.class_id is None:
        logger.error("No annotations found!")
        return

    # Image path to .txt filepath
    annotations_path = Path(imagepath).with_suffix(".txt")

    # Lines : Convert the annotations to YOLO format
    lines: list[str] = detections_to_yolo_annotations(annotations_sv, image_shape=(1, 1, 3))

    # Annotations : Save the annotations
    with open(str(annotations_path), "w") as file:
        file.write("\n".join(lines))


def annotations_load_as_sv(
    images_annotations: dict[str, Optional[str]],
    dataset_path: str,
    filter_filenames: Optional[set[str]] = None,
) -> tuple[sv.Detections, list[str]]:
    """
    Load the annotations from the dataset folder

    Arguments
    ----------
    images_annotations : dict[str, Optional[str]]
        Dictionary of images and their annotations
    dataset_path : str
        Path to the dataset folder
    filter_filenames : set[str]
        Set of filenames to filter the images_annotations, if empty use all

    Returns
    -------
    dict[str, Optional[str]]
        Dictionary of images and their annotations
    """
    sv_detections_list: list[sv.Detections] = []
    negative_samples: list[str] = []

    # Filter : Filter only images_annotations with the filter_filenames, only if filter_filenames is not empty
    filtered_images_annotations = images_annotations
    if filter_filenames is not None:
        filtered_images_annotations = {
            image_path: annotation_path
            for image_path, annotation_path in images_annotations.items()
            if image_path in filter_filenames
        }

    for image_name, annotations_name in tqdm.tqdm(filtered_images_annotations.items(), desc="Loading annotations"):
        # Skip images without annotations
        if annotations_name is None:
            continue

        # Annotations : Load the annotations
        annotations_path = os.path.join(dataset_path, annotations_name)
        lines = read_txt_file(file_path=annotations_path, skip_empty=True)
        # sv.Detections : Create
        file_detections = yolo_annotations_to_detections(
            lines=lines,
            resolution_wh=(1, 1),
            with_masks=False,
            is_obb=False,
        )

        # Negative samples : Check if empty
        if len(file_detections.xyxy) == 0:
            negative_samples.append(image_name)
            continue

        # Annotated sample : Add the file path
        file_detections.data["filepaths"] = np.array([image_name] * len(file_detections.xyxy))
        sv_detections_list.append(file_detections)

    detections = sv.Detections.merge(sv_detections_list)
    return detections, negative_samples


def annotations_log_summary(dataset_name: str, annotations_sv: sv.Detections, negative_samples: list[str]) -> None:
    """
    Log a summary of the annotations, how many classes, how many annotations,
    For each class log count/all and %% of the total annotations, add horizontal bar
    """
    # Check : Empty
    if annotations_sv.class_id is None:
        logger.info("%s dataset is empty.", dataset_name)
        return

    # Data : parse
    unique_classes = np.unique(annotations_sv.class_id)
    total_annotations = len(annotations_sv.xyxy)
    total_files = np.unique(annotations_sv.data.get("filepaths", np.ndarray([]))).shape[0] + len(negative_samples)

    logger.info("%s has %u annotations based on %u files.", dataset_name, total_annotations, total_files)
    logger.info("%s dataset has %u classes.", dataset_name, len(unique_classes))
    for class_id in unique_classes:
        class_count = (annotations_sv.class_id == class_id).sum()
        class_ratio = class_count / total_annotations
        logger.info(
            " - Class %02u : %u/%u (%.2f%%) annotations",
            class_id,
            class_count,
            total_annotations,
            class_ratio * 100,
        )

    # Negative samples : Logging
    logger.info(
        " - Negative : %u/%u (%.2f%%) files",
        len(negative_samples),
        total_files,
        (len(negative_samples) / (total_files + 0.0001)) * 100,
    )


def annotations_filter_filenames(
    annotations: sv.Detections, negatives: list[str], filenames: list[str]
) -> tuple[sv.Detections, list[str]]:
    """Filter only the annotations from files in the filenames list"""
    # Filter : Get the indexes of the filenames
    filter_indexes = np.isin(annotations.data.get("filepaths", np.array([])), filenames)
    annotations_filtered: sv.Detections = annotations[filter_indexes]  # type: ignore

    # Filter : Negative samples
    negatives_filtered: list[str] = [neg for neg in negatives if neg in filenames]

    return annotations_filtered, negatives_filtered


def annotations_diff(
    source_annotations: sv.Detections, dest_annotations: sv.Detections
) -> tuple[sv.Detections, sv.Detections, sv.Detections]:
    """
    Find the difference between two annotations by comparing annotations IOU for only
    matched files between source and destination annotations. Then return two sv.Detections
    objects. First with source new annotations, second with source removed annotations.
    """
    # Check : Empty
    if source_annotations == sv.Detections.empty():
        logger.error("Source dataset is empty!")
        return sv.Detections.empty(), dest_annotations, sv.Detections.empty()

    # Check : Empty
    if dest_annotations == sv.Detections.empty():
        logger.error("Destination dataset is empty!")
        return source_annotations, sv.Detections.empty(), sv.Detections.empty()

    # Data : Get the unique files
    source_files = source_annotations.data.get("filepaths", np.array([]))
    source_files_unique = np.unique(source_files)
    dest_files = dest_annotations.data.get("filepaths", np.array([]))
    dest_files_unique = np.unique(dest_files)
    common_files = np.intersect1d(source_files_unique, dest_files_unique)

    # Filter : Only common files
    source_annotations_filtered: sv.Detections = source_annotations[np.isin(source_files, common_files)]  # type: ignore
    dest_annotations_filtered: sv.Detections = dest_annotations[np.isin(dest_files, common_files)]  # type: ignore

    source_filtered_files = source_annotations_filtered.data.get("filepaths", np.array([]))
    dest_filtered_files = dest_annotations_filtered.data.get("filepaths", np.array([]))

    # Diff : Find the difference
    source_new_annotations: list[sv.Detections] = []
    source_removed_annotations: list[sv.Detections] = []
    source_fitting_annotations: list[sv.Detections] = []
    for file_path in tqdm.tqdm(common_files, desc="Calculating annotations diff"):
        source_annotations_file: sv.Detections = source_annotations_filtered[source_filtered_files == file_path]  # type: ignore
        dest_annotations_file: sv.Detections = dest_annotations_filtered[dest_filtered_files == file_path]  # type: ignore

        # Check : Empty
        if source_annotations_file.xyxy.size == 0:
            source_removed_annotations.append(dest_annotations_file)
            continue

        # Check : Empty
        if dest_annotations_file.xyxy.size == 0:
            source_new_annotations.append(source_annotations_file)
            continue

        # IOU : Calculate
        iou = sv.box_iou_batch(source_annotations_file.xyxy, dest_annotations_file.xyxy)

        source_assigned_idx, dest_assigned_idx = linear_assignment(iou, maximize=True)
        # sources_not_assigned_idx = np.setdiff1d(np.arange(len(source_annotations_file.xyxy)), source_assigned_idx)
        # dest_not_assigned_idx = np.setdiff1d(np.arange(len(dest_annotations_file.xyxy)), dest_assigned_idx)

        # @TODO : Rewrite this using linear_assignment() code.
        # # Highly matching bboxes
        # matching_iou_bboxes = sources_iou_max > 0.70

        # # Source : Smaller bboxes, when IOU is high and area is smaller
        # source_matching_areas = source_annotations_file[matching_iou_bboxes].area  # type: ignore
        # dest_matching_areas = dest_annotations_file[matching_iou_bboxes].area  # type: ignore
        # source_fitting_annotations.append(source_annotations_file[source_matching_areas < dest_matching_areas])  # type: ignore

        # Source : New annotations
        sources_iou_max = iou.max(axis=1)
        source_new_annotations.append(source_annotations_file[sources_iou_max < 0.40])  # type: ignore

        # Source : Removed annotations
        dest_iou_max = iou.max(axis=0)
        source_removed_annotations.append(dest_annotations_file[dest_iou_max < 0.40])  # type: ignore

    # Merge : Return
    return (
        sv.Detections.merge(source_new_annotations),
        sv.Detections.merge(source_removed_annotations),
        sv.Detections.merge(source_fitting_annotations),
    )


def annotations_append(dataset_path: str, new_annotations: sv.Detections) -> None:
    """
    For every unique annotations file, open path annotations .txt, append
    new annotations to the end of the file and save it back.
    """
    annotations_files = new_annotations.data.get("filepaths", np.array([]))
    unique_files = np.unique(annotations_files)
    for filename in tqdm.tqdm(unique_files, desc="Appending annotations"):
        # Dataset : Load annotations from this file inside dataset, fallback to empty
        file_annotations_sv = file_annotations_to_sv(imagepath=os.path.join(dataset_path, filename))
        if file_annotations_sv is None:
            file_annotations_sv = sv.Detections.empty()

        # Filter : File new annotations
        file_new_annotations: sv.Detections = new_annotations[annotations_files == filename]  # type: ignore

        # Merge : Append new annotations
        file_annotations_sv = sv.Detections.merge(
            [
                file_annotations_sv,
                file_new_annotations,
            ]
        )

        # Save : Annotations
        annotations_sv_to_yolo_file(imagepath=os.path.join(dataset_path, filename), annotations_sv=file_annotations_sv)


def annotations_filter_large(annotations: sv.Detections, min_area: float = 0.20) -> sv.Detections:
    """Filter out large annotations from the dataset"""
    return annotations[annotations.area >= min_area]  # type: ignore


def annotations_filter_tiny(annotations: sv.Detections, max_area: float = 0.01) -> sv.Detections:
    """Filter out large annotations from the dataset"""
    return annotations[annotations.area > max_area]  # type: ignore


def annotations_filter_crowded(annotations: sv.Detections, min_objects: int = 7) -> sv.Detections:
    """Filter out large annotations from the dataset"""
    annotations_files = annotations.data.get("filepaths", np.array([], dtype=str))
    unique_files = np.unique(annotations_files)
    files_sum = np.array([np.sum(annotations_files == filename) for filename in unique_files])
    files_approved = unique_files[files_sum >= min_objects]
    return annotations[np.isin(annotations_files, files_approved)]  # type: ignore


def annotations_filter_spacious(annotations: sv.Detections, max_objects: int = 4) -> sv.Detections:
    """Filter out large annotations from the dataset"""
    annotations_files = annotations.data.get("filepaths", np.array([]))
    unique_files = np.unique(annotations_files)
    files_sum = np.array([np.sum(annotations_files == filename) for filename in unique_files])
    files_approved = unique_files[files_sum < max_objects]
    return annotations[np.isin(annotations_files, files_approved)]  # type: ignore


def annotations_filter_orientation(annotations: sv.Detections, orientation: DetectionsOrientation) -> sv.Detections:
    """Filter out horizontal annotations from the dataset"""
    annotations_files = annotations.data.get("filepaths", np.array([]))
    unique_files = np.unique(annotations_files)
    files_orientation: np.ndarray = np.array(
        [get_detections_orientation(annotations[annotations_files == filename]) for filename in unique_files], dtype=int  # type: ignore
    )
    files_approved = unique_files[files_orientation == int(orientation)]
    return annotations[np.isin(annotations_files, files_approved)]  # type: ignore


def annotations_filter_equalize(
    annotations_sv: sv.Detections,
    max_length: int = 3000,
) -> sv.Detections:
    """Filter only the annotations from files in the filenames list"""
    # Check : Empty
    if annotations_sv.class_id is None:
        return annotations_sv

    # Data : parse
    unique_classes = np.unique(annotations_sv.class_id)
    total_annotations = len(annotations_sv.xyxy)

    # Dataset stats : Calculate
    dataset_stats: dict[int | str, float] = {}
    for class_id in unique_classes:
        class_count = (annotations_sv.class_id == class_id).sum()
        class_ratio = float(class_count / total_annotations)
        dataset_stats[int(class_id)] = class_ratio

    # Inverted stats : Create as 100% - class_ratio
    # - this are probabilities of the class to be selected
    max_class_ratio = max(dataset_stats.values())
    differences = {class_id: max_class_ratio - class_ratio for class_id, class_ratio in dataset_stats.items()}

    # Normalize differences to probabilities suming to 1
    total_differences = sum(differences.values())
    class_id_probabilities = {class_id: difference / total_differences for class_id, difference in differences.items()}

    # Based on class_id probabilities, calculate the annotations probabilities, then normalize to sum to 1
    annotations_probabilities = np.array([class_id_probabilities[class_id] for class_id in annotations_sv.class_id])
    annotations_probabilities = annotations_probabilities / annotations_probabilities.sum()

    # Filter : Create equalized subset of max_length with probabilities of selected class_id
    selected_indexes = np.random.choice(
        np.arange(len(annotations_sv.xyxy)),
        size=max_length,
        p=annotations_probabilities,
        replace=True,
    )

    # Filter : Return only the selected indexes
    annotations_filtered: sv.Detections = annotations_sv[selected_indexes]  # type: ignore
    return annotations_filtered


def annotations_warnings_toosmall(annotations_sv: sv.Detections, too_small: float = 0.0010) -> sv.Detections:
    """Filter out too small annotations from the dataset"""
    annotations_too_small: sv.Detections = annotations_sv[annotations_sv.area <= too_small]  # type: ignore
    if len(annotations_too_small.xyxy) != 0:
        logger.warning("Found %u too small <%2.4f annotations", len(annotations_too_small.xyxy), too_small)

    return annotations_too_small


def annotations_warnings_xywh_not_normalized(annotations_sv: sv.Detections) -> sv.Detections:
    """Filter not normalized annotations (XYWH)"""
    xyxy: np.ndarray = annotations_sv.xyxy
    xywh: np.ndarray = xyxy_to_xywh(xyxy)

    annotations_xywh_not_normalized: sv.Detections = annotations_sv[
        (xywh[:, 0] < 0) | (xywh[:, 1] < 0) | (xywh[:, 2] > 1) | (xywh[:, 3] > 1)
    ]  # type: ignore
    if len(annotations_xywh_not_normalized.xyxy) != 0:
        logger.warning(
            "Found %u not normalized 0..1 XYWH annotations boxes!", len(annotations_xywh_not_normalized.xyxy)
        )

    return annotations_xywh_not_normalized


def annotations_warnings_xyxy_not_normalized(annotations_sv: sv.Detections) -> sv.Detections:
    """Filter not normalized annotations (XYXY)"""
    xyxy: np.ndarray = annotations_sv.xyxy
    annotations_xyxy_not_normalized: sv.Detections = annotations_sv[
        (xyxy[:, 0] < 0) | (xyxy[:, 1] < 0) | (xyxy[:, 2] > 1) | (xyxy[:, 3] > 1)
    ]  # type: ignore
    if len(annotations_xyxy_not_normalized.xyxy) != 0:
        logger.warning(
            "Found %u not normalized 0..1 XYXY annotations boxes!", len(annotations_xyxy_not_normalized.xyxy)
        )

    return annotations_xyxy_not_normalized
