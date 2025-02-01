import logging
import os
from typing import Optional

import numpy as np
import supervision as sv  # type: ignore
import tqdm
from supervision.dataset.formats.yolo import yolo_annotations_to_detections
from supervision.utils.file import read_txt_file

logger = logging.getLogger(__name__)


def file_annotations_to_sv(filepath: str) -> Optional[sv.Detections]:
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
    # Annotations : Load the annotations
    lines = read_txt_file(file_path=filepath, skip_empty=True)

    # sv.Detections : Create
    file_detections = yolo_annotations_to_detections(
        lines=lines,
        resolution_wh=(1, 1),
        with_masks=False,
        is_obb=False,
    )

    return file_detections


def annotations_sv_to_yolo_file(filepath: str, annotations_sv: sv.Detections) -> None:
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

    # Annotations : Save the annotations
    with open(filepath, "w") as file:
        for index in range(len(annotations_sv.xyxy)):
            line = " ".join(
                [
                    str(annotations_sv.class_id[index]),
                    str(annotations_sv.xyxy[index][0]),
                    str(annotations_sv.xyxy[index][1]),
                    str(annotations_sv.xyxy[index][2]),
                    str(annotations_sv.xyxy[index][3]),
                ]
            )
            file.write(line + "\n")


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
        len(negative_samples) / total_files * 100,
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
) -> tuple[sv.Detections, sv.Detections]:
    """
    Find the difference between two annotations by comparing annotations IOU for only
    matched files between source and destination annotations. Then return two sv.Detections
    objects. First with source new annotations, second with source removed annotations.
    """
    # Check : Empty
    if source_annotations == sv.Detections.empty():
        logger.error("Source dataset is empty!")
        return sv.Detections.empty(), dest_annotations

    # Check : Empty
    if dest_annotations == sv.Detections.empty():
        logger.error("Destination dataset is empty!")
        return source_annotations, sv.Detections.empty()

    # Data : Get the unique files
    source_files = source_annotations.data.get("filepaths", np.array([]))
    source_files_unique = np.unique(source_files)
    dest_files = dest_annotations.data.get("filepaths", np.array([]))
    dest_files_unique = np.unique(dest_files)
    common_files = np.intersect1d(source_files_unique, dest_files_unique)

    # Filter : Only common files
    source_annotations_filtered = source_annotations[np.isin(source_files, common_files)]
    dest_annotations_filtered = dest_annotations[np.isin(dest_files, common_files)]

    source_filtered_files = source_annotations_filtered.data.get("filepaths", np.array([]))
    dest_filtered_files = dest_annotations_filtered.data.get("filepaths", np.array([]))

    # Diff : Find the difference
    source_new_annotations: list[sv.Detections] = []
    source_removed_annotations: list[sv.Detections] = []
    for file_path in tqdm.tqdm(common_files, desc="Calculating annotations diff"):
        source_annotations_file = source_annotations_filtered[source_filtered_files == file_path]
        dest_annotations_file = dest_annotations_filtered[dest_filtered_files == file_path]

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
        sources_iou_max = iou.max(axis=1)
        dest_iou_max = iou.max(axis=0)

        # Source : New annotations
        source_new_annotations.append(source_annotations_file[sources_iou_max < 0.40])

        # Source : Removed annotations
        source_removed_annotations.append(dest_annotations_file[dest_iou_max < 0.40])

    # Merge : Return
    return sv.Detections.merge(source_new_annotations), sv.Detections.merge(source_removed_annotations)


def annotations_append(dataset_path: str, new_annotations: sv.Detections) -> None:
    """
    For every unique annotations file, open path annotations .txt, append
    new annotations to the end of the file and save it back.
    """
    annotations_files = new_annotations.data.get("filepaths", np.array([]))
    unique_files = np.unique(annotations_files)
    for filename in tqdm.tqdm(unique_files, desc="Appending annotations"):
        # Dataset : Load annotations from this file inside dataset, fallback to empty
        annotations_path = os.path.join(dataset_path, filename)
        file_annotations_sv = file_annotations_to_sv(filepath=annotations_path)
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
        annotations_sv_to_yolo_file(filepath=annotations_path, annotations_sv=file_annotations_sv)
