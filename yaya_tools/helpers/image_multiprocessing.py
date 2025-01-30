"""
Helper functions for image processing.
"""

import logging
from multiprocessing import Pool

import cv2  # types: ignore
from tqdm import tqdm

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
