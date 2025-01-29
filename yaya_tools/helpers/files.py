"""
Helper functions for files management
"""


def image_extensions() -> list[str]:
    """
    Get the list of image extensions

    Returns
    -------
    list[str]
        List of image extensions
    """
    return [".jpg", ".png", ".jpeg", ".bmp", ".gif", ".tiff"]


def is_image_file(filename: str) -> bool:
    """
    Check if the file is an image file

    Arguments
    ----------
    filename : str
        File name

    Returns
    -------
    bool
        True if the file is an image file
    """
    return filename.lower().endswith(tuple(image_extensions()))
