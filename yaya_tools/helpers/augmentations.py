import argparse
import logging
from typing import NamedTuple, Optional

import albumentations as A  # type: ignore
import cv2

from yaya_tools.augmentations.prop import PropsAugmentation


class Augumentation(NamedTuple):
    """This class is used to store the transformation and the type of the transformation"""

    transform: A.Compose
    is_bboxes: bool


# Shape : Albumentations transform
def transform_crop_make(width: int = 640) -> Augumentation:
    """Create crop transformation."""

    # Width and height must be divisible by 32
    # - min width is 640
    # - height almost 16/9 ratio to width
    width = max(640, width)
    height = int(width * 9 / 16)
    height = height - (height % 32)
    width = width - (width % 32)

    transformation = A.Compose(
        [A.RandomCrop(width=width, height=height, p=0.99)],
        bbox_params=A.BboxParams(format="albumentations", min_area=100, min_visibility=0.3),
    )
    return Augumentation(transform=transformation, is_bboxes=True)


def transform_rotate_make(degrees: int = 30) -> Augumentation:
    """Create rotate transformation."""

    transformation = A.Compose(
        [
            A.Rotate(
                limit=(degrees, degrees),
                border_mode=cv2.BORDER_CONSTANT,
                rotate_method="ellipse",
                p=0.99,
            )
        ],
        bbox_params=A.BboxParams(format="albumentations", min_area=100, min_visibility=0.3),
    )
    return Augumentation(transform=transformation, is_bboxes=True)


def transform_randrotate_make(degrees: int = 30) -> Augumentation:
    """Create rotate transformation."""

    transformation = A.Compose(
        [
            A.Rotate(
                limit=degrees,
                border_mode=cv2.BORDER_CONSTANT,
                rotate_method="ellipse",
                p=0.99,
            )
        ],
        bbox_params=A.BboxParams(format="albumentations", min_area=100, min_visibility=0.3),
    )
    return Augumentation(transform=transformation, is_bboxes=True)


def transform_flip_horizontal_make() -> Augumentation:
    """Create flip transformation."""
    transformation = A.Compose(
        [A.HorizontalFlip(p=0.99, always_apply=True)],
        bbox_params=A.BboxParams(format="albumentations", min_area=100, min_visibility=0.3),
    )
    return Augumentation(transform=transformation, is_bboxes=True)


def transform_flip_vertical_make() -> Augumentation:
    """Create flip transformation."""
    transformation = A.Compose(
        [A.VerticalFlip(p=0.99, always_apply=True)],
        bbox_params=A.BboxParams(format="albumentations", min_area=100, min_visibility=0.3),
    )
    return Augumentation(transform=transformation, is_bboxes=True)


def transform_flip_make() -> Augumentation:
    """Create flip transformation."""

    transformation = A.Compose(
        [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)],
        bbox_params=A.BboxParams(format="albumentations", min_area=100, min_visibility=0.3),
    )
    return Augumentation(transform=transformation, is_bboxes=True)


def transform_compression_make() -> Augumentation:
    """Create compression transformation."""
    transformation = A.Compose(
        [A.ImageCompression(quality_lower=10, quality_upper=15, p=0.99)],
        bbox_params=A.BboxParams(format="albumentations", min_area=100, min_visibility=0.3),
    )
    return Augumentation(transform=transformation, is_bboxes=False)


def transform_degrade_make() -> Augumentation:
    """Create compression transformation."""
    transformation = A.Compose(
        [
            A.Downscale(scale_min=0.25, scale_max=0.45, p=0.999),
        ],
        bbox_params=A.BboxParams(format="albumentations", min_area=100, min_visibility=0.3),
    )
    return Augumentation(transform=transformation, is_bboxes=False)


def transform_clahe_make() -> Augumentation:
    """Create CLAHE transformation."""
    transformation = A.Compose(
        [A.CLAHE(p=0.999, always_apply=True)],
        bbox_params=A.BboxParams(format="albumentations", min_area=100, min_visibility=0.3),
    )
    return Augumentation(transform=transformation, is_bboxes=False)


def transform_equalize_make() -> Augumentation:
    """Create equalize transformation."""
    transformation = A.Compose(
        [A.Equalize(p=0.999, always_apply=True)],
        bbox_params=A.BboxParams(format="albumentations", min_area=100, min_visibility=0.3),
    )
    return Augumentation(transform=transformation, is_bboxes=False)


def transform_sharpen_make() -> Augumentation:
    """Create sharpen transformation."""
    transformation = A.Compose(
        [A.Sharpen(alpha=(0.25, 0.5), lightness=(0.7, 1.1), p=0.999, always_apply=True)],
        bbox_params=A.BboxParams(format="albumentations", min_area=100, min_visibility=0.3),
    )
    return Augumentation(transform=transformation, is_bboxes=False)


def transform_downsize_padding_make() -> Augumentation:
    """Downsize with padding using ShiftScaleRotate"""
    transformation = A.Compose(
        [
            A.ShiftScaleRotate(
                shift_limit=0.0,
                rotate_limit=0,
                scale_limit=0.2,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.999,
                always_apply=True,
            )
        ],
        bbox_params=A.BboxParams(format="albumentations", min_area=100, min_visibility=0.3),
    )
    return Augumentation(transform=transformation, is_bboxes=True)


def transform_blur_delicate_make() -> Augumentation:
    """Create blur transformation."""
    transform = A.Compose(
        [A.Blur(blur_limit=3, p=0.99)],
        bbox_params=A.BboxParams(format="albumentations", min_area=100, min_visibility=0.3),
    )
    return Augumentation(transform=transform, is_bboxes=False)


def transform_blur_make() -> Augumentation:
    """Create blur transformation."""
    transformation = A.Compose(
        [A.Blur(blur_limit=7, p=0.99)],
        bbox_params=A.BboxParams(format="albumentations", min_area=100, min_visibility=0.3),
    )
    return Augumentation(transform=transformation, is_bboxes=False)


def transform_median_blur_make() -> Augumentation:
    """Create median blur transformation."""
    transformation = A.Compose(
        [A.MedianBlur(blur_limit=7, p=0.99)],
        bbox_params=A.BboxParams(format="albumentations", min_area=100, min_visibility=0.3),
    )
    return Augumentation(transform=transformation, is_bboxes=False)


def transform_colorshift_make() -> Augumentation:
    """Create random brighten transformation."""
    transformation = A.Compose(
        [
            A.ColorJitter(
                brightness=(1, 1),
                contrast=(1, 1),
                saturation=(1, 1),
                hue=(-0.25, 0.25),
                always_apply=True,
                p=0.999,
            )
        ],
        bbox_params=A.BboxParams(format="albumentations", min_area=100, min_visibility=0.3),
    )
    return Augumentation(transform=transformation, is_bboxes=False)


def transform_brighten_make() -> Augumentation:
    """Create random brighten transformation."""
    transformation = A.Compose(
        [
            A.RandomBrightnessContrast(
                brightness_limit=(0.0, 0.4),
                contrast_limit=(-0.35, 0.35),
                always_apply=True,
                p=0.999,
            )
        ],
        bbox_params=A.BboxParams(format="albumentations", min_area=100, min_visibility=0.3),
    )
    return Augumentation(transform=transformation, is_bboxes=False)


def transform_darken_make() -> Augumentation:
    """Create random darken transformation."""
    transformation = A.Compose(
        [
            A.RandomBrightnessContrast(
                brightness_limit=(-0.4, 0.0),
                contrast_limit=(-0.35, 0.35),
                always_apply=True,
                p=0.999,
            )
        ],
        bbox_params=A.BboxParams(format="albumentations", min_area=100, min_visibility=0.3),
    )
    return Augumentation(transform=transformation, is_bboxes=False)


def transform_snow_make() -> Augumentation:
    """Create snow transformation."""
    transformation = A.Compose(
        [A.RandomSnow(p=0.999)],
        bbox_params=A.BboxParams(format="albumentations", min_area=100, min_visibility=0.3),
    )
    return Augumentation(transform=transformation, is_bboxes=False)


def transform_rain_make() -> Augumentation:
    """Create rain transformation."""
    transformation = A.Compose(
        [A.RandomRain(drop_length=10, blur_value=4, p=0.999)],
        bbox_params=A.BboxParams(format="albumentations", min_area=100, min_visibility=0.3),
    )
    return Augumentation(transform=transformation, is_bboxes=False)


def transform_spatter_big_make() -> Augumentation:
    """Create spatter transformation."""
    transformation = A.Compose(
        [A.Spatter(p=0.999, gauss_sigma=4.5)],
        bbox_params=A.BboxParams(format="albumentations", min_area=100, min_visibility=0.3),
    )
    return Augumentation(transform=transformation, is_bboxes=False)


def transform_spatter_make() -> Augumentation:
    """Create spatter transformation."""
    transformation = A.Compose(
        [A.Spatter(p=0.999)],
        bbox_params=A.BboxParams(format="albumentations", min_area=100, min_visibility=0.3),
    )
    return Augumentation(transform=transformation, is_bboxes=False)


def transform_spatter_small_make() -> Augumentation:
    """Create spatter transformation."""
    transformation = A.Compose(
        [A.Spatter(p=0.999, gauss_sigma=0.5)],
        bbox_params=A.BboxParams(format="albumentations", min_area=100, min_visibility=0.3),
    )
    return Augumentation(transform=transformation, is_bboxes=False)


def transform_blackboxing_make(size: int = 50) -> Augumentation:
    """Blackboxing parts of image."""
    logging.warning("Blackboxing not supports bbox target!")
    transformation = A.Compose(
        [
            A.CoarseDropout(
                min_holes=7,
                max_holes=10,
                min_height=size,
                max_height=size,
                min_width=size,
                max_width=size,
                always_apply=True,
                p=0.999,
            )
        ],
    )
    return Augumentation(transform=transformation, is_bboxes=False)


def transform_isonoise_make() -> Augumentation:
    """Create isonoise transformation."""
    transformation = A.Compose(
        [
            A.ISONoise(
                color_shift=(0.04, 0.30),
                intensity=(0.33, 1.25),
                always_apply=True,
                p=0.999,
            )
        ],
        bbox_params=A.BboxParams(format="albumentations", min_area=100, min_visibility=0.3),
    )
    return Augumentation(transform=transformation, is_bboxes=False)


def transform_gaussnoise_make() -> Augumentation:
    """Create gauss noise transformation."""
    transformation = A.Compose(
        [
            A.GaussNoise(
                var_limit=(10.0, 70.0),
                mean=0,
                always_apply=True,
                p=0.999,
            )
        ],
        bbox_params=A.BboxParams(format="albumentations", min_area=100, min_visibility=0.3),
    )
    return Augumentation(transform=transformation, is_bboxes=False)


def transform_multinoise_make() -> Augumentation:
    """Create multiplicative noise transformation."""
    transformation = A.Compose(
        [
            A.MultiplicativeNoise(
                multiplier=(0.85, 1.15),
                per_channel=True,
                elementwise=True,
                always_apply=True,
                p=0.999,
            )
        ],
        bbox_params=A.BboxParams(format="albumentations", min_area=100, min_visibility=0.3),
    )
    return Augumentation(transform=transformation, is_bboxes=False)


def transform_fog_make() -> Augumentation:
    """Create fog transformation."""
    transformation = A.Compose(
        [A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.5, alpha_coef=0.5, p=0.999)],
        bbox_params=A.BboxParams(format="albumentations", min_area=100, min_visibility=0.3),
    )
    return Augumentation(transform=transformation, is_bboxes=True)


def transform_sunflare_make() -> Augumentation:
    """Create sunflare transformation."""
    transformation = A.Compose(
        [
            A.RandomSunFlare(
                src_radius=260,
                num_flare_circles_lower=2,
                num_flare_circles_upper=6,
                p=0.999,
            )
        ],
        bbox_params=A.BboxParams(format="albumentations", min_area=100, min_visibility=0.3),
    )
    return Augumentation(transform=transformation, is_bboxes=True)


def transform_grayscale_make() -> Augumentation:
    """Create grayscale transformation."""
    transformation = A.Compose(
        [A.ToGray(p=0.999)],
        bbox_params=A.BboxParams(format="albumentations", min_area=100, min_visibility=0.3),
    )
    return Augumentation(transform=transformation, is_bboxes=False)


def transform_sepia_make() -> Augumentation:
    """Create sepia transformation."""
    transformation = A.Compose(
        [A.ToSepia(p=0.999)],
        bbox_params=A.BboxParams(format="albumentations", min_area=100, min_visibility=0.3),
    )
    return Augumentation(transform=transformation, is_bboxes=False)


def transform_multi_props_make() -> Augumentation:
    """Add random props from yaya_tools/data/props"""
    transformation = A.Compose(
        [
            PropsAugmentation(
                prop_dir="yaya_tools/data/props",
                n_props=(1, 4),
                opacity=(0.80, 1.0),  # <- class parameter
                autoscale=True,  # <- class parameter
                scale_range=(0.25, 0.5),  # fraction of min(H, W) for prop's longer side
                rotate_limit=15,
                flip_prob=0.5,
                remove_bbox_if_covered_gt=0.70,
                p=1.0,
            ),
        ],
        bbox_params=A.BboxParams(format="albumentations", min_area=100, min_visibility=0.3),
    )
    return Augumentation(transform=transformation, is_bboxes=True)


def transform_big_props_make() -> Augumentation:
    """Add random props from yaya_tools/data/props"""
    transformation = A.Compose(
        [
            PropsAugmentation(
                prop_dir="yaya_tools/data/props",
                n_props=(1, 2),
                opacity=(0.99, 1.0),  # <- class parameter
                autoscale=True,  # <- class parameter
                scale_range=(0.80, 1.25),  # fraction of min(H, W) for prop's longer side
                rotate_limit=10,
                flip_prob=0.5,
                remove_bbox_if_covered_gt=0.70,
                p=1.0,
            ),
        ],
        bbox_params=A.BboxParams(format="albumentations", min_area=100, min_visibility=0.3),
    )
    return Augumentation(transform=transformation, is_bboxes=True)


def augmentation_select(args: argparse.Namespace) -> Optional[Augumentation]:
    """Select any of augmentation, None if not set."""
    if args.flip_horizontal:
        return transform_flip_horizontal_make()
    if args.flip_vertical:
        return transform_flip_vertical_make()
    if args.crop != -1:
        return transform_crop_make(args.crop)
    if args.rotate != -1:
        return transform_rotate_make(args.rotate)
    if args.randrotate:
        return transform_rotate_make(args.randrotate)
    if args.brighten:
        return transform_brighten_make()
    if args.sharpen:
        return transform_sharpen_make()
    if args.darken:
        return transform_darken_make()
    if args.clahe:
        return transform_clahe_make()
    if args.equalize:
        return transform_equalize_make()
    if args.colorshift:
        return transform_colorshift_make()
    if args.isonoise:
        return transform_isonoise_make()
    if args.gaussnoise:
        return transform_gaussnoise_make()
    if args.multi_noise:
        return transform_multinoise_make()
    if args.blur:
        return transform_blur_make()
    if args.blur_delicate:
        return transform_blur_delicate_make()
    if args.downsize_padding:
        return transform_downsize_padding_make()
    if args.compression:
        return transform_compression_make()
    if args.degrade:
        return transform_degrade_make()
    if args.spatter:
        return transform_spatter_make()
    if args.spatter_big:
        return transform_spatter_big_make()
    if args.spatter_small:
        return transform_spatter_small_make()
    if args.blackboxing != -1:
        return transform_blackboxing_make()
    if args.snow:
        return transform_snow_make()
    if args.rain:
        return transform_rain_make()
    if args.fog:
        return transform_fog_make()
    if args.sunflare:
        return transform_sunflare_make()
    if args.medianblur:
        return transform_median_blur_make()
    if args.grayscale:
        return transform_grayscale_make()
    if args.sepia:
        return transform_sepia_make()
    if args.props:
        return transform_multi_props_make()
    if args.big_props:
        return transform_big_props_make()

    return None
