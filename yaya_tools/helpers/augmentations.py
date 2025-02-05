import logging
from typing import NamedTuple

import albumentations as A  # type: ignore
import cv2


class Augumentation(NamedTuple):
    """This class is used to store the transformation and the type of the transformation"""

    transform: A.Compose
    is_bboxes: bool


# Shape : Albumentations transform
def transform_crop_make(width: int = 640) -> A.Compose:
    """Create crop transformation."""

    # Width and height must be divisible by 32
    # - min width is 640
    # - height almost 16/9 ratio to width
    width = max(640, width)
    height = int(width * 9 / 16)
    height = height - (height % 32)
    width = width - (width % 32)

    return A.Compose(
        [A.RandomCrop(width=width, height=height, p=0.99)],
        bbox_params=A.BboxParams(format="yolo", min_area=100, min_visibility=0.3),
    )


def transform_rotate_make(degrees: int = 30) -> A.Compose:
    """Create rotate transformation."""

    return A.Compose(
        [
            A.Rotate(
                limit=(degrees, degrees),
                border_mode=cv2.BORDER_CONSTANT,
                rotate_method="ellipse",
                p=0.99,
            )
        ],
        bbox_params=A.BboxParams(format="yolo", min_area=100, min_visibility=0.3),
    )


def transform_randrotate_make(degrees: int = 30) -> A.Compose:
    """Create rotate transformation."""

    return A.Compose(
        [
            A.Rotate(
                limit=degrees,
                border_mode=cv2.BORDER_CONSTANT,
                rotate_method="ellipse",
                p=0.99,
            )
        ],
        bbox_params=A.BboxParams(format="yolo", min_area=100, min_visibility=0.3),
    )


def transform_flip_horizontal_make() -> A.Compose:
    """Create flip transformation."""
    return A.Compose(
        [A.HorizontalFlip(p=0.99, always_apply=True)],
        bbox_params=A.BboxParams(format="yolo", min_area=100, min_visibility=0.3),
    )


def transform_flip_vertical_make() -> A.Compose:
    """Create flip transformation."""
    return A.Compose(
        [A.VerticalFlip(p=0.99, always_apply=True)],
        bbox_params=A.BboxParams(format="yolo", min_area=100, min_visibility=0.3),
    )


def transform_flip_make() -> A.Compose:
    """Create flip transformation."""

    return A.Compose(
        [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)],
        bbox_params=A.BboxParams(format="yolo", min_area=100, min_visibility=0.3),
    )


# Shape : Albumentations transform
transform_shape = A.Compose(
    [
        A.SomeOf(
            [
                A.ImageCompression(quality_lower=30, quality_upper=55, p=0.1),
                A.MotionBlur(blur_limit=3, p=0.1),
            ],
            n=2,
        ),
        A.GridDistortion(num_steps=3, distort_limit=0.25, p=0.1),
        A.SomeOf(
            [
                A.RandomCrop(width=480, height=320, p=0.60),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,
                    rotate_limit=15,
                    p=0.25,
                    border_mode=cv2.BORDER_CONSTANT,
                ),
                A.ElasticTransform(alpha_affine=9, p=0.2, border_mode=cv2.BORDER_CONSTANT),
            ],
            n=1,
        ),
        A.SomeOf(
            [
                A.OpticalDistortion(distort_limit=0.2, p=0.2, border_mode=cv2.BORDER_CONSTANT),
                A.ZoomBlur(max_factor=1.1, p=0.2),
                A.GaussNoise(p=0.2),
                A.RandomShadow(p=0.1),
            ],
            n=1,
        ),
    ],
    bbox_params=A.BboxParams(format="yolo", min_area=100, min_visibility=0.3),
)


def transform_compression_make() -> A.Compose:
    """Create compression transformation."""
    return A.Compose(
        [A.ImageCompression(quality_lower=10, quality_upper=15, p=0.99)],
        bbox_params=A.BboxParams(format="yolo", min_area=100, min_visibility=0.3),
    )


def transform_degrade_make() -> A.Compose:
    """Create compression transformation."""
    return A.Compose(
        [
            A.Downscale(scale_min=0.25, scale_max=0.45, p=0.999),
        ],
        bbox_params=A.BboxParams(format="yolo", min_area=100, min_visibility=0.3),
    )


def transform_clahe_make() -> A.Compose:
    """Create CLAHE transformation."""
    return A.Compose(
        [A.CLAHE(p=0.999, always_apply=True)],
        bbox_params=A.BboxParams(format="yolo", min_area=100, min_visibility=0.3),
    )


def transform_equalize_make() -> A.Compose:
    """Create equalize transformation."""
    return A.Compose(
        [A.Equalize(p=0.999, always_apply=True)],
        bbox_params=A.BboxParams(format="yolo", min_area=100, min_visibility=0.3),
    )


def transform_sharpen_make() -> A.Compose:
    """Create sharpen transformation."""
    return A.Compose(
        [A.Sharpen(alpha=(0.25, 0.5), lightness=(0.7, 1.1), p=0.999, always_apply=True)],
        bbox_params=A.BboxParams(format="yolo", min_area=100, min_visibility=0.3),
    )


def transform_downsize_padding_make() -> A.Compose:
    """Downsize with padding using ShiftScaleRotate"""
    return A.Compose(
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
        bbox_params=A.BboxParams(format="yolo", min_area=100, min_visibility=0.3),
    )


def transform_blur_delicate_make() -> Augumentation:
    """Create blur transformation."""
    transform = A.Compose(
        [A.Blur(blur_limit=3, p=0.99)],
        bbox_params=A.BboxParams(format="yolo", min_area=100, min_visibility=0.3),
    )
    return Augumentation(transform=transform, is_bboxes=False)


def transform_blur_make() -> A.Compose:
    """Create blur transformation."""
    return A.Compose(
        [A.Blur(blur_limit=7, p=0.99)],
        bbox_params=A.BboxParams(format="yolo", min_area=100, min_visibility=0.3),
    )


def transform_median_blur_make() -> A.Compose:
    """Create median blur transformation."""
    return A.Compose(
        [A.MedianBlur(blur_limit=7, p=0.99)],
        bbox_params=A.BboxParams(format="yolo", min_area=100, min_visibility=0.3),
    )


def transform_colorshift_make() -> A.Compose:
    """Create random brighten transformation."""
    return A.Compose(
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
        bbox_params=A.BboxParams(format="yolo", min_area=100, min_visibility=0.3),
    )


def transform_brighten_make() -> A.Compose:
    """Create random brighten transformation."""
    return A.Compose(
        [
            A.RandomBrightnessContrast(
                brightness_limit=(0.0, 0.4),
                contrast_limit=(-0.35, 0.35),
                always_apply=True,
                p=0.999,
            )
        ],
        bbox_params=A.BboxParams(format="yolo", min_area=100, min_visibility=0.3),
    )


def transform_darken_make() -> A.Compose:
    """Create random darken transformation."""
    return A.Compose(
        [
            A.RandomBrightnessContrast(
                brightness_limit=(-0.4, 0.0),
                contrast_limit=(-0.35, 0.35),
                always_apply=True,
                p=0.999,
            )
        ],
        bbox_params=A.BboxParams(format="yolo", min_area=100, min_visibility=0.3),
    )


def transform_snow_make() -> A.Compose:
    """Create snow transformation."""
    return A.Compose(
        [A.RandomSnow(p=0.999)],
        bbox_params=A.BboxParams(format="yolo", min_area=100, min_visibility=0.3),
    )


def transform_rain_make() -> A.Compose:
    """Create rain transformation."""
    return A.Compose(
        [A.RandomRain(drop_length=10, blur_value=4, p=0.999)],
        bbox_params=A.BboxParams(format="yolo", min_area=100, min_visibility=0.3),
    )


def transform_spatter_big_make() -> A.Compose:
    """Create spatter transformation."""
    return A.Compose(
        [A.Spatter(p=0.999, gauss_sigma=4.5)],
        bbox_params=A.BboxParams(format="yolo", min_area=100, min_visibility=0.3),
    )


def transform_spatter_make() -> A.Compose:
    """Create spatter transformation."""
    return A.Compose(
        [A.Spatter(p=0.999)],
        bbox_params=A.BboxParams(format="yolo", min_area=100, min_visibility=0.3),
    )


def transform_spatter_small_make() -> A.Compose:
    """Create spatter transformation."""
    return A.Compose(
        [A.Spatter(p=0.999, gauss_sigma=0.5)],
        bbox_params=A.BboxParams(format="yolo", min_area=100, min_visibility=0.3),
    )


def transform_blackboxing_make(size: int = 50) -> A.Compose:
    """Blackboxing parts of image."""
    logging.warning("Blackboxing not supports bbox target!")
    return A.Compose(
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


def transform_isonoise_make() -> A.Compose:
    """Create isonoise transformation."""
    return A.Compose(
        [
            A.ISONoise(
                color_shift=(0.04, 0.30),
                intensity=(0.33, 1.25),
                always_apply=True,
                p=0.999,
            )
        ],
        bbox_params=A.BboxParams(format="yolo", min_area=100, min_visibility=0.3),
    )


def transform_gaussnoise_make() -> A.Compose:
    """Create gauss noise transformation."""
    return A.Compose(
        [
            A.GaussNoise(
                var_limit=(10.0, 70.0),
                mean=0,
                always_apply=True,
                p=0.999,
            )
        ],
        bbox_params=A.BboxParams(format="yolo", min_area=100, min_visibility=0.3),
    )


def transform_multinoise_make() -> A.Compose:
    """Create multiplicative noise transformation."""
    return A.Compose(
        [
            A.MultiplicativeNoise(
                multiplier=(0.85, 1.15),
                per_channel=True,
                elementwise=True,
                always_apply=True,
                p=0.999,
            )
        ],
        bbox_params=A.BboxParams(format="yolo", min_area=100, min_visibility=0.3),
    )


def transform_fog_make() -> A.Compose:
    """Create fog transformation."""
    return A.Compose(
        [A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.5, alpha_coef=0.5, p=0.999)],
        bbox_params=A.BboxParams(format="yolo", min_area=100, min_visibility=0.3),
    )


def transform_sunflare_make() -> A.Compose:
    """Create sunflare transformation."""
    return A.Compose(
        [
            A.RandomSunFlare(
                src_radius=260,
                num_flare_circles_lower=2,
                num_flare_circles_upper=6,
                p=0.999,
            )
        ],
        bbox_params=A.BboxParams(format="yolo", min_area=100, min_visibility=0.3),
    )


# Color : Albumentations transform
transform_color = A.Compose(
    [
        # Colors : Brightnes, contrast, tone curve, hue, saturation, value
        A.SomeOf(
            [
                A.RandomBrightnessContrast(p=0.3),
                A.RandomToneCurve(scale=0.5, p=0.3),
                A.HueSaturationValue(p=0.3),
                A.ColorJitter(p=0.2),
                A.Equalize(p=0.3),
                A.ToSepia(p=0.1),
            ],
            n=2,
        ),
        # Quality : Jpeg compression, multiplicative noise, downscale
        A.OneOf(
            [
                A.ImageCompression(quality_lower=30, quality_upper=55, p=0.3),
                A.MultiplicativeNoise(p=0.2),
                A.Downscale(scale_min=0.4, scale_max=0.6, p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.ISONoise(color_shift=(0.01, 0.08), intensity=(0.2, 0.8), p=0.1),
                A.PixelDropout(dropout_prob=0.1, p=0.1),
                A.Spatter(p=0.1),
                A.Superpixels(p=0.1),
                A.GlassBlur(sigma=0.2, max_delta=2, iterations=1, p=0.1),
            ]
        ),
        # Weather : Dropouts, rain, snow, sun flare, fog
        A.OneOf(
            [
                A.RandomRain(drop_length=10, blur_value=4, p=0.1),
                A.RandomSnow(p=0.1),
                A.RandomSunFlare(
                    src_radius=260,
                    num_flare_circles_lower=2,
                    num_flare_circles_upper=6,
                    p=0.1,
                ),
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.5, alpha_coef=0.5, p=0.1),
            ]
        ),
    ],
    bbox_params=A.BboxParams(format="yolo", min_area=100, min_visibility=0.2),
)

# All : Full transform
transform_all = A.Compose(
    [
        A.SomeOf([transform_color], n=3, p=0.5),
        A.SomeOf([transform_shape], n=3, p=0.5),
    ],
    bbox_params=A.BboxParams(format="yolo", min_area=100, min_visibility=0.2),
)
