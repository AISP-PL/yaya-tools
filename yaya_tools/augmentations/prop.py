# props_augmentation.py
import glob
import os
from typing import Any, List, Sequence, Tuple, Union

import cv2
import numpy as np
from albumentations.core.transforms_interface import DualTransform  # type: ignore

FloatOrRange = Union[float, Tuple[float, float]]


def _as_range(v: FloatOrRange) -> Tuple[float, float]:
    """Convert float or (min, max) to (min, max)"""
    if isinstance(v, (list, tuple)):
        assert len(v) == 2, "Range must be (min, max)"
        return float(v[0]), float(v[1])

    return float(v), float(v)


def _rand_from(v: FloatOrRange) -> float:
    """Sample a random float from fixed value or range."""
    lo, hi = _as_range(v)
    return np.random.uniform(lo, hi)


class PropsAugmentation(DualTransform):
    """
    Drop random 'props' (overlays) onto an image, blend with opacity, and
    remove bboxes that are heavily occluded by the prop mask.

    - Loads all PNGs from prop_dir (expects transparent background if available).
    - Converts each prop to RGBA and binary mask (non-zero alpha -> 1).
    - Places N random props (configurable range) with random rotation and flip.
    - Optional autoscaling relative to image size; otherwise uses scale factor.
    - Blends onto image with configurable opacity (fixed or range).
    - If > `remove_bbox_if_covered_gt` of a bbox area is covered by the union prop mask,
      that bbox is dropped.

    Notes:
      * Expects bboxes in PAS-CAL VOC pixel coords (x_min, y_min, x_max, y_max).
      * Use with A.Compose(..., bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])).
    """

    def __init__(
        self,
        prop_dir: str = "data/props",
        n_props: Tuple[int, int] = (1, 2),
        opacity: FloatOrRange = (0.8, 1.0),
        autoscale: bool = True,
        # If autoscale=True: fraction of the image's shorter side used as the prop's longer side.
        # If autoscale=False: multiplicative factor applied to prop's original size.
        scale_range: Tuple[float, float] = (0.25, 0.6),
        rotate_limit: float = 5.0,  # degrees, sampled uniformly from [-rotate_limit, +rotate_limit]
        flip_prob: float = 0.5,
        remove_bbox_if_covered_gt: float = 0.70,
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        self.prop_dir = prop_dir
        self.n_props = n_props
        self.opacity = opacity
        self.autoscale = autoscale
        self.scale_range = scale_range
        self.rotate_limit = float(rotate_limit)
        self.flip_prob = float(flip_prob)
        self.remove_bbox_if_covered_gt = float(remove_bbox_if_covered_gt)

        self._props_rgba: List[np.ndarray] = []
        self._props_mask: List[np.ndarray] = []
        self._load_props()

    @property
    def targets(self) -> dict[str, Any]:
        # Support image, masks (if present), bboxes (filtered here), keypoints untouched
        return {"image": self.apply, "bboxes": self.apply_to_bboxes}

    def _load_props(self) -> None:
        """Load all PNG props from directory"""
        paths = sorted(glob.glob(os.path.join(self.prop_dir, "*.png")))
        if not paths:
            raise FileNotFoundError(f"No PNG props found in: {self.prop_dir}")

        for pth in paths:
            img = cv2.imread(pth, cv2.IMREAD_UNCHANGED)  # HxWx(3|4)
            if img is None:
                continue
            if img.ndim == 2:  # grayscale -> RGB
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
            elif img.shape[2] == 3:
                # No alpha channel; synthesize fully-opaque alpha
                bgr = img
                alpha = np.full(bgr.shape[:2], 255, dtype=np.uint8)
                img = np.dstack([bgr, alpha])
            elif img.shape[2] == 4:
                pass
            else:
                continue

            rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            alpha = rgba[..., 3]
            # Binary mask: non-zero alpha -> 1
            mask = (alpha > 0).astype(np.uint8)
            self._props_rgba.append(rgba)
            self._props_mask.append(mask)

        if not self._props_rgba:
            raise RuntimeError(f"Failed to load any valid props from {self.prop_dir}")

    def get_params_dependent_on_targets(self, params: dict[str, Any]) -> dict[str, Any]:
        """Get parameters that depend on the target image."""
        image: np.ndarray = params["image"]
        H, W = image.shape[:2]

        n = int(np.random.randint(self.n_props[0], self.n_props[1] + 1))
        placements = []
        union_mask = np.zeros((H, W), dtype=np.uint8)

        for _ in range(n):
            idx = int(np.random.randint(0, len(self._props_rgba)))
            rgba_base = self._props_rgba[idx]
            mask_base = self._props_mask[idx]

            ph, pw = mask_base.shape[:2]

            # Determine scale
            if self.autoscale:
                # scale target: prop's longer side --> s * min(H, W)
                s_frac = np.random.uniform(self.scale_range[0], self.scale_range[1])
                target_long = max(4, s_frac * min(H, W))
                scale = float(target_long) / float(max(ph, pw))
            else:
                scale = np.random.uniform(self.scale_range[0], self.scale_range[1])

            # Resize prop
            new_w = max(1, int(round(pw * scale)))
            new_h = max(1, int(round(ph * scale)))
            rgba = cv2.resize(rgba_base, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            msk = cv2.resize(mask_base, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            # Optional flip
            if np.random.rand() < self.flip_prob:
                rgba = cv2.flip(rgba, 1)
                msk = cv2.flip(msk, 1)

            # Rotation
            angle = float(np.random.uniform(-self.rotate_limit, self.rotate_limit))
            rgba, msk = self._rotate_rgba_and_mask(rgba, msk, angle)

            h, w = msk.shape[:2]

            # Random position (clip to image frame)
            if h >= H or w >= W:
                # Clamp to fit by scaling down slightly
                scale_back = min((H - 1) / max(1, h), (W - 1) / max(1, w), 0.95)
                if scale_back <= 0:
                    continue
                new_w2 = max(1, int(round(w * scale_back)))
                new_h2 = max(1, int(round(h * scale_back)))
                rgba = cv2.resize(rgba, (new_w2, new_h2), interpolation=cv2.INTER_LINEAR)
                msk = cv2.resize(msk, (new_w2, new_h2), interpolation=cv2.INTER_NEAREST)
                h, w = msk.shape[:2]

            max_y = max(0, H - h)
            max_x = max(0, W - w)
            y = int(np.random.randint(0, max_y + 1)) if max_y > 0 else 0
            x = int(np.random.randint(0, max_x + 1)) if max_x > 0 else 0

            # Store placement and write into union mask
            placements.append(
                {
                    "rgba": rgba,
                    "mask": msk,
                    "x": x,
                    "y": y,
                    "opacity": _rand_from(self.opacity),
                }
            )
            union_mask[y : y + h, x : x + w] |= msk.astype(np.uint8)

        return {"placements": placements, "occlusion_mask": union_mask}

    @staticmethod
    def _rotate_rgba_and_mask(rgba: np.ndarray, mask: np.ndarray, angle_deg: float):
        """Rotate with auto-bound so nothing gets cut off."""
        h, w = mask.shape[:2]
        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

        # compute bounds of rotated image
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        new_w = int(np.ceil((h * sin) + (w * cos)))
        new_h = int(np.ceil((h * cos) + (w * sin)))

        # adjust transform to keep image centered
        M[0, 2] += (new_w / 2.0) - center[0]
        M[1, 2] += (new_h / 2.0) - center[1]

        rotated_rgba = cv2.warpAffine(
            rgba,
            M,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        rotated_mask = cv2.warpAffine(
            mask,
            M,
            (new_w, new_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return rotated_rgba, rotated_mask

    def apply(self, image: np.ndarray, placements=None, occlusion_mask=None, **kwargs) -> np.ndarray:
        if not placements:
            return image

        out = image.copy()
        H, W = out.shape[:2]

        for pl in placements:
            rgba: np.ndarray = pl["rgba"]
            msk: np.ndarray = pl["mask"]
            x: int = pl["x"]
            y: int = pl["y"]
            op: float = float(pl["opacity"])

            h, w = msk.shape[:2]
            x2 = min(W, x + w)
            y2 = min(H, y + h)
            if x >= W or y >= H or x2 <= 0 or y2 <= 0:
                continue  # completely out of frame

            # Compute slicing (in case of border clipping)
            prop_x0 = 0 if x >= 0 else -x
            prop_y0 = 0 if y >= 0 else -y
            img_x0 = max(0, x)
            img_y0 = max(0, y)
            prop_w = x2 - img_x0
            prop_h = y2 - img_y0
            if prop_w <= 0 or prop_h <= 0:
                continue

            prop_rgb = rgba[prop_y0 : prop_y0 + prop_h, prop_x0 : prop_x0 + prop_w, :3].astype(np.float32)
            prop_a = (rgba[prop_y0 : prop_y0 + prop_h, prop_x0 : prop_x0 + prop_w, 3].astype(np.float32) / 255.0) * op
            prop_a = np.clip(prop_a, 0.0, 1.0)

            roi = out[img_y0 : img_y0 + prop_h, img_x0 : img_x0 + prop_w, :].astype(np.float32)

            # Alpha blend
            a3 = prop_a[..., None]
            blended = roi * (1.0 - a3) + prop_rgb * a3
            out[img_y0 : img_y0 + prop_h, img_x0 : img_x0 + prop_w, :] = blended.astype(np.uint8)

        return out

    def apply_to_bboxes(
        self, bboxes: Sequence[Tuple[float, float, float, float]], occlusion_mask=None, rows=0, cols=0, **kwargs
    ):
        if occlusion_mask is None or len(bboxes) == 0:
            return list(bboxes)

        H, W = occlusion_mask.shape[:2]
        keep: List[Tuple[float, float, float, float]] = []

        for bbox in bboxes:
            # Expect (x_min, y_min, x_max, y_max[, ...])
            x1, y1, x2, y2 = bbox[:4]
            # Clamp to image
            xi1 = int(max(0, min(W - 1, np.floor(x1))))
            yi1 = int(max(0, min(H - 1, np.floor(y1))))
            xi2 = int(max(0, min(W, np.ceil(x2))))
            yi2 = int(max(0, min(H, np.ceil(y2))))

            w = max(0, xi2 - xi1)
            h = max(0, yi2 - yi1)
            area = max(1, w * h)

            if w == 0 or h == 0:
                # degenerate / fully outside
                continue

            sub = occlusion_mask[yi1:yi2, xi1:xi2]
            covered = int(sub.sum())
            coverage = covered / float(area)

            if coverage <= self.remove_bbox_if_covered_gt:
                keep.append(bbox)
            # else: drop it

        return keep

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        # For serialization
        return (
            "prop_dir",
            "n_props",
            "opacity",
            "autoscale",
            "scale_range",
            "rotate_limit",
            "flip_prob",
            "remove_bbox_if_covered_gt",
        )
