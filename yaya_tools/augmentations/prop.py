# props_augmentation.py
import glob
import logging
import os
from typing import Any, List, Tuple, Union

import cv2
import numpy as np
from albumentations.core.transforms_interface import DualTransform  # type: ignore

FloatOrRange = Union[float, Tuple[float, float]]

logger = logging.getLogger(__name__)


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

    def _load_props(self) -> None:
        """Load all PNG props from directory"""
        paths = sorted(glob.glob(os.path.join(self.prop_dir, "*.png")))
        if not paths:
            raise FileNotFoundError(f"No PNG props found in: {self.prop_dir}")

        for pth in paths:
            img = cv2.imread(pth, cv2.IMREAD_UNCHANGED)  # HxWx(3|4)
            if img is None:
                continue
            elif img.shape[2] == 3:
                # No alpha channel; synthesize fully-opaque alpha
                bgr = img
                alpha = np.full(bgr.shape[:2], 255, dtype=np.uint8)
                img = np.dstack([bgr, alpha])
            elif img.shape[2] == 4:
                pass
            else:
                continue

            rgba = img  # cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            alpha = rgba[..., 3]  # type: ignore
            # Binary mask: non-zero alpha -> 1
            mask = (alpha > 0).astype(np.uint8)
            self._props_rgba.append(rgba)
            self._props_mask.append(mask)

        if not self._props_rgba:
            raise RuntimeError(f"Failed to load any valid props from {self.prop_dir}")

        # Logger : Info
        logger.info(f"Loaded {len(self._props_rgba)} props from {self.prop_dir}")

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
        rotated_mask = cv2.warpAffine(  # type: ignore
            mask,
            M,
            (new_w, new_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return rotated_rgba, rotated_mask

    def get_params_dependent_on_targets(self, params) -> dict[str, Any]:
        """Get parameters which depend on the image shape."""
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

            # scale
            if self.autoscale:
                s_frac = np.random.uniform(self.scale_range[0], self.scale_range[1])
                target_long = max(4, s_frac * min(H, W))
                scale = float(target_long) / float(max(ph, pw))
            else:
                scale = np.random.uniform(self.scale_range[0], self.scale_range[1])

            new_w = max(1, int(round(pw * scale)))
            new_h = max(1, int(round(ph * scale)))
            rgba = cv2.resize(rgba_base, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            msk = cv2.resize(mask_base, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            # flip
            if np.random.rand() < self.flip_prob:
                rgba = cv2.flip(rgba, 1)
                msk = cv2.flip(msk, 1)

            # rotate
            angle = float(np.random.uniform(-self.rotate_limit, self.rotate_limit))
            rgba, msk = self._rotate_rgba_and_mask(rgba, msk, angle)

            h, w = msk.shape[:2]
            if h >= H or w >= W:
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

            placements.append({"rgba": rgba, "mask": msk, "x": x, "y": y, "opacity": _rand_from(self.opacity)})
            union_mask[y : y + h, x : x + w] |= msk.astype(np.uint8)

        # everything needed by apply/apply_to_bboxes must be returned here
        return {"placements": placements, "occlusion_mask": union_mask, "rows": H, "cols": W}

    def apply(self, image: np.ndarray, *args, **params) -> np.ndarray:
        """Apply the props augmentation to the image."""
        placements = params.get("placements", None)
        if placements is None:
            shape = params.get("shape")
            if shape is None:
                H, W = image.shape[:2]
            else:
                H, W = int(shape[0]), int(shape[1])
            seed = int(params.get("seed", np.random.randint(0, 2**31 - 1)))
            params_all = self.get_params_dependent_on_targets({"image": image, "seed": seed, "shape": (H, W)})
            placements = params_all.get("placements", [])

        if not placements:
            return image

        out = image.copy()
        H, W = out.shape[:2]

        for pl in placements:
            rgba: np.ndarray = pl["rgba"]
            msk: np.ndarray = pl["mask"]
            x: int = int(pl["x"])
            y: int = int(pl["y"])
            op: float = float(pl["opacity"])

            h, w = msk.shape[:2]
            if x >= W or y >= H or x + w <= 0 or y + h <= 0:
                continue

            img_x0 = max(0, x)
            img_y0 = max(0, y)
            prop_x0 = max(0, -x)
            prop_y0 = max(0, -y)
            prop_w = min(w - prop_x0, W - img_x0)
            prop_h = min(h - prop_y0, H - img_y0)
            if prop_w <= 0 or prop_h <= 0:
                continue

            prop_rgb = rgba[prop_y0 : prop_y0 + prop_h, prop_x0 : prop_x0 + prop_w, :3].astype(np.float32)
            prop_a = (rgba[prop_y0 : prop_y0 + prop_h, prop_x0 : prop_x0 + prop_w, 3].astype(np.float32) / 255.0) * op
            prop_a = np.clip(prop_a, 0.0, 1.0)

            roi = out[img_y0 : img_y0 + prop_h, img_x0 : img_x0 + prop_w, :].astype(np.float32)
            blended = roi * (1.0 - prop_a[..., None]) + prop_rgb * prop_a[..., None]
            out[img_y0 : img_y0 + prop_h, img_x0 : img_x0 + prop_w, :] = blended.astype(np.uint8)

        return out

    def apply_to_bboxes(self, bboxes: np.ndarray, *args, **params) -> np.ndarray:
        """
        bboxes: array of shape (N, 4+) in pascal_voc pixel coords.
        Returns an array with boxes removed if occluded > threshold.
        """
        occlusion_mask = params.get("occlusion_mask", None)
        H = int(params.get("rows", 0))
        W = int(params.get("cols", 0))

        if occlusion_mask is None or bboxes is None or len(bboxes) == 0:
            # Ensure ndarray out
            return np.asarray(bboxes) if not isinstance(bboxes, np.ndarray) else bboxes

        if isinstance(occlusion_mask, list):
            occlusion_mask = np.array(occlusion_mask, dtype=np.uint8)

        Hm, Wm = occlusion_mask.shape[:2]
        if H and W and (Hm != H or Wm != W):
            occlusion_mask = cv2.resize(occlusion_mask, (W, H), interpolation=cv2.INTER_NEAREST)
            Hm, Wm = H, W

        b = np.asarray(bboxes, dtype=np.float32)
        keep_mask = np.ones((b.shape[0],), dtype=bool)

        for i, (x1, y1, x2, y2, *_rest) in enumerate(b):
            xi1 = int(max(0, min(Wm - 1, np.floor(x1))))
            yi1 = int(max(0, min(Hm - 1, np.floor(y1))))
            xi2 = int(max(0, min(Wm, np.ceil(x2))))
            yi2 = int(max(0, min(Hm, np.ceil(y2))))

            w = max(0, xi2 - xi1)
            h = max(0, yi2 - yi1)
            area = max(1, w * h)
            if w == 0 or h == 0:
                keep_mask[i] = False
                continue

            sub = occlusion_mask[yi1:yi2, xi1:xi2]
            covered = int(sub.sum())
            coverage = covered / float(area)

            if coverage > self.remove_bbox_if_covered_gt:
                keep_mask[i] = False

        return b[keep_mask]

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
