"""
MIT-B2 U-Net Inference Engine — Historical Oil Spill Investigation
==================================================================
Module: sar_pipeline/mit_b2_inference.py

Provides a clean callable interface for running the MiT-B2 + U-Net
segmentation model on uploaded images from the historical investigation workflow.

Model specifics:
    - Architecture : MiTB2UNet (nvidia/mit-b2 encoder + 4-stage U-Net decoder)
    - Checkpoint   : models/mit_b2_unet_best.pth  (330 MB, Git LFS)
    - Input        : [B, 3, H, W] normalized RGB image — resized to 256×256
    - Output       : [B, 1, H, W] raw logits — sigmoid -> binary mask
    - Val Dice     : 0.84 | IoU : 0.75

Preprocessing for uploaded images (JPEG / PNG / user photo):
    Image -> resize to 256×256 -> float32 [0,1] / 255 -> channel normalize
    Uses ImageNet mean/std because the SegFormer backbone (nvidia/mit-b2) was
    pretrained on ImageNet.  This matches the training pipeline for the checkpoint.

IMPORTANT — DATA MODE:
    This module is ONLY for the historical investigation feature.
    Do NOT use it for the real-time pipeline (which uses SARUNetDetector /
    VanillaUNet / sar_unet_oil_spill.pt).

DO NOT call this model "AI" — it is a deep-learning segmentation model.
"""

from __future__ import annotations

import base64
import io
import logging
import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ── Model configuration ────────────────────────────────────────────────────────
_CHECKPOINT_REL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "mit_b2_unet_best.pth"
)
_TARGET_SIZE: int = 256        # model input spatial size
_THRESHOLD: float = 0.45       # sigmoid threshold for binary mask
_MIN_SPILL_PIXELS: int = 50    # reject sub-50-pixel blobs

# ImageNet normalization — matches SegFormer (nvidia/mit-b2) backbone pretraining
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class SpillSegmentationResult:
    """
    Output of MIT-B2 U-Net inference on a user-uploaded SAR/optical image.

    geometry_source:
        "IMAGE_DERIVED" — segmentation produced a usable polygon
        "POINT_ONLY"    — no confident spill detected; use user-supplied lat/lon
    """
    model_name: str = "MIT-B2-U-Net"

    # Detection outcome
    detected: bool = False
    geometry_source: str = "POINT_ONLY"   # "IMAGE_DERIVED" | "POINT_ONLY"

    # Mask statistics (pixel space)
    spill_pixel_count: int = 0
    total_pixels: int = 0
    spill_fraction: float = 0.0            # 0-1 fraction of image

    # Geographic / spatial outputs
    spill_area_km2: Optional[float] = None
    centroid_rel: Optional[Tuple[float, float]] = None   # (row_frac, col_frac) relative to image
    bounding_box_px: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2) pixel
    bounding_box_rel: Optional[Tuple[float, float, float, float]] = None  # relative 0-1

    # Polygon in relative image coordinates [(row_frac, col_frac), ...]
    # Caller maps to geographic coords using the supplied lat/lon + pixel resolution
    polygon_rel: Optional[List[Tuple[float, float]]] = None

    # Segmentation confidence (mean probability over positive pixels)
    segmentation_confidence: Optional[float] = None

    # Binary mask as base64-encoded PNG (RGBA: mask in alpha channel)
    mask_png_base64: Optional[str] = None

    # Reason string for non-detections
    reason: str = ""

    # Error message (set if inference failed)
    error: Optional[str] = None


# ── Singleton model holder ────────────────────────────────────────────────────

class MiTB2InferenceEngine:
    """
    Singleton inference engine for MiTB2UNet.

    The model is loaded once on first use (lazy loading) and reused
    for all subsequent inference calls.  Thread-safety for the global
    load is handled by Python's GIL during the load call.

    Usage:
        engine = MiTB2InferenceEngine.instance()
        result = engine.analyze(image_bytes)
    """

    _singleton: Optional["MiTB2InferenceEngine"] = None

    def __init__(self) -> None:
        self._model = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._loaded = False
        self._load_error: Optional[str] = None

    @classmethod
    def instance(cls) -> "MiTB2InferenceEngine":
        if cls._singleton is None:
            cls._singleton = cls()
        return cls._singleton

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self._load_error:
            raise RuntimeError(f"MIT-B2 model previously failed to load: {self._load_error}")

        checkpoint_path = os.path.abspath(_CHECKPOINT_REL_PATH)
        if not os.path.exists(checkpoint_path):
            msg = (
                f"MIT-B2 checkpoint not found at {checkpoint_path}. "
                "Ensure mit_b2_unet_best.pth is present (Git LFS pull)."
            )
            self._load_error = msg
            raise RuntimeError(msg)

        try:
            from sar_pipeline.mit_b2_unet import load_mit_b2_checkpoint
            logger.info("[MIT-B2] Loading checkpoint from %s on device=%s", checkpoint_path, self._device)
            self._model = load_mit_b2_checkpoint(checkpoint_path, device=self._device)
            self._model.eval()
            self._loaded = True
            logger.info("[MIT-B2] Checkpoint loaded successfully.")
        except Exception as exc:
            self._load_error = str(exc)
            logger.error("[MIT-B2] Failed to load checkpoint: %s", exc)
            raise

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(
        self,
        image_input,   # bytes | PIL.Image | np.ndarray (H, W, 3) uint8
        pixel_resolution_m: float = 31.25,  # default: 8km/256px
    ) -> SpillSegmentationResult:
        """
        Run MIT-B2 U-Net segmentation on the provided image.

        Parameters
        ----------
        image_input : bytes | PIL.Image | np.ndarray
            Uploaded image.  Converted to RGB numpy array internally.
        pixel_resolution_m : float
            Approximate ground resolution (meters per pixel) at 256×256.
            Used to compute spill_area_km2.

        Returns
        -------
        SpillSegmentationResult
            Never raises — errors are captured in result.error.
        """
        try:
            self._ensure_loaded()
        except RuntimeError as exc:
            return SpillSegmentationResult(error=str(exc), reason="Model load failed")

        try:
            rgb_arr = _to_rgb_numpy(image_input)
        except Exception as exc:
            return SpillSegmentationResult(error=str(exc), reason="Image preprocessing failed")

        try:
            tensor = _preprocess_for_mit_b2(rgb_arr)   # [1, 3, 256, 256]
            tensor = tensor.to(self._device)

            with torch.no_grad():
                logits = self._model(tensor)             # [1, 1, 256, 256]
                probs  = torch.sigmoid(logits)           # [1, 1, 256, 256]
                mask   = (probs >= _THRESHOLD).float()   # [1, 1, 256, 256]

            prob_np = probs.squeeze().cpu().numpy()   # (256, 256)
            mask_np = mask.squeeze().cpu().numpy()    # (256, 256) float {0,1}

            return _build_result(prob_np, mask_np, pixel_resolution_m)

        except Exception as exc:
            logger.error("[MIT-B2] Inference error: %s", exc, exc_info=True)
            return SpillSegmentationResult(
                error=str(exc),
                reason="Inference failed",
            )


# ── Top-level convenience function ────────────────────────────────────────────

def analyze_oil_spill(
    image_input,
    pixel_resolution_m: float = 31.25,
) -> SpillSegmentationResult:
    """
    Convenience wrapper — uses the global MiTB2InferenceEngine singleton.

    Parameters
    ----------
    image_input : bytes | PIL.Image | np.ndarray
        Uploaded image to segment.
    pixel_resolution_m : float
        Ground resolution at model input size (256×256), metres per pixel.

    Returns
    -------
    SpillSegmentationResult
    """
    return MiTB2InferenceEngine.instance().analyze(image_input, pixel_resolution_m)


# ── Internal preprocessing ────────────────────────────────────────────────────

def _to_rgb_numpy(image_input) -> np.ndarray:
    """
    Convert image_input to a (H, W, 3) uint8 RGB numpy array.

    Accepts: bytes, PIL.Image, or np.ndarray.
    """
    from PIL import Image

    if isinstance(image_input, (bytes, bytearray)):
        img = Image.open(io.BytesIO(image_input)).convert("RGB")
        return np.array(img, dtype=np.uint8)

    # PIL.Image
    if hasattr(image_input, "convert"):
        img = image_input.convert("RGB")
        return np.array(img, dtype=np.uint8)

    # numpy array
    if isinstance(image_input, np.ndarray):
        arr = image_input
        if arr.ndim == 2:
            # Grayscale -> replicate to 3 channels
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.shape[-1] == 4:
            arr = arr[..., :3]  # drop alpha
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)
        return arr

    raise TypeError(f"Unsupported image_input type: {type(image_input)}")


def _preprocess_for_mit_b2(rgb_arr: np.ndarray) -> torch.Tensor:
    """
    Resize to (256, 256), normalize with ImageNet mean/std, return [1, 3, 256, 256].

    ImageNet normalization used because the SegFormer encoder (nvidia/mit-b2)
    was pre-trained on ImageNet and the checkpoint was fine-tuned on SAR imagery
    starting from those weights.
    """
    from PIL import Image

    img_pil = Image.fromarray(rgb_arr, mode="RGB")
    img_pil = img_pil.resize((_TARGET_SIZE, _TARGET_SIZE), Image.BILINEAR)
    arr = np.array(img_pil, dtype=np.float32) / 255.0   # (256, 256, 3) in [0, 1]

    # ImageNet channel normalization
    arr = (arr - _IMAGENET_MEAN) / (_IMAGENET_STD + 1e-8)   # (256, 256, 3)

    # HWC -> CHW -> add batch dim
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).float()  # [1, 3, 256, 256]
    return tensor


# ── Result construction ───────────────────────────────────────────────────────

def _build_result(
    prob_np: np.ndarray,     # (256, 256) float in [0, 1]
    mask_np: np.ndarray,     # (256, 256) float {0, 1}
    pixel_resolution_m: float,
) -> SpillSegmentationResult:
    """Build SpillSegmentationResult from raw model outputs."""
    H, W = mask_np.shape
    total_pixels = H * W
    spill_pixels = int(np.sum(mask_np > 0.5))
    spill_fraction = round(spill_pixels / max(1, total_pixels), 5)

    # No spill detected
    if spill_pixels < _MIN_SPILL_PIXELS:
        return SpillSegmentationResult(
            detected=False,
            geometry_source="POINT_ONLY",
            spill_pixel_count=spill_pixels,
            total_pixels=total_pixels,
            spill_fraction=spill_fraction,
            mask_png_base64=_encode_mask_png(mask_np),
            reason=(
                "No oil spill detected above confidence threshold. "
                f"({spill_pixels} positive pixels < {_MIN_SPILL_PIXELS} minimum). "
                "Using user-supplied lat/lon as spill location (POINT_ONLY)."
            ),
        )

    # Compute mean confidence over positive pixels
    pos_mask = mask_np > 0.5
    mean_conf = float(np.mean(prob_np[pos_mask])) if np.any(pos_mask) else 0.0

    # Bounding box (pixel space)
    rows, cols = np.where(pos_mask)
    y1, y2 = int(rows.min()), int(rows.max())
    x1, x2 = int(cols.min()), int(cols.max())

    # Centroid (relative coords)
    cy_rel = round(float(np.mean(rows)) / H, 5)
    cx_rel = round(float(np.mean(cols)) / W, 5)

    # Area estimate
    area_km2 = round(spill_pixels * (pixel_resolution_m / 1000.0) ** 2, 4)

    # Polygon from largest connected component (relative coords)
    polygon_rel = _extract_polygon_rel(mask_np, H, W)

    return SpillSegmentationResult(
        detected=True,
        geometry_source="IMAGE_DERIVED" if polygon_rel is not None else "POINT_ONLY",
        spill_pixel_count=spill_pixels,
        total_pixels=total_pixels,
        spill_fraction=spill_fraction,
        spill_area_km2=area_km2,
        centroid_rel=(cy_rel, cx_rel),
        bounding_box_px=(x1, y1, x2, y2),
        bounding_box_rel=(
            round(x1 / W, 4), round(y1 / H, 4),
            round(x2 / W, 4), round(y2 / H, 4),
        ),
        polygon_rel=polygon_rel,
        segmentation_confidence=round(mean_conf, 4),
        mask_png_base64=_encode_mask_png(mask_np),
        reason="",
    )


def _extract_polygon_rel(
    mask_np: np.ndarray,
    H: int,
    W: int,
) -> Optional[List[Tuple[float, float]]]:
    """
    Extract simplified polygon from the largest connected component of the mask.

    Returns a list of (row_frac, col_frac) relative coordinates,
    or None if no valid component found.
    """
    try:
        from scipy.ndimage import label, find_objects

        mask_bin = (mask_np > 0.5).astype(np.uint8)
        labeled, n = label(mask_bin)
        if n == 0:
            return None

        # Find largest component
        sizes = [(i + 1, int((labeled == i + 1).sum())) for i in range(n)]
        best_id = max(sizes, key=lambda x: x[1])[0]
        feat = (labeled == best_id)

        rows, cols = np.where(feat)
        if len(rows) < 3:
            return None

        # Convex hull of boundary pixels (fast approximation)
        pts = list(zip(rows.tolist(), cols.tolist()))
        # Sample at most 200 pts to keep polygon small
        step = max(1, len(pts) // 200)
        pts = pts[::step]

        hull = _convex_hull_2d(pts)
        if not hull:
            return None

        return [(round(r / H, 5), round(c / W, 5)) for r, c in hull]
    except ImportError:
        # scipy not available — return bounding box polygon
        rows_a, cols_a = np.where(mask_np > 0.5)
        if len(rows_a) == 0:
            return None
        r1, r2 = int(rows_a.min()), int(rows_a.max())
        c1, c2 = int(cols_a.min()), int(cols_a.max())
        return [
            (r1 / H, c1 / W), (r1 / H, c2 / W),
            (r2 / H, c2 / W), (r2 / H, c1 / W),
        ]
    except Exception as exc:
        logger.warning("[MIT-B2] Polygon extraction failed: %s", exc)
        return None


def _convex_hull_2d(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Graham scan convex hull over (row, col) integer points."""
    if len(points) < 3:
        return list(set(points))
    pts = list(set(points))
    pivot = min(pts, key=lambda p: (p[0], p[1]))

    def angle(p):
        return math.atan2(p[0] - pivot[0], p[1] - pivot[1])

    def dist_sq(a, b):
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

    sorted_pts = sorted(pts, key=lambda p: (angle(p), dist_sq(pivot, p)))
    hull = []
    for p in sorted_pts:
        while len(hull) >= 2:
            o, a = hull[-2], hull[-1]
            cross = (a[1] - o[1]) * (p[0] - o[0]) - (a[0] - o[0]) * (p[1] - o[1])
            if cross <= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    return hull


def _encode_mask_png(mask_np: np.ndarray) -> str:
    """
    Encode binary mask as a base64 RGBA PNG for overlay in the frontend.

    Strategy: mask pixels -> semi-transparent orange (R=255, G=140, B=0, A=180)
               background  -> fully transparent (A=0)
    """
    from PIL import Image

    H, W = mask_np.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    mask_bool = mask_np > 0.5
    rgba[mask_bool, 0] = 255   # R
    rgba[mask_bool, 1] = 140   # G
    rgba[mask_bool, 2] = 0     # B
    rgba[mask_bool, 3] = 180   # A (semi-transparent)

    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def segmentation_polygon_to_geographic(
    polygon_rel: List[Tuple[float, float]],
    spill_lat: float,
    spill_lon: float,
    image_lat_span: float = 0.15,   # degrees: approx lat extent of image
    image_lon_span: float = 0.15,   # degrees: approx lon extent of image
) -> List[Tuple[float, float]]:
    """
    Convert segmentation polygon from relative image coords to (lat, lon).

    This is an APPROXIMATE mapping.  Assumes the image is centered on
    (spill_lat, spill_lon) and covers (image_lat_span × image_lon_span) degrees.

    Row 0 = top = max_lat, Row 1 = bottom = min_lat.
    Col 0 = left = min_lon, Col 1 = right = max_lon.

    Parameters
    ----------
    polygon_rel : [(row_frac, col_frac), ...]
    spill_lat, spill_lon : scene center
    image_lat_span : total latitude degrees the image covers
    image_lon_span : total longitude degrees the image covers

    Returns
    -------
    [(lat, lon), ...]
    """
    min_lat = spill_lat - image_lat_span / 2.0
    max_lat = spill_lat + image_lat_span / 2.0
    min_lon = spill_lon - image_lon_span / 2.0
    max_lon = spill_lon + image_lon_span / 2.0

    geo_pts = []
    for (row_frac, col_frac) in polygon_rel:
        lat = max_lat - row_frac * (max_lat - min_lat)  # row 0 = top = max_lat
        lon = min_lon + col_frac * (max_lon - min_lon)
        geo_pts.append((round(lat, 6), round(lon, 6)))
    return geo_pts


def centroid_to_geographic(
    centroid_rel: Tuple[float, float],
    spill_lat: float,
    spill_lon: float,
    image_lat_span: float = 0.15,
    image_lon_span: float = 0.15,
) -> Tuple[float, float]:
    """Convert centroid (row_frac, col_frac) to (lat, lon)."""
    row_frac, col_frac = centroid_rel
    min_lat = spill_lat - image_lat_span / 2.0
    max_lat = spill_lat + image_lat_span / 2.0
    min_lon = spill_lon - image_lon_span / 2.0
    max_lon = spill_lon + image_lon_span / 2.0
    lat = max_lat - row_frac * (max_lat - min_lat)
    lon = min_lon + col_frac * (max_lon - min_lon)
    return round(lat, 6), round(lon, 6)
