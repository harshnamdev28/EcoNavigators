"""
Tests: MIT-B2 U-Net Inference Engine
=====================================
Tests for sar_pipeline/mit_b2_inference.py

These tests avoid loading the full 330 MB checkpoint by mocking the model.
The checkpoint-loading tests are marked with @pytest.mark.slow and can be
skipped on CI with: pytest -m "not slow"
"""

import io
import math
import os
import sys

import numpy as np
import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from sar_pipeline.mit_b2_inference import (
    SpillSegmentationResult,
    _build_result,
    _convex_hull_2d,
    _encode_mask_png,
    _extract_polygon_rel,
    _preprocess_for_mit_b2,
    _to_rgb_numpy,
    analyze_oil_spill,
    centroid_to_geographic,
    segmentation_polygon_to_geographic,
)


# ── Helper fixtures ───────────────────────────────────────────────────────────

def _make_rgb_image(h=256, w=256, r=200, g=100, b=50) -> np.ndarray:
    """Return a solid-color (H, W, 3) uint8 RGB array."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :, 0] = r
    arr[:, :, 1] = g
    arr[:, :, 2] = b
    return arr


def _make_jpeg_bytes(h=256, w=256) -> bytes:
    """Return a valid JPEG bytes object."""
    from PIL import Image
    img = Image.fromarray(_make_rgb_image(h, w), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _make_png_bytes(h=64, w=64) -> bytes:
    from PIL import Image
    img = Image.fromarray(_make_rgb_image(h, w), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── _to_rgb_numpy tests ───────────────────────────────────────────────────────

def test_to_rgb_numpy_from_bytes():
    jpg = _make_jpeg_bytes()
    arr = _to_rgb_numpy(jpg)
    assert arr.ndim == 3
    assert arr.shape[2] == 3
    assert arr.dtype == np.uint8


def test_to_rgb_numpy_from_pil():
    from PIL import Image
    img = Image.fromarray(_make_rgb_image(), mode="RGB")
    arr = _to_rgb_numpy(img)
    assert arr.shape == (256, 256, 3)


def test_to_rgb_numpy_from_numpy_uint8():
    raw = _make_rgb_image(64, 64)
    arr = _to_rgb_numpy(raw)
    assert arr.shape == (64, 64, 3)
    assert arr.dtype == np.uint8


def test_to_rgb_numpy_from_grayscale_numpy():
    gray = np.full((64, 64), 128, dtype=np.uint8)
    arr = _to_rgb_numpy(gray)
    assert arr.shape == (64, 64, 3)


def test_to_rgb_numpy_from_float_numpy():
    raw = np.random.rand(32, 32, 3).astype(np.float32)
    arr = _to_rgb_numpy(raw)
    assert arr.dtype == np.uint8


def test_to_rgb_numpy_invalid_type():
    with pytest.raises(TypeError):
        _to_rgb_numpy(12345)


# ── _preprocess_for_mit_b2 tests ──────────────────────────────────────────────

def test_preprocess_shape():
    import torch
    rgb = _make_rgb_image(128, 128)
    tensor = _preprocess_for_mit_b2(rgb)
    assert tensor.shape == (1, 3, 256, 256)
    assert tensor.dtype == torch.float32


def test_preprocess_normalized():
    """After ImageNet normalization, values should typically be in [-3, 3]."""
    rgb = _make_rgb_image(256, 256, r=127, g=127, b=127)
    tensor = _preprocess_for_mit_b2(rgb)
    assert float(tensor.abs().max()) < 5.0


def test_preprocess_different_input_sizes():
    import torch
    for h, w in [(64, 64), (512, 512), (100, 200)]:
        rgb = _make_rgb_image(h, w)
        t = _preprocess_for_mit_b2(rgb)
        assert t.shape == (1, 3, 256, 256), f"Shape mismatch for input ({h}, {w})"


# ── _build_result tests ───────────────────────────────────────────────────────

def test_build_result_no_spill():
    """Small number of positive pixels → POINT_ONLY, detected=False."""
    prob = np.zeros((256, 256), dtype=np.float32)
    mask = np.zeros((256, 256), dtype=np.float32)
    mask[10:12, 10:15] = 1.0   # 10 pixels < 50 minimum

    result = _build_result(prob, mask, pixel_resolution_m=31.25)
    assert result.detected is False
    assert result.geometry_source == "POINT_ONLY"
    assert result.spill_pixel_count < 50


def test_build_result_with_spill():
    """Large spill patch → detected=True, IMAGE_DERIVED."""
    H, W = 256, 256
    prob = np.zeros((H, W), dtype=np.float32)
    mask = np.zeros((H, W), dtype=np.float32)
    # 100×50 = 5000 pixels, well above threshold
    prob[50:150, 100:200] = 0.85
    mask[50:150, 100:200] = 1.0

    result = _build_result(prob, mask, pixel_resolution_m=31.25)
    assert result.detected is True
    assert result.geometry_source == "IMAGE_DERIVED"
    assert result.spill_pixel_count == 100 * 100
    assert result.spill_fraction == pytest.approx(100 * 100 / (H * W), rel=0.01)


def test_build_result_centroid_correct():
    """Centroid should be at centre of rectangular spill region."""
    H, W = 256, 256
    prob = np.full((H, W), 0.85, dtype=np.float32)
    mask = np.zeros((H, W), dtype=np.float32)
    mask[100:156, 100:156] = 1.0   # 56×56 square centred at (128, 128)

    result = _build_result(prob, mask, pixel_resolution_m=31.25)
    assert result.detected is True
    cy_rel, cx_rel = result.centroid_rel
    # Centroid pixel: row ≈ 127.5/256, col ≈ 127.5/256
    assert cy_rel == pytest.approx(127.5 / 256, abs=0.01)
    assert cx_rel == pytest.approx(127.5 / 256, abs=0.01)


def test_build_result_bounding_box():
    H, W = 256, 256
    prob = np.full((H, W), 0.8, dtype=np.float32)
    mask = np.zeros((H, W), dtype=np.float32)
    mask[20:80, 30:90] = 1.0

    result = _build_result(prob, mask, pixel_resolution_m=31.25)
    x1, y1, x2, y2 = result.bounding_box_px
    assert x1 == 30
    assert y1 == 20
    assert x2 == 89
    assert y2 == 79


def test_build_result_area_km2():
    """Area in km² = pixel_count * (res_m / 1000)^2."""
    H, W = 256, 256
    prob = np.full((H, W), 0.9, dtype=np.float32)
    mask = np.zeros((H, W), dtype=np.float32)
    mask[0:100, 0:100] = 1.0

    res_m = 50.0
    result = _build_result(prob, mask, pixel_resolution_m=res_m)
    expected_km2 = 100 * 100 * (res_m / 1000.0) ** 2
    assert result.spill_area_km2 == pytest.approx(expected_km2, rel=0.01)


def test_build_result_mask_png_is_valid_base64():
    import base64
    mask = np.zeros((256, 256), dtype=np.float32)
    mask[50:150, 50:150] = 1.0
    b64 = _encode_mask_png(mask)
    raw = base64.b64decode(b64)
    # PNG magic bytes
    assert raw[:4] == b'\x89PNG'


# ── _extract_polygon_rel tests ────────────────────────────────────────────────

def test_extract_polygon_rel_returns_list():
    mask = np.zeros((256, 256), dtype=np.float32)
    mask[50:200, 50:200] = 1.0
    poly = _extract_polygon_rel(mask, 256, 256)
    assert poly is not None
    assert isinstance(poly, list)
    assert len(poly) >= 3


def test_extract_polygon_rel_values_in_0_1():
    mask = np.zeros((256, 256), dtype=np.float32)
    mask[10:246, 10:246] = 1.0
    poly = _extract_polygon_rel(mask, 256, 256)
    for row_frac, col_frac in poly:
        assert 0.0 <= row_frac <= 1.0
        assert 0.0 <= col_frac <= 1.0


def test_extract_polygon_rel_empty_mask():
    mask = np.zeros((256, 256), dtype=np.float32)
    poly = _extract_polygon_rel(mask, 256, 256)
    assert poly is None


# ── segmentation_polygon_to_geographic tests ──────────────────────────────────

def test_segmentation_polygon_to_geographic_center():
    """Centroid of image at (0.5, 0.5) should map to scene center."""
    lat, lon = 25.0, -80.0
    poly_rel = [(0.5, 0.5)]
    geo = segmentation_polygon_to_geographic(poly_rel, lat, lon,
                                              image_lat_span=0.15, image_lon_span=0.15)
    assert geo[0][0] == pytest.approx(lat, abs=0.001)
    assert geo[0][1] == pytest.approx(lon, abs=0.001)


def test_segmentation_polygon_to_geographic_top_left():
    """(0.0, 0.0) relative = top-left = (max_lat, min_lon)."""
    lat, lon = 25.0, -80.0
    poly_rel = [(0.0, 0.0)]
    geo = segmentation_polygon_to_geographic(poly_rel, lat, lon,
                                              image_lat_span=0.10, image_lon_span=0.10)
    expected_lat = lat + 0.05   # max_lat
    expected_lon = lon - 0.05   # min_lon
    assert geo[0][0] == pytest.approx(expected_lat, abs=0.001)
    assert geo[0][1] == pytest.approx(expected_lon, abs=0.001)


def test_centroid_to_geographic_center():
    lat, lon = 10.0, 50.0
    result_lat, result_lon = centroid_to_geographic(
        (0.5, 0.5), lat, lon, image_lat_span=0.20, image_lon_span=0.20
    )
    assert result_lat == pytest.approx(lat, abs=0.001)
    assert result_lon == pytest.approx(lon, abs=0.001)


# ── _convex_hull_2d tests ─────────────────────────────────────────────────────

def test_convex_hull_2d_square():
    """Hull of a square's corners should return 4 points."""
    pts = [(0, 0), (0, 10), (10, 0), (10, 10), (5, 5)]
    hull = _convex_hull_2d(pts)
    assert len(hull) >= 3


def test_convex_hull_2d_collinear():
    """Collinear points — at least 2 points returned."""
    pts = [(0, 0), (0, 5), (0, 10)]
    hull = _convex_hull_2d(pts)
    assert len(hull) >= 2


# ── SpillSegmentationResult dataclass tests ───────────────────────────────────

def test_segmentation_result_defaults():
    r = SpillSegmentationResult()
    assert r.detected is False
    assert r.geometry_source == "POINT_ONLY"
    assert r.spill_pixel_count == 0
    assert r.model_name == "MIT-B2-U-Net"
    assert r.error is None


def test_segmentation_result_error_field():
    r = SpillSegmentationResult(error="model not found", reason="Model load failed")
    assert r.error == "model not found"
    assert r.detected is False


# ── analyze_oil_spill integration (mocked model) ──────────────────────────────

def test_analyze_oil_spill_bad_bytes():
    """Corrupted bytes should return error result, not raise."""
    result = analyze_oil_spill(b"not an image")
    assert result.error is not None
    assert result.detected is False


def test_analyze_oil_spill_valid_image_no_checkpoint(monkeypatch):
    """
    With a mocked model (all-negative logits → no spill), analyze_oil_spill should return
    a SpillSegmentationResult with detected=False without crashing.
    sigmoid(-10.0) ≈ 0.000045, well below the 0.45 detection threshold.
    """
    import torch
    import torch.nn as nn

    class _FakeModel(nn.Module):
        def forward(self, x):
            B, C, H, W = x.shape
            # Strongly negative logits → sigmoid ≈ 0 → no spill detected
            return torch.full((B, 1, H, W), -10.0)

    from sar_pipeline import mit_b2_inference as mi
    engine = mi.MiTB2InferenceEngine()
    engine._model = _FakeModel()
    engine._loaded = True
    engine._device = "cpu"
    mi.MiTB2InferenceEngine._singleton = engine

    jpg = _make_jpeg_bytes(256, 256)
    result = engine.analyze(jpg)
    assert isinstance(result, mi.SpillSegmentationResult)
    assert result.error is None
    assert result.detected is False   # sigmoid(-10) ≈ 0 → no spill

    # Reset singleton for other tests
    mi.MiTB2InferenceEngine._singleton = None


def test_analyze_oil_spill_detects_with_mocked_positive_model(monkeypatch):
    """
    With a model returning all-ones logits (high probability everywhere),
    detection should fire and IMAGE_DERIVED geometry should be set.
    """
    import torch
    import torch.nn as nn

    class _PositiveModel(nn.Module):
        def forward(self, x):
            B, C, H, W = x.shape
            return torch.ones(B, 1, H, W) * 5.0   # sigmoid(5) >> 0.45 threshold

    from sar_pipeline import mit_b2_inference as mi
    engine = mi.MiTB2InferenceEngine()
    engine._model = _PositiveModel()
    engine._loaded = True
    engine._device = "cpu"
    mi.MiTB2InferenceEngine._singleton = engine

    jpg = _make_jpeg_bytes(256, 256)
    result = engine.analyze(jpg)
    assert result.detected is True
    assert result.geometry_source == "IMAGE_DERIVED"
    assert result.spill_pixel_count == 256 * 256
    assert result.spill_fraction == pytest.approx(1.0, abs=0.001)

    mi.MiTB2InferenceEngine._singleton = None
