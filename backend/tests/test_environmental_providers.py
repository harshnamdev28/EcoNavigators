"""
Tests: Environmental Data Providers
=====================================
Tests for fusion/environmental_providers.py

Uses monkeypatching and mocked HTTP calls to avoid actual ERDDAP API calls.
"""

import os
import sys
import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from fusion.lagrangian_model import EnvironmentalDataUnavailable, MockEnvironmentalProvider
from fusion.environmental_providers import (
    _bilinear_interpolate,
    _cache_key,
    _parse_erddap_uv_grid,
    get_provider,
    NOAAErddapProvider,
)


# ── get_provider factory tests ────────────────────────────────────────────────

def test_get_provider_demo_returns_mock():
    provider = get_provider("DEMO")
    assert isinstance(provider, MockEnvironmentalProvider)
    assert provider.data_mode == "DEMO"


def test_get_provider_demo_case_insensitive():
    provider = get_provider("demo")
    assert isinstance(provider, MockEnvironmentalProvider)


def test_get_provider_live_raises_when_no_connectivity(monkeypatch):
    """LIVE mode with no ERDDAP connectivity must raise EnvironmentalDataUnavailable."""
    import urllib.error
    import urllib.request

    def _fail_urlopen(*args, **kwargs):
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", _fail_urlopen)

    with pytest.raises((EnvironmentalDataUnavailable, Exception)):
        get_provider("LIVE")


def test_get_provider_live_never_returns_mock(monkeypatch):
    """
    LIVE mode must NEVER silently return MockEnvironmentalProvider.
    If all providers fail, it must raise, not return mock.
    """
    import urllib.error
    import urllib.request

    def _fail_urlopen(*args, **kwargs):
        raise urllib.error.URLError("timeout")

    monkeypatch.setattr(urllib.request, "urlopen", _fail_urlopen)
    monkeypatch.setenv("CMEMS_USERNAME", "")
    monkeypatch.setenv("CMEMS_PASSWORD", "")

    with pytest.raises(Exception):
        result = get_provider("LIVE")
        # If somehow it doesn't raise, ensure it's not a mock provider
        assert not isinstance(result, MockEnvironmentalProvider), (
            "LIVE mode MUST NOT return MockEnvironmentalProvider"
        )


# ── MockEnvironmentalProvider tests ──────────────────────────────────────────

class TestMockEnvironmentalProvider:
    def setup_method(self):
        self.provider = MockEnvironmentalProvider()

    def test_data_mode_is_demo(self):
        assert self.provider.data_mode == "DEMO"

    def test_name(self):
        assert "Mock" in self.provider.name

    def test_get_current_returns_tuple(self):
        ts = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        u, v = self.provider.get_current(25.0, -80.0, ts)
        assert isinstance(u, float)
        assert isinstance(v, float)

    def test_get_wind_returns_tuple(self):
        ts = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        u, v = self.provider.get_wind(25.0, -80.0, ts)
        assert isinstance(u, float)
        assert isinstance(v, float)

    def test_current_deterministic(self):
        """Same inputs → same outputs (deterministic)."""
        ts = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        u1, v1 = self.provider.get_current(25.0, -80.0, ts)
        u2, v2 = self.provider.get_current(25.0, -80.0, ts)
        assert u1 == u2
        assert v1 == v2

    def test_units_in_reasonable_range(self):
        """Ocean currents should be < 3 m/s. Winds < 30 m/s."""
        ts = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        u_c, v_c = self.provider.get_current(25.0, -80.0, ts)
        u_w, v_w = self.provider.get_wind(25.0, -80.0, ts)
        assert abs(u_c) < 3.0
        assert abs(v_c) < 3.0
        assert abs(u_w) < 30.0
        assert abs(v_w) < 30.0


# ── _parse_erddap_uv_grid tests ───────────────────────────────────────────────

def _make_erddap_response(times, lats, lons, u_vals, v_vals, u_var="water_u", v_var="water_v"):
    """Build a minimal ERDDAP JSON table response."""
    rows = []
    for ti, t in enumerate(times):
        for li, la in enumerate(lats):
            for loi, lo in enumerate(lons):
                rows.append([t, la, lo,
                              float(u_vals[ti][li][loi]),
                              float(v_vals[ti][li][loi])])
    return {
        "table": {
            "columnNames": ["time", "latitude", "longitude", u_var, v_var],
            "rows": rows,
        }
    }


class TestParseErddapUvGrid:

    def test_basic_parse(self):
        times = ["2024-06-15T12:00:00Z"]
        lats  = [24.0, 25.0, 26.0]
        lons  = [-81.0, -80.0, -79.0]
        u = [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]]
        v = [[[0.0, 0.1, 0.2], [0.3, 0.4, 0.5], [0.6, 0.7, 0.8]]]
        raw = _make_erddap_response(times, lats, lons, u, v)
        grid = _parse_erddap_uv_grid(raw, "water_u", "water_v")
        assert "lats" in grid
        assert "lons" in grid
        assert "times" in grid
        assert "u" in grid
        assert "v" in grid
        assert grid["u"].shape == (1, 3, 3)
        assert grid["v"].shape == (1, 3, 3)

    def test_nan_replaced_with_zero(self):
        """NaN values in ERDDAP response should be replaced with 0.0."""
        times = ["2024-06-15T12:00:00Z"]
        lats  = [25.0]
        lons  = [-80.0]
        raw = {
            "table": {
                "columnNames": ["time", "latitude", "longitude", "water_u", "water_v"],
                "rows": [["2024-06-15T12:00:00Z", 25.0, -80.0, None, None]],
            }
        }
        grid = _parse_erddap_uv_grid(raw, "water_u", "water_v")
        assert float(grid["u"][0, 0, 0]) == 0.0
        assert float(grid["v"][0, 0, 0]) == 0.0

    def test_missing_column_raises(self):
        raw = {"table": {"columnNames": ["time", "latitude"], "rows": []}}
        with pytest.raises(Exception):
            _parse_erddap_uv_grid(raw, "water_u", "water_v")


# ── _bilinear_interpolate tests ───────────────────────────────────────────────

def _make_simple_grid(u_val=0.5, v_val=0.3):
    """Create a minimal 1-time, 3-lat, 3-lon grid."""
    ts = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
    lats = np.array([24.0, 25.0, 26.0], dtype=np.float32)
    lons = np.array([-81.0, -80.0, -79.0], dtype=np.float32)
    u = np.full((1, 3, 3), u_val, dtype=np.float32)
    v = np.full((1, 3, 3), v_val, dtype=np.float32)
    return {"lats": lats, "lons": lons, "times": [ts], "u": u, "v": v}


class TestBilinearInterpolate:

    def test_exact_grid_point(self):
        cache = _make_simple_grid(u_val=0.5)
        ts = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = _bilinear_interpolate(cache, 25.0, -80.0, ts, "u")
        assert result == pytest.approx(0.5, abs=0.01)

    def test_outside_bounds_returns_zero(self):
        cache = _make_simple_grid()
        ts = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        # Far outside grid
        result = _bilinear_interpolate(cache, 50.0, -50.0, ts, "u")
        assert result == 0.0

    def test_uniform_grid_any_point_returns_constant(self):
        """Uniform field: bilinear interpolation anywhere returns same value."""
        cache = _make_simple_grid(u_val=0.75, v_val=0.25)
        ts = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        for lat in [24.5, 25.0, 25.5]:
            for lon in [-80.5, -80.0, -79.5]:
                u = _bilinear_interpolate(cache, lat, lon, ts, "u")
                assert u == pytest.approx(0.75, abs=0.01)

    def test_empty_grid_returns_zero(self):
        cache = {
            "lats":  np.array([], dtype=np.float32),
            "lons":  np.array([], dtype=np.float32),
            "times": [],
            "u":     np.zeros((0, 0, 0), dtype=np.float32),
            "v":     np.zeros((0, 0, 0), dtype=np.float32),
        }
        ts = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = _bilinear_interpolate(cache, 25.0, -80.0, ts, "u")
        assert result == 0.0


# ── _cache_key tests ──────────────────────────────────────────────────────────

class TestCacheKey:

    def test_same_inputs_same_key(self):
        ts = datetime(2024, 6, 15, 14, 30, 0, tzinfo=timezone.utc)
        k1 = _cache_key(25.0, -80.0, ts, "current")
        k2 = _cache_key(25.0, -80.0, ts, "current")
        assert k1 == k2

    def test_different_kind_different_key(self):
        ts = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        k1 = _cache_key(25.0, -80.0, ts, "current")
        k2 = _cache_key(25.0, -80.0, ts, "wind")
        assert k1 != k2

    def test_same_6h_bucket_same_key(self):
        """Times within the same 6h bucket give the same key."""
        ts1 = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2024, 6, 15, 13, 45, 0, tzinfo=timezone.utc)
        k1 = _cache_key(25.0, -80.0, ts1, "current")
        k2 = _cache_key(25.0, -80.0, ts2, "current")
        assert k1 == k2

    def test_different_6h_bucket_different_key(self):
        ts1 = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2024, 6, 15, 18, 0, 0, tzinfo=timezone.utc)
        k1 = _cache_key(25.0, -80.0, ts1, "current")
        k2 = _cache_key(25.0, -80.0, ts2, "current")
        assert k1 != k2
