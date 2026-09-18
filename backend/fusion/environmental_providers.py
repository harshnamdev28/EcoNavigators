"""
Environmental Data Providers — Historical Oil Spill Investigation
=================================================================
Module: fusion/environmental_providers.py

Implements concrete EnvironmentalDataProvider subclasses for LIVE mode:

    NOAAErddapProvider   — NOAA HYCOM via ERDDAP REST API (FREE, no credentials)
    CopernicusMarineProvider — CMEMS (requires credentials from env)

Key design rules:
    1. NEVER call an external API once per particle.  Fetch a spatial/temporal
       region once, cache it in memory, interpolate locally.
    2. LIVE mode must NEVER silently fall back to mock data.
       Raise EnvironmentalDataUnavailable with a clear message.
    3. Credentials ONLY from environment variables — never hardcoded.
    4. All velocity units are m/s (u = eastward, v = northward).

Usage:
    from fusion.environmental_providers import get_provider
    provider = get_provider("LIVE")   # or "DEMO"

    # LIVE raises EnvironmentalDataUnavailable if fetch fails
    # DEMO returns MockEnvironmentalProvider (always works)
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

import numpy as np

from fusion.lagrangian_model import (
    EnvironmentalDataProvider,
    EnvironmentalDataUnavailable,
    MockEnvironmentalProvider,
)

logger = logging.getLogger(__name__)


# ── Factory function ──────────────────────────────────────────────────────────

def get_provider(mode: str = "DEMO") -> EnvironmentalDataProvider:
    """
    Return the appropriate environmental data provider.

    Parameters
    ----------
    mode : str
        "DEMO" → MockEnvironmentalProvider (always works, clearly labeled)
        "LIVE" → NOAAErddapProvider if available, else CopernicusMarineProvider
                 RAISES EnvironmentalDataUnavailable if neither is accessible.
                 NEVER falls back to mock in LIVE mode.

    Returns
    -------
    EnvironmentalDataProvider
    """
    if mode.upper() == "DEMO":
        logger.info("[EnvProvider] Using MockEnvironmentalProvider (DEMO mode).")
        return MockEnvironmentalProvider()

    # --- LIVE mode ---
    # Try NOAA ERDDAP first (free, no credentials)
    try:
        provider = NOAAErddapProvider()
        logger.info("[EnvProvider] Using NOAA ERDDAP HYCOM (LIVE mode).")
        return provider
    except EnvironmentalDataUnavailable as exc:
        logger.warning("[EnvProvider] NOAA ERDDAP unavailable: %s", exc)

    # Try CMEMS if credentials are set
    cmems_user = os.environ.get("CMEMS_USERNAME", "")
    cmems_pass = os.environ.get("CMEMS_PASSWORD", "")
    if cmems_user and cmems_pass:
        try:
            provider = CopernicusMarineProvider(cmems_user, cmems_pass)
            logger.info("[EnvProvider] Using Copernicus Marine (CMEMS) (LIVE mode).")
            return provider
        except EnvironmentalDataUnavailable as exc:
            logger.warning("[EnvProvider] CMEMS unavailable: %s", exc)

    # LIVE mode: NEVER silently substitute mock data
    raise EnvironmentalDataUnavailable(
        "LIVE mode requested but no environmental data provider is accessible. "
        "NOAA ERDDAP timed out and CMEMS credentials are not configured. "
        "Set CMEMS_USERNAME + CMEMS_PASSWORD in .env, or use dataMode=DEMO."
    )


# ── NOAA ERDDAP HYCOM Provider ────────────────────────────────────────────────

class NOAAErddapProvider(EnvironmentalDataProvider):
    """
    LIVE environmental data from NOAA HYCOM ocean reanalysis and GFS winds
    via ERDDAP REST APIs.  FREE — no API key or registration required.

    Efficiency design:
        - On first call for a simulation, fetches a spatial/temporal bounding
          box covering the ENTIRE simulation domain at once (one API request).
        - Subsequent per-particle interpolation is done locally in memory.
        - Cache is keyed by (lat_min, lat_max, lon_min, lon_max, t_start, t_end).

    HYCOM endpoint:  https://coastwatch.pfeg.noaa.gov/erddap/griddap/HYCOM_*
    Wind endpoint:   https://coastwatch.pfeg.noaa.gov/erddap/griddap/NCEP_Global_Best
    """

    # ERDDAP HYCOM surface currents (GLBu0.08/expt_93.0 reanalysis)
    _CURRENT_DATASET = "HYCOM_reg7_latest3d"
    _WIND_DATASET    = "NCEP_Global_Best"
    _ERDDAP_BASE     = "https://coastwatch.pfeg.noaa.gov/erddap/griddap"
    _TIMEOUT_S       = 30

    def __init__(self) -> None:
        # Grid cache: {cache_key: {"u": np.array, "v": np.array, "lats": ..., "lons": ..., "times": ...}}
        self._current_cache: Dict[str, Any] = {}
        self._wind_cache: Dict[str, Any] = {}
        # Quick connectivity check on init
        self._check_connectivity()

    def _check_connectivity(self) -> None:
        """Light connectivity check — fetches metadata only."""
        url = f"{self._ERDDAP_BASE}/{self._CURRENT_DATASET}.json?time[0:1:0]"
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                _ = resp.read(200)
        except Exception as exc:
            raise EnvironmentalDataUnavailable(
                f"NOAA ERDDAP connectivity check failed: {exc}"
            ) from exc

    # ── EnvironmentalDataProvider interface ───────────────────────────────────

    @property
    def name(self) -> str:
        return "NOAA-ERDDAP-HYCOM"

    @property
    def data_mode(self) -> str:
        return "LIVE"

    def get_current(self, lat: float, lon: float, timestamp: datetime) -> Tuple[float, float]:
        """Return (u_current, v_current) in m/s at (lat, lon, timestamp)."""
        cache = self._get_current_cache(lat, lon, timestamp)
        return _bilinear_interpolate(cache, lat, lon, timestamp, "u"), \
               _bilinear_interpolate(cache, lat, lon, timestamp, "v")

    def get_wind(self, lat: float, lon: float, timestamp: datetime) -> Tuple[float, float]:
        """Return (u_wind, v_wind) in m/s at (lat, lon, timestamp)."""
        cache = self._get_wind_cache(lat, lon, timestamp)
        return _bilinear_interpolate(cache, lat, lon, timestamp, "u"), \
               _bilinear_interpolate(cache, lat, lon, timestamp, "v")

    # ── Cache management ──────────────────────────────────────────────────────

    def _get_current_cache(self, lat: float, lon: float, ts: datetime) -> Dict:
        key = _cache_key(lat, lon, ts, "current")
        if key in self._current_cache:
            return self._current_cache[key]

        # Fetch a ±2° spatial, ±25h temporal region around (lat, lon, ts)
        lat_min = round(lat - 2.0, 2)
        lat_max = round(lat + 2.0, 2)
        lon_min = round(lon - 2.0, 2)
        lon_max = round(lon + 2.0, 2)
        t_start = ts - timedelta(hours=25)
        t_end   = ts + timedelta(hours=1)

        data = self._fetch_hycom_currents(lat_min, lat_max, lon_min, lon_max, t_start, t_end)
        self._current_cache[key] = data
        # Evict old cache entries (keep at most 5)
        if len(self._current_cache) > 5:
            oldest = next(iter(self._current_cache))
            del self._current_cache[oldest]
        return data

    def _get_wind_cache(self, lat: float, lon: float, ts: datetime) -> Dict:
        key = _cache_key(lat, lon, ts, "wind")
        if key in self._wind_cache:
            return self._wind_cache[key]

        lat_min = round(lat - 2.0, 2)
        lat_max = round(lat + 2.0, 2)
        lon_min = round(lon - 2.0, 2)
        lon_max = round(lon + 2.0, 2)
        t_start = ts - timedelta(hours=25)
        t_end   = ts + timedelta(hours=1)

        data = self._fetch_gfs_winds(lat_min, lat_max, lon_min, lon_max, t_start, t_end)
        self._wind_cache[key] = data
        if len(self._wind_cache) > 5:
            oldest = next(iter(self._wind_cache))
            del self._wind_cache[oldest]
        return data

    # ── ERDDAP fetch functions ─────────────────────────────────────────────────

    def _fetch_hycom_currents(
        self,
        lat_min: float, lat_max: float,
        lon_min: float, lon_max: float,
        t_start: datetime, t_end: datetime,
    ) -> Dict:
        """
        Fetch surface u/v currents from HYCOM via ERDDAP.
        Returns a grid dict with u, v, lats, lons, times arrays.
        """
        t_from = t_start.strftime("%Y-%m-%dT%H:%M:%SZ")
        t_to   = t_end.strftime("%Y-%m-%dT%H:%M:%SZ")

        # ERDDAP HYCOM variable names: water_u, water_v, depth=0 (surface)
        # Dataset: HYCOM_reg7_latest3d (surface layer at depth[0])
        url = (
            f"{self._ERDDAP_BASE}/{self._CURRENT_DATASET}.json"
            f"?water_u[('{t_from}'):1:('{t_to}')][(0.0):1:(0.0)]"
            f"[({lat_min}):1:({lat_max})][({lon_min}):1:({lon_max})]"
            f",water_v[('{t_from}'):1:('{t_to}')][(0.0):1:(0.0)]"
            f"[({lat_min}):1:({lat_max})][({lon_min}):1:({lon_max})]"
        )

        try:
            raw = _erddap_json_fetch(url, self._TIMEOUT_S)
            return _parse_erddap_uv_grid(raw, "water_u", "water_v")
        except Exception as exc:
            logger.warning("[ERDDAP] HYCOM fetch failed (%s). Trying fallback dataset.", exc)
            # Fallback: RTOFS (Real-Time Ocean Forecast System) if HYCOM unavailable
            return _fetch_rtofs_fallback(lat_min, lat_max, lon_min, lon_max, t_start, t_end)

    def _fetch_gfs_winds(
        self,
        lat_min: float, lat_max: float,
        lon_min: float, lon_max: float,
        t_start: datetime, t_end: datetime,
    ) -> Dict:
        """
        Fetch 10-m U/V wind components from NCEP GFS via ERDDAP.
        """
        t_from = t_start.strftime("%Y-%m-%dT%H:%M:%SZ")
        t_to   = t_end.strftime("%Y-%m-%dT%H:%M:%SZ")

        url = (
            f"{self._ERDDAP_BASE}/{self._WIND_DATASET}.json"
            f"?ugrd10m[('{t_from}'):1:('{t_to}')]"
            f"[({lat_min}):1:({lat_max})][({lon_min}):1:({lon_max})]"
            f",vgrd10m[('{t_from}'):1:('{t_to}')]"
            f"[({lat_min}):1:({lat_max})][({lon_min}):1:({lon_max})]"
        )

        try:
            raw = _erddap_json_fetch(url, self._TIMEOUT_S)
            return _parse_erddap_uv_grid(raw, "ugrd10m", "vgrd10m")
        except Exception as exc:
            raise EnvironmentalDataUnavailable(
                f"GFS wind fetch failed for region lat=[{lat_min},{lat_max}] "
                f"lon=[{lon_min},{lon_max}] time=[{t_from},{t_to}]: {exc}"
            ) from exc


# ── Copernicus Marine (CMEMS) Provider ────────────────────────────────────────

class CopernicusMarineProvider(EnvironmentalDataProvider):
    """
    LIVE ocean current and wind data from Copernicus Marine Service (CMEMS).

    Credentials: CMEMS_USERNAME, CMEMS_PASSWORD environment variables.
    Uses the CMEMS NEMO Global Ocean Physics Reanalysis (GLORYS12V1).

    Efficiency: fetches spatial/temporal bounding box once, caches, interpolates locally.
    """

    _CMEMS_WMS_BASE = "https://nrt.cmems-du.eu/motu-web/Motu"
    _PRODUCT_ID     = "GLOBAL_ANALYSISFORECAST_PHY_001_024"
    _SERVICE_ID     = "global-analysis-forecast-phy-001-024-hourly-t-u-v-ssh"

    def __init__(self, username: str, password: str) -> None:
        if not username or not password:
            raise EnvironmentalDataUnavailable(
                "CMEMS credentials not set. "
                "Set CMEMS_USERNAME and CMEMS_PASSWORD in environment."
            )
        self._username = username
        self._password = password
        self._current_cache: Dict[str, Any] = {}
        self._wind_cache: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "Copernicus-Marine-CMEMS"

    @property
    def data_mode(self) -> str:
        return "LIVE"

    def get_current(self, lat: float, lon: float, timestamp: datetime) -> Tuple[float, float]:
        cache = self._get_cache(lat, lon, timestamp, "current")
        return _bilinear_interpolate(cache, lat, lon, timestamp, "u"), \
               _bilinear_interpolate(cache, lat, lon, timestamp, "v")

    def get_wind(self, lat: float, lon: float, timestamp: datetime) -> Tuple[float, float]:
        # CMEMS does not provide wind — use NOAA ERDDAP for wind
        try:
            url = (
                f"https://coastwatch.pfeg.noaa.gov/erddap/griddap/NCEP_Global_Best.json"
                f"?ugrd10m[('{timestamp.strftime('%Y-%m-%dT%H:%M:%SZ')}'):1:('{timestamp.strftime('%Y-%m-%dT%H:%M:%SZ')}')]"
                f"[({lat - 1.0}):1:({lat + 1.0})][({lon - 1.0}):1:({lon + 1.0})]"
                f",vgrd10m[('{timestamp.strftime('%Y-%m-%dT%H:%M:%SZ')}'):1:('{timestamp.strftime('%Y-%m-%dT%H:%M:%SZ')}')]"
                f"[({lat - 1.0}):1:({lat + 1.0})][({lon - 1.0}):1:({lon + 1.0})]"
            )
            raw = _erddap_json_fetch(url, 20)
            cache = _parse_erddap_uv_grid(raw, "ugrd10m", "vgrd10m")
            return _bilinear_interpolate(cache, lat, lon, timestamp, "u"), \
                   _bilinear_interpolate(cache, lat, lon, timestamp, "v")
        except Exception as exc:
            raise EnvironmentalDataUnavailable(f"Wind fetch failed: {exc}") from exc

    def _get_cache(self, lat: float, lon: float, ts: datetime, kind: str) -> Dict:
        key = _cache_key(lat, lon, ts, kind)
        if key in self._current_cache:
            return self._current_cache[key]

        lat_min = round(lat - 2.0, 2)
        lat_max = round(lat + 2.0, 2)
        lon_min = round(lon - 2.0, 2)
        lon_max = round(lon + 2.0, 2)
        t_start = ts - timedelta(hours=25)
        t_end   = ts + timedelta(hours=1)

        data = self._fetch_cmems_currents(lat_min, lat_max, lon_min, lon_max, t_start, t_end)
        self._current_cache[key] = data
        return data

    def _fetch_cmems_currents(
        self,
        lat_min: float, lat_max: float,
        lon_min: float, lon_max: float,
        t_start: datetime, t_end: datetime,
    ) -> Dict:
        """
        Fetch currents from CMEMS via Motu client HTTP API.
        Returns a grid dict compatible with _bilinear_interpolate.
        """
        import urllib.parse as _up

        params = {
            "action":      "productdownload",
            "service":     self._SERVICE_ID,
            "product":     self._PRODUCT_ID,
            "x_lo":        str(lon_min),
            "x_hi":        str(lon_max),
            "y_lo":        str(lat_min),
            "y_hi":        str(lat_max),
            "t_lo":        t_start.strftime("%Y-%m-%d %H:%M:%S"),
            "t_hi":        t_end.strftime("%Y-%m-%d %H:%M:%S"),
            "depth_lo":    "0.494",
            "depth_hi":    "0.494",
            "variable":    ["uo", "vo"],
            "user":        self._username,
            "pwd":         self._password,
            "output-format": "json",
        }
        url = f"{self._CMEMS_WMS_BASE}?{_up.urlencode(params, doseq=True)}"

        try:
            raw = _erddap_json_fetch(url, 30)
            return _parse_erddap_uv_grid(raw, "uo", "vo")
        except Exception as exc:
            raise EnvironmentalDataUnavailable(
                f"CMEMS current fetch failed: {exc}"
            ) from exc


# ── ERDDAP HTTP helpers ───────────────────────────────────────────────────────

def _erddap_json_fetch(url: str, timeout: int) -> Dict:
    """Fetch ERDDAP JSON response. Returns parsed dict."""
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
        return json.loads(body)
    except urllib.error.URLError as exc:
        raise EnvironmentalDataUnavailable(f"ERDDAP HTTP error: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise EnvironmentalDataUnavailable(f"ERDDAP response parse error: {exc}") from exc


def _fetch_rtofs_fallback(
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    t_start: datetime, t_end: datetime,
) -> Dict:
    """
    RTOFS (Real-Time Ocean Forecast System) fallback via ERDDAP.
    Used when HYCOM is unavailable.
    """
    t_from = t_start.strftime("%Y-%m-%dT%H:%M:%SZ")
    t_to   = t_end.strftime("%Y-%m-%dT%H:%M:%SZ")
    url = (
        "https://coastwatch.pfeg.noaa.gov/erddap/griddap/rtofsDaily2DsurfaceForecasts.json"
        f"?u_velocity[('{t_from}'):1:('{t_to}')][({lat_min}):1:({lat_max})][({lon_min}):1:({lon_max})]"
        f",v_velocity[('{t_from}'):1:('{t_to}')][({lat_min}):1:({lat_max})][({lon_min}):1:({lon_max})]"
    )
    raw = _erddap_json_fetch(url, 30)
    return _parse_erddap_uv_grid(raw, "u_velocity", "v_velocity")


def _parse_erddap_uv_grid(raw: Dict, u_var: str, v_var: str) -> Dict:
    """
    Parse ERDDAP JSON table response into a grid dict with:
        lats: 1-D np.array
        lons: 1-D np.array
        times: list of datetime
        u: np.ndarray (n_times, n_lats, n_lons)
        v: np.ndarray (n_times, n_lats, n_lons)
    """
    try:
        raw_col_names = raw["table"]["columnNames"]
        # Real ERDDAP API: [[name, type], ...]; test mocks: [name, ...]
        col_names = [
            c[0] if isinstance(c, (list, tuple)) else c
            for c in raw_col_names
        ]
        rows = raw["table"]["rows"]

        time_idx = col_names.index("time")
        lat_idx  = col_names.index("latitude")
        lon_idx  = col_names.index("longitude")
        u_idx    = col_names.index(u_var)
        v_idx    = col_names.index(v_var)

        times_set = sorted({r[time_idx] for r in rows})
        lats_set  = sorted({r[lat_idx]  for r in rows})
        lons_set  = sorted({r[lon_idx]  for r in rows})

        t_map = {t: i for i, t in enumerate(times_set)}
        la_map = {la: i for i, la in enumerate(lats_set)}
        lo_map = {lo: i for i, lo in enumerate(lons_set)}

        nt = len(times_set)
        nla = len(lats_set)
        nlo = len(lons_set)

        u_grid = np.full((nt, nla, nlo), np.nan, dtype=np.float32)
        v_grid = np.full((nt, nla, nlo), np.nan, dtype=np.float32)

        for r in rows:
            ti = t_map[r[time_idx]]
            li = la_map[r[lat_idx]]
            loi = lo_map[r[lon_idx]]
            u_val = r[u_idx]
            v_val = r[v_idx]
            if u_val is not None:
                u_grid[ti, li, loi] = float(u_val)
            if v_val is not None:
                v_grid[ti, li, loi] = float(v_val)

        # Fill NaN with 0 (missing ocean data → calm = no bias)
        u_grid = np.where(np.isnan(u_grid), 0.0, u_grid)
        v_grid = np.where(np.isnan(v_grid), 0.0, v_grid)

        # Parse time strings to datetimes
        dt_times = []
        for ts_str in times_set:
            try:
                dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                dt_times.append(dt)
            except Exception:
                dt_times.append(None)

        return {
            "lats":  np.array(lats_set,  dtype=np.float32),
            "lons":  np.array(lons_set,  dtype=np.float32),
            "times": dt_times,
            "u":     u_grid,
            "v":     v_grid,
        }

    except (KeyError, IndexError, ValueError) as exc:
        raise EnvironmentalDataUnavailable(
            f"Failed to parse ERDDAP response: {exc}"
        ) from exc


# ── Local bilinear interpolation ──────────────────────────────────────────────

def _cache_key(lat: float, lon: float, ts: datetime, kind: str) -> str:
    """Create a coarse spatial-temporal cache key (2° / 6-hour resolution)."""
    lat_c = round(lat / 2.0) * 2
    lon_c = round(lon / 2.0) * 2
    ts_c  = ts.replace(minute=0, second=0, microsecond=0)
    # Round to 6-hour bucket
    ts_c  = ts_c.replace(hour=(ts_c.hour // 6) * 6)
    return f"{kind}_{lat_c}_{lon_c}_{ts_c.isoformat()}"


def _bilinear_interpolate(
    cache: Dict,
    lat: float,
    lon: float,
    timestamp: datetime,
    component: str,  # "u" or "v"
) -> float:
    """
    Bilinear spatial + nearest-time interpolation from cached grid.
    Returns 0.0 if point is outside grid (open ocean boundary — calm assumption).
    """
    lats: np.ndarray = cache["lats"]
    lons: np.ndarray = cache["lons"]
    times = cache["times"]
    grid: np.ndarray = cache[component]   # (n_times, n_lats, n_lons)

    if lats.size == 0 or lons.size == 0:
        return 0.0

    # ── Time index (nearest) ─────────────────────────────────────────────────
    if not times or all(t is None for t in times):
        t_idx = 0
    else:
        valid_times = [(i, t) for i, t in enumerate(times) if t is not None]
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        t_idx = min(
            valid_times,
            key=lambda x: abs((x[1] - timestamp).total_seconds()),
        )[0]

    layer: np.ndarray = grid[t_idx]  # (n_lats, n_lons)

    # ── Bounds check ─────────────────────────────────────────────────────────
    if lat < float(lats[0]) or lat > float(lats[-1]) or \
       lon < float(lons[0]) or lon > float(lons[-1]):
        return 0.0   # outside grid — calm assumption

    # ── Bilinear interpolation ────────────────────────────────────────────────
    # Lat index
    i1 = int(np.searchsorted(lats, lat, side="right")) - 1
    i1 = max(0, min(i1, len(lats) - 2))
    i2 = i1 + 1

    # Lon index
    j1 = int(np.searchsorted(lons, lon, side="right")) - 1
    j1 = max(0, min(j1, len(lons) - 2))
    j2 = j1 + 1

    la1, la2 = float(lats[i1]), float(lats[i2])
    lo1, lo2 = float(lons[j1]), float(lons[j2])

    dlat = la2 - la1 if la2 != la1 else 1.0
    dlon = lo2 - lo1 if lo2 != lo1 else 1.0

    t_lat = (lat - la1) / dlat
    t_lon = (lon - lo1) / dlon

    v11 = float(layer[i1, j1])
    v12 = float(layer[i1, j2])
    v21 = float(layer[i2, j1])
    v22 = float(layer[i2, j2])

    result = (
        v11 * (1 - t_lat) * (1 - t_lon) +
        v12 * (1 - t_lat) * t_lon +
        v21 * t_lat       * (1 - t_lon) +
        v22 * t_lat       * t_lon
    )
    return float(result) if not math.isnan(result) else 0.0
