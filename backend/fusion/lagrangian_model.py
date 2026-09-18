"""
Lagrangian Particle-Based Oil-Spill Transport and Backtracking Model
====================================================================
Module: fusion/lagrangian_model.py

Scientific description:
    Simulates an ensemble of particles representing possible oil-spill
    trajectories using physical velocity fields.

    Surface oil velocity model:
        V_oil = V_current + W * V_wind + V_diffusion

    Where:
        V_current  = (u_c, v_c) -- surface ocean current vector (m/s)
        V_wind     = (u_w, v_w) -- 10-m wind velocity vector (m/s)
        W          = windage coefficient (default 0.03, range 0.01-0.04)
        V_diffusion = Gaussian noise ~ N(0, sigma) per timestep

    Backward integration (15-min timestep):
        delta_lat = -(v_oil * dt) / 111320.0
        delta_lon = -(u_oil * dt) / (111320.0 * cos(lat))

    Windage ensemble: three sub-ensembles W in {0.01, 0.03, 0.04}

IMPORTANT - DATA MODE:
    dataMode = "DEMO"  -> mock/test environmental data (clearly labeled)
    dataMode = "LIVE"  -> real historical environmental data from provider

DO NOT call this an "AI model". It is a physics/numerical model.
"""

import math
import random
import uuid
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
_METERS_PER_DEG_LAT: float = 111_320.0
_HORIZONTAL_DIFFUSIVITY_M2S: float = 10.0
_DEFAULT_WINDAGE: float = 0.03
_WINDAGE_ENSEMBLE: Tuple[float, ...] = (0.01, 0.03, 0.04)
_DEFAULT_N_PARTICLES: int = 500
_DEFAULT_TIMESTEP_MIN: int = 15
_DEFAULT_DURATION_HOURS: int = 24
_MAX_DURATION_HOURS: int = 48


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SpillObservation:
    """
    Generic spill observation object -- accepts input from either:
      - Mode A: user-supplied image + lat/lon/timestamp
      - Mode B: SAR/ML pipeline providing mask + polygon + centroid
    """
    latitude: float
    longitude: float
    timestamp: datetime
    image_path: Optional[str] = None
    image_filename: Optional[str] = None
    polygon: Optional[List[Tuple[float, float]]] = None   # [(lat, lon), ...]
    spill_mask: Optional[Any] = None
    uncertainty_radius_m: float = 5_000.0

    def __post_init__(self) -> None:
        if not (-90.0 <= self.latitude <= 90.0):
            raise ValueError(f"Invalid latitude: {self.latitude}")
        if not (-180.0 <= self.longitude <= 180.0):
            raise ValueError(f"Invalid longitude: {self.longitude}")
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)


@dataclass
class EnvironmentalField:
    """Environmental velocity fields at a specific lat/lon/time. All in m/s."""
    u_current: float
    v_current: float
    u_wind: float
    v_wind: float
    data_mode: str = "DEMO"
    provider_name: str = "Unknown"
    note: str = ""


@dataclass
class TrajectoryStep:
    """Single timestep in the backward particle trajectory."""
    step: int
    hours_ago: float
    timestamp: str
    lat: float
    lon: float
    envelope: List[Tuple[float, float]]
    particle_count: int
    u_oil: float
    v_oil: float


@dataclass
class SourceRegion:
    """
    Estimated candidate source region from backward trajectory.
    CANDIDATE REGION -- not a confirmed source.
    """
    centroid_lat: float
    centroid_lon: float
    polygon: List[Tuple[float, float]]
    uncertainty_note: str = (
        "This is a probabilistic candidate source region derived from "
        "Lagrangian particle backtracking. It represents where particles "
        "were located at the start of the simulation window -- NOT a "
        "confirmed spill source. Uncertainty increases with simulation duration."
    )
    windage_consistency: Optional[str] = None


@dataclass
class InvestigationResult:
    """Full result from a Lagrangian backward investigation."""
    investigation_id: str
    spill_observation: SpillObservation
    data_mode: str
    provider_name: str
    provider_note: str
    n_particles: int
    timestep_minutes: int
    duration_hours: float
    windage_coefficients: List[float]
    trajectory_steps: List[TrajectoryStep]
    source_region: SourceRegion
    envelope_polygon: List[Tuple[float, float]]
    final_particle_positions: List[Tuple[float, float]]


# ---------------------------------------------------------------------------
# Environmental Data Provider (Abstract Base)
# ---------------------------------------------------------------------------

class EnvironmentalDataProvider(ABC):
    """
    Abstract interface for historical environmental data.
    Implement to connect to: CMEMS, NOAA GFS/HYCOM, INCOIS ORAS5.
    Credentials must come from environment variables -- never hardcoded.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier string."""

    @property
    @abstractmethod
    def data_mode(self) -> str:
        """Returns 'DEMO' or 'LIVE'."""

    @abstractmethod
    def get_current(self, lat: float, lon: float, timestamp: datetime) -> Tuple[float, float]:
        """Returns (u_current, v_current) in m/s. u=eastward, v=northward."""

    @abstractmethod
    def get_wind(self, lat: float, lon: float, timestamp: datetime) -> Tuple[float, float]:
        """Returns (u_wind, v_wind) in m/s. u=eastward, v=northward."""


class EnvironmentalDataUnavailable(Exception):
    """Raised when a provider cannot supply the requested data."""


# ---------------------------------------------------------------------------
# Mock Provider -- DEMO MODE (clearly labeled)
# ---------------------------------------------------------------------------

class MockEnvironmentalProvider(EnvironmentalDataProvider):
    """
    WARNING: DEMO DATA -- NOT OPERATIONALLY VALID.

    Deterministic mock provider returning plausible surface current and wind
    values for development/testing. Based on realistic Gulf of Mexico /
    Florida Straits climatological averages.

    All results from this provider carry dataMode = "DEMO".
    NEVER use for operational investigations.
    """

    @property
    def name(self) -> str:
        return "MockEnvironmentalProvider"

    @property
    def data_mode(self) -> str:
        return "DEMO"

    def get_current(self, lat: float, lon: float, timestamp: datetime) -> Tuple[float, float]:
        """Deterministic mock current (m/s) varying with position."""
        u_c = 0.15 + 0.08 * math.sin(math.radians(lat * 3.7))
        v_c = 0.45 + 0.12 * math.cos(math.radians(lon * 2.1))
        return u_c, v_c

    def get_wind(self, lat: float, lon: float, timestamp: datetime) -> Tuple[float, float]:
        """Deterministic mock 10-m wind (m/s) -- trade wind pattern."""
        u_w = -4.5 + 1.2 * math.sin(math.radians(lat * 1.8))
        v_w = 1.2 + 0.6 * math.cos(math.radians(lon * 1.4))
        return u_w, v_w


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _meters_per_deg_lon(lat_deg: float) -> float:
    return _METERS_PER_DEG_LAT * math.cos(math.radians(lat_deg))


def _compute_diffusion_sigma(timestep_sec: float) -> float:
    """sigma = sqrt(2 * K * dt) for Fickian diffusion."""
    return math.sqrt(2.0 * _HORIZONTAL_DIFFUSIVITY_M2S * timestep_sec)


def _convex_hull(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Convex hull of (lat, lon) points via Graham scan."""
    if len(points) < 3:
        return list(points)
    pts = [(p[1], p[0]) for p in points]  # work in (lon, lat)
    pivot = min(pts, key=lambda p: (p[1], p[0]))

    def polar_angle(p: Tuple[float, float]) -> float:
        return math.atan2(p[1] - pivot[1], p[0] - pivot[0])

    def distance_sq(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

    sorted_pts = sorted(set(pts), key=lambda p: (polar_angle(p), distance_sq(pivot, p)))
    hull: List[Tuple[float, float]] = []
    for p in sorted_pts:
        while len(hull) >= 2:
            o, a, b = hull[-2], hull[-1], p
            cross = (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
            if cross <= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    return [(p[1], p[0]) for p in hull]  # back to (lat, lon)


def _cloud_centroid(positions: List[Tuple[float, float]]) -> Tuple[float, float]:
    if not positions:
        return 0.0, 0.0
    return (
        sum(p[0] for p in positions) / len(positions),
        sum(p[1] for p in positions) / len(positions),
    )


def _sample_initial_positions(
    obs: SpillObservation,
    n: int,
    rng: random.Random,
) -> List[Tuple[float, float]]:
    """Sample n initial particle positions from the spill observation."""
    positions: List[Tuple[float, float]] = []

    if obs.polygon and len(obs.polygon) >= 3:
        lats = [p[0] for p in obs.polygon]
        lons = [p[1] for p in obs.polygon]
        lat_min, lat_max = min(lats), max(lats)
        lon_min, lon_max = min(lons), max(lons)
        attempts = 0
        while len(positions) < n and attempts < n * 20:
            lat = rng.uniform(lat_min, lat_max)
            lon = rng.uniform(lon_min, lon_max)
            positions.append((lat, lon))
            attempts += 1
        while len(positions) < n:
            positions.append((obs.latitude, obs.longitude))
    else:
        sigma_lat = obs.uncertainty_radius_m / _METERS_PER_DEG_LAT
        m_per_deg_lon = max(1.0, _meters_per_deg_lon(obs.latitude))
        for _ in range(n):
            lat = max(-90.0, min(90.0, rng.gauss(obs.latitude, sigma_lat)))
            lon = max(-180.0, min(180.0, rng.gauss(obs.longitude, obs.uncertainty_radius_m / m_per_deg_lon)))
            positions.append((lat, lon))

    return positions


# ---------------------------------------------------------------------------
# Main Lagrangian Model
# ---------------------------------------------------------------------------

class LagrangianModel:
    """
    Lagrangian particle-based oil-spill transport and backtracking model.

    Usage:
        provider = MockEnvironmentalProvider()
        model = LagrangianModel(provider)
        result = model.run_backward(observation)
    """

    def __init__(self, provider: EnvironmentalDataProvider) -> None:
        self._provider = provider

    def run_backward(
        self,
        obs: SpillObservation,
        n_particles: int = _DEFAULT_N_PARTICLES,
        duration_hours: float = _DEFAULT_DURATION_HOURS,
        timestep_min: int = _DEFAULT_TIMESTEP_MIN,
        windage_coefficients: Tuple[float, ...] = _WINDAGE_ENSEMBLE,
        seed: Optional[int] = None,
    ) -> InvestigationResult:
        """
        Run ensemble backward Lagrangian simulation.

        V_oil = V_current + W * V_wind + V_diffusion  (physical vector addition)
        Backward: delta_lat = -(v_oil * dt) / 111320
                  delta_lon = -(u_oil * dt) / (111320 * cos(lat))
        """
        if duration_hours <= 0:
            raise ValueError(f"duration_hours must be positive, got {duration_hours}")
        if duration_hours > _MAX_DURATION_HOURS:
            raise ValueError(f"duration_hours {duration_hours} exceeds max {_MAX_DURATION_HOURS}h")
        if n_particles < 1:
            raise ValueError(f"n_particles must be >= 1, got {n_particles}")
        if timestep_min < 1:
            raise ValueError(f"timestep_min must be >= 1, got {timestep_min}")

        rng = random.Random(seed)
        investigation_id = str(uuid.uuid4())
        timestep_sec = timestep_min * 60.0
        n_steps = int((duration_hours * 60) / timestep_min)
        diffusion_sigma = _compute_diffusion_sigma(timestep_sec)

        # Split particles across windage ensemble
        n_ensemble = len(windage_coefficients)
        particles_per_w = [n_particles // n_ensemble] * n_ensemble
        particles_per_w[-1] += n_particles - sum(particles_per_w)

        # Initialise particles
        all_particles: List[Tuple[float, float]] = []
        for count in particles_per_w:
            all_particles.extend(_sample_initial_positions(obs, count, rng))

        particle_windage: List[float] = []
        for w, count in zip(windage_coefficients, particles_per_w):
            particle_windage.extend([w] * count)

        trajectory_steps: List[TrajectoryStep] = []
        envelope_all: List[Tuple[float, float]] = list(all_particles)

        # Initial env for step-0 reporting
        try:
            u_c0, v_c0 = self._provider.get_current(obs.latitude, obs.longitude, obs.timestamp)
            u_w0, v_w0 = self._provider.get_wind(obs.latitude, obs.longitude, obs.timestamp)
        except EnvironmentalDataUnavailable:
            u_c0, v_c0, u_w0, v_w0 = 0.0, 0.0, 0.0, 0.0
            logger.warning("Env data unavailable at step 0 for %.4f,%.4f", obs.latitude, obs.longitude)
        mean_w0 = sum(windage_coefficients) / len(windage_coefficients)
        u_oil0 = u_c0 + mean_w0 * u_w0
        v_oil0 = v_c0 + mean_w0 * v_w0

        centroid0 = _cloud_centroid(all_particles)
        hull0 = _convex_hull(all_particles[::max(1, len(all_particles) // 50)])
        trajectory_steps.append(TrajectoryStep(
            step=0,
            hours_ago=0.0,
            timestamp=obs.timestamp.isoformat(),
            lat=round(centroid0[0], 6),
            lon=round(centroid0[1], 6),
            envelope=[(round(p[0], 5), round(p[1], 5)) for p in hull0],
            particle_count=len(all_particles),
            u_oil=round(u_oil0, 4),
            v_oil=round(v_oil0, 4),
        ))

        # Backward integration loop
        for step in range(1, n_steps + 1):
            t_step = obs.timestamp - timedelta(minutes=step * timestep_min)
            hours_ago = step * timestep_min / 60.0

            new_particles: List[Tuple[float, float]] = []
            u_oils: List[float] = []
            v_oils: List[float] = []

            for i, (lat, lon) in enumerate(all_particles):
                W = particle_windage[i]
                try:
                    u_c, v_c = self._provider.get_current(lat, lon, t_step)
                    u_w, v_w = self._provider.get_wind(lat, lon, t_step)
                except EnvironmentalDataUnavailable:
                    u_c, v_c, u_w, v_w = 0.0, 0.0, 0.0, 0.0
                    logger.warning("Env data unavailable at %.4f,%.4f %s", lat, lon, t_step)

                # Physical vector addition -- NOT a weighted blend
                noise_u = rng.gauss(0.0, diffusion_sigma)
                noise_v = rng.gauss(0.0, diffusion_sigma)
                u_oil = u_c + W * u_w + noise_u
                v_oil = v_c + W * v_w + noise_v
                u_oils.append(u_oil)
                v_oils.append(v_oil)

                # Backward displacement
                m_per_deg_lon = max(1.0, _meters_per_deg_lon(lat))
                new_lat = max(-90.0, min(90.0, lat - (v_oil * timestep_sec) / _METERS_PER_DEG_LAT))
                new_lon = max(-180.0, min(180.0, lon - (u_oil * timestep_sec) / m_per_deg_lon))
                new_particles.append((new_lat, new_lon))

            all_particles = new_particles
            envelope_all.extend(all_particles)

            centroid = _cloud_centroid(all_particles)
            sample = all_particles[::max(1, len(all_particles) // 50)]
            hull = _convex_hull(sample)
            mean_u = sum(u_oils) / len(u_oils) if u_oils else 0.0
            mean_v = sum(v_oils) / len(v_oils) if v_oils else 0.0

            trajectory_steps.append(TrajectoryStep(
                step=step,
                hours_ago=round(hours_ago, 2),
                timestamp=t_step.isoformat(),
                lat=round(centroid[0], 6),
                lon=round(centroid[1], 6),
                envelope=[(round(p[0], 5), round(p[1], 5)) for p in hull],
                particle_count=len(all_particles),
                u_oil=round(mean_u, 4),
                v_oil=round(mean_v, 4),
            ))

        # Source region
        source_hull = _convex_hull(all_particles)
        source_centroid = _cloud_centroid(all_particles)
        windage_note = _assess_windage_consistency(all_particles, particles_per_w, windage_coefficients)

        source_region = SourceRegion(
            centroid_lat=round(source_centroid[0], 6),
            centroid_lon=round(source_centroid[1], 6),
            polygon=[(round(p[0], 5), round(p[1], 5)) for p in source_hull],
            windage_consistency=windage_note,
        )

        corridor_hull = _convex_hull(envelope_all[::max(1, len(envelope_all) // 200)])

        return InvestigationResult(
            investigation_id=investigation_id,
            spill_observation=obs,
            data_mode=self._provider.data_mode,
            provider_name=self._provider.name,
            provider_note=(
                "WARNING: DEMO MODE -- Environmental data is synthetic and NOT "
                "suitable for operational use. Integrate a live provider (CMEMS/NOAA) "
                "for real investigations."
                if self._provider.data_mode == "DEMO"
                else "Live environmental data from " + self._provider.name
            ),
            n_particles=n_particles,
            timestep_minutes=timestep_min,
            duration_hours=duration_hours,
            windage_coefficients=list(windage_coefficients),
            trajectory_steps=trajectory_steps,
            source_region=source_region,
            envelope_polygon=[(round(p[0], 5), round(p[1], 5)) for p in corridor_hull],
            final_particle_positions=[(round(p[0], 6), round(p[1], 6)) for p in all_particles],
        )


def _assess_windage_consistency(
    all_particles: List[Tuple[float, float]],
    particles_per_w: List[int],
    windage_coefficients: Tuple[float, ...],
) -> Optional[str]:
    if len(windage_coefficients) < 2:
        return None
    groups: List[List[Tuple[float, float]]] = []
    idx = 0
    for count in particles_per_w:
        groups.append(all_particles[idx: idx + count])
        idx += count
    centroids = [_cloud_centroid(g) for g in groups if g]
    if len(centroids) < 2:
        return None
    max_dist_km = 0.0
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            lat1, lon1 = centroids[i]
            lat2, lon2 = centroids[j]
            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)
            a = (math.sin(dlat / 2) ** 2 +
                 math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
            dist_km = 6371.0 * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
            max_dist_km = max(max_dist_km, dist_km)
    if max_dist_km < 30.0:
        return (
            f"Trajectory corridor consistent across windage scenarios W={windage_coefficients} "
            f"(max divergence {max_dist_km:.1f} km). Increases confidence in candidate source "
            "region -- does NOT confirm vessel identity or legal causation."
        )
    return (
        f"Trajectory corridors diverge across windage scenarios "
        f"(max {max_dist_km:.1f} km). Higher windage uncertainty -- interpret source region with caution."
    )


def parse_spill_timestamp(ts_str: str) -> datetime:
    """Parse ISO-8601 timestamp string to timezone-aware datetime."""
    if not ts_str or not isinstance(ts_str, str):
        raise ValueError(f"Invalid timestamp: {ts_str!r}")
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"Cannot parse timestamp {ts_str!r}: {exc}") from exc
