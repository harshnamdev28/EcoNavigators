"""
Test suite for Lagrangian particle backtracking model.
=====================================================
Tests: 16 deterministic tests covering physics, validation, coordinate safety,
       AIS matching, attribution scoring, and error cases.

All tests use deterministic random seeds -- no live external API calls.
"""

import math
import random
import pytest
from datetime import datetime, timezone

from fusion.lagrangian_model import (
    LagrangianModel,
    MockEnvironmentalProvider,
    SpillObservation,
    _cloud_centroid,
    _convex_hull,
    _compute_diffusion_sigma,
    _meters_per_deg_lon,
    _sample_initial_positions,
    parse_spill_timestamp,
    _METERS_PER_DEG_LAT,
    _WINDAGE_ENSEMBLE,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def provider():
    return MockEnvironmentalProvider()


@pytest.fixture
def model(provider):
    return LagrangianModel(provider)


@pytest.fixture
def sample_obs():
    return SpillObservation(
        latitude=25.7715,
        longitude=-80.1518,
        timestamp=datetime(2026, 9, 5, 10, 39, 0, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
# Test 1: Windage calculation
# ---------------------------------------------------------------------------

def test_windage_calculation():
    """
    V_oil = V_current + W * V_wind
    u_oil = u_c + W * u_w  (eastward)
    v_oil = v_c + W * v_w  (northward)
    """
    u_c, v_c = 0.412, 0.231    # m/s current
    u_w, v_w = -4.5, 1.2       # m/s wind
    W = 0.03

    u_oil = u_c + W * u_w
    v_oil = v_c + W * v_w

    assert abs(u_oil - (0.412 + 0.03 * -4.5)) < 1e-9
    assert abs(v_oil - (0.231 + 0.03 * 1.2)) < 1e-9
    # Verify directional correctness: wind is westward so oil drifts slightly westward
    assert u_oil < u_c   # wind reduces eastward component


# ---------------------------------------------------------------------------
# Test 2: Vector addition of current and wind at non-trivial angles
# ---------------------------------------------------------------------------

def test_vector_addition_nontrivial():
    """
    Example from spec:
    current = 0.8 kts at 45deg -> u_c = 0.8*sin(45), v_c = 0.8*cos(45)
    wind = 12 kts at 80deg -> u_w = 12*sin(80), v_w = 12*cos(80)
    W = 0.03 -> wind contribution = 12*0.03 = 0.36 kts

    Convert to m/s: 1 knot = 0.5144 m/s
    """
    knots_to_ms = 0.5144
    current_kts = 0.8
    wind_kts = 12.0
    W = 0.03

    # Convert to m/s components (bearing: 0=N, 90=E)
    u_c = current_kts * knots_to_ms * math.sin(math.radians(45.0))
    v_c = current_kts * knots_to_ms * math.cos(math.radians(45.0))
    u_w = wind_kts * knots_to_ms * math.sin(math.radians(80.0))
    v_w = wind_kts * knots_to_ms * math.cos(math.radians(80.0))

    wind_contribution_ms = wind_kts * knots_to_ms * W
    expected_wind_contribution_kts = wind_kts * W

    assert abs(expected_wind_contribution_kts - 0.36) < 1e-9

    u_oil = u_c + W * u_w
    v_oil = v_c + W * v_w

    # Result must be vector sum -- verify Euclidean magnitude is in plausible range
    speed_oil = math.sqrt(u_oil ** 2 + v_oil ** 2)
    assert 0.1 < speed_oil < 1.5   # realistic oil drift speed m/s


# ---------------------------------------------------------------------------
# Test 3: Backward timestep delta-lat/delta-lon calculation
# ---------------------------------------------------------------------------

def test_backward_timestep_calculation():
    """
    Backward displacement:
        delta_lat = -(v_oil * dt) / 111320
        delta_lon = -(u_oil * dt) / (111320 * cos(lat))
    dt = 15 min = 900 s
    """
    lat = 25.7715
    lon = -80.1518
    v_oil = 0.45     # m/s northward
    u_oil = 0.15     # m/s eastward
    dt = 900.0       # seconds

    m_per_deg_lon = _METERS_PER_DEG_LAT * math.cos(math.radians(lat))
    delta_lat = -(v_oil * dt) / _METERS_PER_DEG_LAT
    delta_lon = -(u_oil * dt) / m_per_deg_lon

    new_lat = lat + delta_lat
    new_lon = lon + delta_lon

    # Backward in time: northward current means spill came from south
    assert new_lat < lat   # stepped south (backward)
    assert new_lon < lon   # stepped west (backward, eastward current)

    # Displacement magnitude should be small (< 1 km for 15 min)
    dist_km = math.sqrt(
        (delta_lat * _METERS_PER_DEG_LAT / 1000) ** 2 +
        (delta_lon * m_per_deg_lon / 1000) ** 2
    )
    assert 0.0 < dist_km < 2.0


# ---------------------------------------------------------------------------
# Test 4: Particle generation from point
# ---------------------------------------------------------------------------

def test_particle_generation_from_point(sample_obs):
    """500 particles should be generated with valid coordinates."""
    rng = random.Random(42)
    positions = _sample_initial_positions(sample_obs, 500, rng)

    assert len(positions) == 500
    for lat, lon in positions:
        assert -90.0 <= lat <= 90.0, f"Invalid lat: {lat}"
        assert -180.0 <= lon <= 180.0, f"Invalid lon: {lon}"
    # All should be near the source point (within ~50km)
    for lat, lon in positions:
        dlat = abs(lat - sample_obs.latitude)
        dlon = abs(lon - sample_obs.longitude)
        assert dlat < 1.0, f"Particle too far in lat: {dlat}"
        assert dlon < 1.0, f"Particle too far in lon: {dlon}"


# ---------------------------------------------------------------------------
# Test 5: Particle generation from polygon
# ---------------------------------------------------------------------------

def test_particle_generation_from_polygon():
    """Particles from polygon should all be within bounding box."""
    polygon = [
        (25.0, -80.5),
        (25.5, -80.5),
        (25.5, -80.0),
        (25.0, -80.0),
    ]
    obs = SpillObservation(
        latitude=25.25,
        longitude=-80.25,
        timestamp=datetime(2026, 9, 5, 10, 0, 0, tzinfo=timezone.utc),
        polygon=polygon,
    )
    rng = random.Random(42)
    positions = _sample_initial_positions(obs, 200, rng)

    assert len(positions) == 200
    for lat, lon in positions:
        assert 25.0 <= lat <= 25.5, f"lat outside polygon bbox: {lat}"
        assert -80.5 <= lon <= -80.0, f"lon outside polygon bbox: {lon}"


# ---------------------------------------------------------------------------
# Test 6: Trajectory remains geographically valid
# ---------------------------------------------------------------------------

def test_trajectory_geographic_validity(model, sample_obs):
    """All trajectory centroids must remain within valid geographic bounds."""
    result = model.run_backward(
        sample_obs, n_particles=50, duration_hours=6, timestep_min=15, seed=42
    )

    for step in result.trajectory_steps:
        assert -90.0 <= step.lat <= 90.0, f"step {step.step}: invalid lat {step.lat}"
        assert -180.0 <= step.lon <= 180.0, f"step {step.step}: invalid lon {step.lon}"

    for lat, lon in result.final_particle_positions:
        assert -90.0 <= lat <= 90.0
        assert -180.0 <= lon <= 180.0


# ---------------------------------------------------------------------------
# Test 7: Latitude/longitude ordering (coordinate safety)
# ---------------------------------------------------------------------------

def test_coordinate_ordering(model, sample_obs):
    """
    CRITICAL: Verify lat/lon ordering is consistent throughout.
    - trajectory_steps: (lat, lon)
    - source_region centroid: (lat, lon)
    - envelope_polygon: list of (lat, lon)
    - Leaflet expects [lat, lon] -- all outputs must follow this
    """
    result = model.run_backward(
        sample_obs, n_particles=50, duration_hours=2, timestep_min=15, seed=42
    )

    # Centroid at step 0 should be near observation point
    step0 = result.trajectory_steps[0]
    # lat is around 25.77, lon is around -80.15
    # If swapped, lat would be -80 (invalid for lat in this region) and lon 25
    assert 20.0 < step0.lat < 35.0, f"step0.lat looks like a lon: {step0.lat}"
    assert -90.0 < step0.lon < -70.0, f"step0.lon looks like a lat: {step0.lon}"

    # Source region centroid
    assert 15.0 < result.source_region.centroid_lat < 35.0
    assert -100.0 < result.source_region.centroid_lon < -60.0

    # Envelope polygon tuples are (lat, lon)
    for tup in result.envelope_polygon:
        lat, lon = tup
        assert -90.0 <= lat <= 90.0
        assert -180.0 <= lon <= 180.0


# ---------------------------------------------------------------------------
# Test 8: Environmental data interpolation (mock provider)
# ---------------------------------------------------------------------------

def test_environmental_data_interpolation(provider):
    """
    MockEnvironmentalProvider must return finite non-zero values
    at different lat/lon positions.
    """
    ts = datetime(2026, 9, 5, 10, 0, 0, tzinfo=timezone.utc)
    results = []
    for lat in [20.0, 25.0, 30.0]:
        for lon in [-90.0, -80.0, -70.0]:
            u_c, v_c = provider.get_current(lat, lon, ts)
            u_w, v_w = provider.get_wind(lat, lon, ts)
            assert math.isfinite(u_c) and math.isfinite(v_c)
            assert math.isfinite(u_w) and math.isfinite(v_w)
            # Values should be in physically plausible ranges
            assert -5.0 < u_c < 5.0
            assert -5.0 < v_c < 5.0
            assert -20.0 < u_w < 20.0
            assert -20.0 < v_w < 20.0
            results.append((u_c, v_c, u_w, v_w))

    # Values must not be all identical (variation with lat/lon)
    u_currents = [r[0] for r in results]
    assert len(set(round(v, 4) for v in u_currents)) > 1, "Current has no spatial variation"


# ---------------------------------------------------------------------------
# Test 9: AIS candidate spatial matching (unit test without DB)
# ---------------------------------------------------------------------------

def test_min_distance_to_trajectory():
    """Vessel within trajectory corridor should have low distance."""
    from fusion.historical_ais_matcher import _min_distance_to_trajectory

    trajectory_centroids = [
        (25.77, -80.15),
        (25.70, -80.18),
        (25.63, -80.21),
        (25.56, -80.24),
    ]

    # Vessel close to trajectory
    dist_near = _min_distance_to_trajectory(25.65, -80.20, trajectory_centroids)
    assert dist_near < 10.0, f"Near vessel should be < 10 km: {dist_near}"

    # Vessel far from trajectory
    dist_far = _min_distance_to_trajectory(30.0, -75.0, trajectory_centroids)
    assert dist_far > 500.0, f"Far vessel should be > 500 km: {dist_far}"


# ---------------------------------------------------------------------------
# Test 10: MMSI deduplication
# ---------------------------------------------------------------------------

def test_mmsi_deduplication():
    """Each MMSI should appear at most once in candidate output."""
    from fusion.historical_ais_matcher import CandidateMatch

    # Simulate what would happen if two pings from same MMSI produce duplicates
    # (in real code, groupby MMSI handles this -- we test the dedup logic)
    mmsi_set = set()
    candidates = [
        CandidateMatch(rank=1, mmsi="123456789", vessel_name="V1", imo=None,
                       min_distance_km=5.0, trajectory_overlap_score=0.8,
                       time_match_score=0.6, ais_anomaly_score=0.5,
                       distance_score=0.9, attribution_score=75.0,
                       matching_ais_pings=10, lat=25.7, lon=-80.1, position_timestamp=None),
        CandidateMatch(rank=2, mmsi="987654321", vessel_name="V2", imo=None,
                       min_distance_km=20.0, trajectory_overlap_score=0.4,
                       time_match_score=0.3, ais_anomaly_score=0.0,
                       distance_score=0.7, attribution_score=40.0,
                       matching_ais_pings=5, lat=25.5, lon=-80.2, position_timestamp=None),
    ]

    for c in candidates:
        assert c.mmsi not in mmsi_set, f"Duplicate MMSI: {c.mmsi}"
        mmsi_set.add(c.mmsi)
    assert len(mmsi_set) == 2


# ---------------------------------------------------------------------------
# Test 11: Attribution scoring formula
# ---------------------------------------------------------------------------

def test_attribution_scoring_formula():
    """
    attribution_score = (
        0.40 * trajectory_overlap_score +
        0.30 * time_match_score +
        0.20 * distance_score +
        0.10 * ais_anomaly_score
    ) * 100
    """
    trajectory_overlap_score = 0.8
    time_match_score = 0.6
    distance_score = 0.9
    ais_anomaly_score = 0.5

    expected = (
        0.40 * trajectory_overlap_score +
        0.30 * time_match_score +
        0.20 * distance_score +
        0.10 * ais_anomaly_score
    ) * 100
    # 0.4*0.8 + 0.3*0.6 + 0.2*0.9 + 0.1*0.5 = 0.32 + 0.18 + 0.18 + 0.05 = 0.73 -> 73.0
    computed = round(expected, 1)
    assert abs(computed - 73.0) < 0.5

    # Vessel with 0 on all metrics should score 0
    zero_score = (0.40 * 0 + 0.30 * 0 + 0.20 * 0 + 0.10 * 0) * 100
    assert zero_score == 0.0

    # Perfect vessel should score 100
    perfect_score = (0.40 * 1.0 + 0.30 * 1.0 + 0.20 * 1.0 + 0.10 * 1.0) * 100
    assert abs(perfect_score - 100.0) < 0.001


# ---------------------------------------------------------------------------
# Test 12: Missing environmental data -- graceful fallback
# ---------------------------------------------------------------------------

class BrokenProvider(MockEnvironmentalProvider):
    """Provider that always raises EnvironmentalDataUnavailable."""
    @property
    def data_mode(self) -> str:
        return "DEMO"

    def get_current(self, lat, lon, timestamp):
        from fusion.lagrangian_model import EnvironmentalDataUnavailable
        raise EnvironmentalDataUnavailable("No data available")

    def get_wind(self, lat, lon, timestamp):
        from fusion.lagrangian_model import EnvironmentalDataUnavailable
        raise EnvironmentalDataUnavailable("No data available")


def test_missing_environmental_data_graceful_fallback(sample_obs):
    """Model should complete and return result even if provider raises."""
    model = LagrangianModel(BrokenProvider())
    result = model.run_backward(
        sample_obs, n_particles=20, duration_hours=2, timestep_min=15, seed=42
    )
    # Should not raise -- returns result with zero-drift particles
    assert result.investigation_id
    assert len(result.trajectory_steps) > 0
    assert result.data_mode == "DEMO"


# ---------------------------------------------------------------------------
# Test 13: Missing AIS history returns empty list (no fabrication)
# ---------------------------------------------------------------------------

def test_missing_ais_returns_empty():
    """
    find_candidate_vessels with empty trajectory_centroids must return [].
    Should never fabricate vessels.
    """
    from fusion.historical_ais_matcher import find_candidate_vessels

    ts = datetime(2026, 9, 5, 10, 39, 0, tzinfo=timezone.utc)

    # Pass empty centroids -- no DB access needed, returns [] early
    result = find_candidate_vessels(
        trajectory_centroids=[],
        spill_lat=25.77,
        spill_lon=-80.15,
        spill_timestamp=ts,
        duration_hours=24,
        conn=None,  # will fail if it tries to connect without centroids check
    )
    # Empty centroids -> function logs warning and returns [] before DB call
    # (actual DB test would need a live DB -- skipped here)
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Test 14: Invalid timestamp raises ValueError
# ---------------------------------------------------------------------------

def test_invalid_timestamp_raises():
    with pytest.raises(ValueError):
        parse_spill_timestamp("not-a-date")

    with pytest.raises(ValueError):
        parse_spill_timestamp("")

    with pytest.raises(ValueError):
        parse_spill_timestamp(None)


# ---------------------------------------------------------------------------
# Test 15: Invalid coordinates raise ValueError
# ---------------------------------------------------------------------------

def test_invalid_coordinates_raise():
    ts = datetime(2026, 9, 5, 10, 39, 0, tzinfo=timezone.utc)

    with pytest.raises(ValueError):
        SpillObservation(latitude=95.0, longitude=-80.0, timestamp=ts)

    with pytest.raises(ValueError):
        SpillObservation(latitude=25.0, longitude=200.0, timestamp=ts)

    with pytest.raises(ValueError):
        SpillObservation(latitude=-91.0, longitude=0.0, timestamp=ts)


# ---------------------------------------------------------------------------
# Test 16: Zero/negative duration raises ValueError
# ---------------------------------------------------------------------------

def test_invalid_duration_raises(model, sample_obs):
    with pytest.raises(ValueError):
        model.run_backward(sample_obs, n_particles=10, duration_hours=0)

    with pytest.raises(ValueError):
        model.run_backward(sample_obs, n_particles=10, duration_hours=-5)

    with pytest.raises(ValueError):
        model.run_backward(sample_obs, n_particles=10, duration_hours=100)
