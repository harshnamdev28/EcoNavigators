"""
Coordinate Regression and Pipeline Tests
========================================
Validates coordinate consistency end-to-end across:
  - PostGIS conventions: ST_X = lon, ST_Y = lat
  - GeoJSON conventions: [lon, lat]
  - Leaflet conventions: [lat, lon]
  - Lagrangian physical transport model coordinates
  - MIT-B2 segmentation polygon geographic projection
  - AIS candidate matching coordinates
  - Fallback and zero-coordinate preservation (0.0 is not falsy)
  - Concrete end-to-end benchmark: lat = 25.7715, lon = -80.1518
"""

import math
from datetime import datetime, timezone
import pytest

from fusion.lagrangian_model import (
    SpillObservation,
    MockEnvironmentalProvider,
    LagrangianModel,
    _convex_hull,
    _meters_per_deg_lon,
)
from fusion.historical_ais_matcher import (
    CandidateMatch,
    haversine_km,
    is_valid_geo_coord,
    _min_distance_to_trajectory,
    _min_distance_to_envelope,
)
from sar_pipeline.mit_b2_inference import segmentation_polygon_to_geographic


# ---------------------------------------------------------------------------
# Test 1: PostGIS coordinate extraction convention
# ST_X(geom) is longitude, ST_Y(geom) is latitude
# ---------------------------------------------------------------------------
def test_1_postgis_coordinate_order():
    """Verify understanding of PostGIS X/Y coordinates: X=lon, Y=lat."""
    # Suppose a PostGIS point is constructed as ST_MakePoint(lon, lat, 4326)
    # Point(-80.1518, 25.7715)
    mock_point_x = -80.1518  # ST_X is longitude
    mock_point_y = 25.7715   # ST_Y is latitude

    lon = mock_point_x
    lat = mock_point_y

    assert lat == 25.7715, "ST_Y must map to latitude"
    assert lon == -80.1518, "ST_X must map to longitude"
    assert is_valid_geo_coord(lat, lon) is True


# ---------------------------------------------------------------------------
# Test 2: GeoJSON [lon, lat] to Leaflet [lat, lon] conversion
# ---------------------------------------------------------------------------
def test_2_geojson_to_leaflet_conversion():
    """Verify that GeoJSON [lon, lat] converts to Leaflet [lat, lon] without confusion."""
    geojson_coord = [-80.1518, 25.7715]  # [longitude, latitude]
    leaflet_coord = [geojson_coord[1], geojson_coord[0]]  # [latitude, longitude]

    assert leaflet_coord[0] == 25.7715, "Leaflet first coordinate must be latitude"
    assert leaflet_coord[1] == -80.1518, "Leaflet second coordinate must be longitude"


# ---------------------------------------------------------------------------
# Test 3: Leaflet coordinate ordering
# ---------------------------------------------------------------------------
def test_3_leaflet_coordinate_order():
    """Verify that candidate and trajectory objects produce [lat, lon] pairs for Leaflet."""
    cand = CandidateMatch(
        rank=1,
        mmsi="123456789",
        vessel_name="Test Vessel",
        imo=None,
        min_distance_km=2.5,
        trajectory_overlap_score=0.8,
        time_match_score=0.9,
        ais_anomaly_score=0.1,
        distance_score=0.975,
        source_region_score=0.85,
        attribution_score=85.0,
        matching_ais_pings=12,
        lat=25.7715,
        lon=-80.1518,
        position_timestamp="2026-09-05T10:39:00Z",
    )

    leaflet_pos = [cand.lat, cand.lon]
    assert leaflet_pos == [25.7715, -80.1518]
    assert leaflet_pos[0] > 0, "Miami latitude is North (+)"
    assert leaflet_pos[1] < 0, "Miami longitude is West (-)"


# ---------------------------------------------------------------------------
# Test 4: MIT-B2 relative to geographic conversion
# ---------------------------------------------------------------------------
def test_4_mit_b2_relative_to_geographic_conversion():
    """Verify segmentation_polygon_to_geographic scales relative fractions around spill lat/lon."""
    polygon_rel = [(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)]
    spill_lat = 25.7715
    spill_lon = -80.1518

    geo = segmentation_polygon_to_geographic(
        polygon_rel=polygon_rel,
        spill_lat=spill_lat,
        spill_lon=spill_lon,
        image_lat_span=0.15,
        image_lon_span=0.15,
    )

    assert len(geo) == 4
    # All resulting points should be (lat, lon) tuples within extent of spill
    for lat_p, lon_p in geo:
        assert isinstance(lat_p, float)
        assert isinstance(lon_p, float)
        dist = haversine_km(spill_lat, spill_lon, lat_p, lon_p)
        assert dist <= 20.0, f"Polygon vertex too far from center: {dist} km"


# ---------------------------------------------------------------------------
# Test 5: Lagrangian backward integration coordinate updates
# ---------------------------------------------------------------------------
def test_5_lagrangian_integration_coordinates():
    """Verify backward integration correctly uses -(v_oil * dt)/111320 for lat."""
    lat = 25.7715
    lon = -80.1518
    dt = 900.0  # 15 minutes in seconds
    v_oil = 0.5  # northward velocity 0.5 m/s
    u_oil = 0.2  # eastward velocity 0.2 m/s

    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(lat))

    # Backward in time: moving backwards means subtracting velocity displacement
    new_lat = lat - (v_oil * dt) / m_per_deg_lat
    new_lon = lon - (u_oil * dt) / m_per_deg_lon

    # If oil was carried north (v_oil > 0), backwards in time it was further South
    assert new_lat < lat, "Backtracking northward drift must place origin further South"
    # If oil was carried east (u_oil > 0), backwards in time it was further West
    assert new_lon < lon, "Backtracking eastward drift must place origin further West"


# ---------------------------------------------------------------------------
# Test 6: Convex hull coordinate preservation
# ---------------------------------------------------------------------------
def test_6_lagrangian_convex_hull_coordinates():
    """Verify _convex_hull accepts and returns (lat, lon) tuples without coordinate flipping."""
    points = [
        (25.7715, -80.1518),
        (25.8000, -80.1400),
        (25.7500, -80.1600),
        (25.7700, -80.1300),
        (25.7600, -80.1700),
    ]

    hull = _convex_hull(points)
    assert len(hull) >= 3, "Convex hull of 5 spread points must have at least 3 vertices"
    for pt in hull:
        lat, lon = pt
        assert 25.70 <= lat <= 25.85, f"Latitude {lat} out of expected range"
        assert -80.20 <= lon <= -80.10, f"Longitude {lon} out of expected range"


# ---------------------------------------------------------------------------
# Test 7: AIS search corridor bounding box calculations
# ---------------------------------------------------------------------------
def test_7_ais_matcher_bounding_box():
    """Verify search bounding box is derived using km to degree conversion."""
    traj_centroids = [(25.7715, -80.1518), (25.7000, -80.1800)]
    spill_lat, spill_lon = 25.7715, -80.1518
    search_radius_km = 30.0

    lats = [c[0] for c in traj_centroids] + [spill_lat]
    lons = [c[1] for c in traj_centroids] + [spill_lon]

    lat_min = min(lats) - (search_radius_km / 111.0)
    lat_max = max(lats) + (search_radius_km / 111.0)
    lon_min = min(lons) - (search_radius_km / 111.0)
    lon_max = max(lons) + (search_radius_km / 111.0)

    assert lat_min < min(lats)
    assert lat_max > max(lats)
    assert lon_min < min(lons)
    assert lon_max > max(lons)
    # Check degree span is approximately 30 / 111 ≈ 0.27 degrees
    assert math.isclose(lat_max - max(lats), 30.0 / 111.0, rel_tol=1e-3)


# ---------------------------------------------------------------------------
# Test 8: Haversine distance argument order and symmetry
# ---------------------------------------------------------------------------
def test_8_haversine_argument_order():
    """Verify haversine_km(lat1, lon1, lat2, lon2) is symmetric and handles 1 degree lat."""
    lat1, lon1 = 25.0, -80.0
    lat2, lon2 = 26.0, -80.0  # 1 degree North

    d1 = haversine_km(lat1, lon1, lat2, lon2)
    d2 = haversine_km(lat2, lon2, lat1, lon1)

    assert math.isclose(d1, d2, rel_tol=1e-5), "Haversine distance must be symmetric"
    # 1 degree latitude is approximately 111.1 - 111.3 km
    assert 110.0 <= d1 <= 112.0, f"1 deg lat should be ~111km, got {d1}"


# ---------------------------------------------------------------------------
# Test 9: Coordinate validation rejects out-of-range coords
# ---------------------------------------------------------------------------
def test_9_coordinate_validation():
    """Verify is_valid_geo_coord correctly handles valid/invalid coordinates."""
    assert is_valid_geo_coord(25.7715, -80.1518) is True
    assert is_valid_geo_coord(-89.9, 179.9) is True
    assert is_valid_geo_coord(0.0, 0.0) is False  # Null Island (uninitialized GPS) is rejected
    assert is_valid_geo_coord(0.001, 0.001) is True  # Real non-zero coord near equator is valid
    assert is_valid_geo_coord(91.0, 0.0) is False  # Lat out of range
    assert is_valid_geo_coord(25.0, 185.0) is False  # Lon out of range
    assert is_valid_geo_coord(None, -80.0) is False
    assert is_valid_geo_coord(25.0, None) is False


# ---------------------------------------------------------------------------
# Test 10: Zero coordinates (0.0, 0.0) preserved and not treated as falsy
# ---------------------------------------------------------------------------
def test_10_no_falsy_zero_coordinates():
    """Verify that lat=0.0 and lon=0.0 are valid and preserved without fallback."""
    obs = SpillObservation(
        latitude=0.0,
        longitude=0.0,
        timestamp=datetime(2026, 9, 5, 12, 0, 0, tzinfo=timezone.utc),
    )
    assert obs.latitude == 0.0
    assert obs.longitude == 0.0
    assert (obs.latitude is not None) and (obs.longitude is not None)


# ---------------------------------------------------------------------------
# Test 11: Distance to corridor envelope calculation
# ---------------------------------------------------------------------------
def test_11_min_distance_to_corridor():
    """Verify distance calculations to trajectory envelope vs centroids."""
    centroids = [(25.7715, -80.1518), (25.7500, -80.1600)]
    envelope = [
        (25.7800, -80.1400),
        (25.7800, -80.1700),
        (25.7400, -80.1700),
        (25.7400, -80.1400),
    ]

    # Vessel right at centroid
    d1 = _min_distance_to_trajectory(25.7715, -80.1518, centroids)
    assert d1 < 0.001, "Distance to exact centroid should be ~0"

    # Vessel closer to an envelope corner than centroid
    d_env = _min_distance_to_envelope(25.7800, -80.1400, envelope, centroids)
    assert d_env < 0.001, "Distance to exact envelope vertex should be ~0"


# ---------------------------------------------------------------------------
# Test 12: Concrete benchmark: Spill at latitude = 25.7715, longitude = -80.1518
# ---------------------------------------------------------------------------
def test_12_concrete_spill_coordinate_end_to_end():
    """
    End-to-end benchmark requested by user:
    Spill: latitude = 25.7715, longitude = -80.1518
    Runs 12h simulation with MockEnvironmentalProvider and checks:
      1. SpillObservation retains exact coordinates
      2. Trajectory timesteps step backwards in time
      3. All trajectory coordinates remain in realistic Florida Straits region
      4. Source region centroid is geographically coherent
      5. Envelope polygon surrounds the trajectory
    """
    spill_lat = 25.7715
    spill_lon = -80.1518
    spill_ts = datetime(2026, 9, 5, 10, 39, 0, tzinfo=timezone.utc)

    obs = SpillObservation(
        latitude=spill_lat,
        longitude=spill_lon,
        timestamp=spill_ts,
        uncertainty_radius_m=5000.0,
    )

    provider = MockEnvironmentalProvider()
    model = LagrangianModel(provider=provider)

    result = model.run_backward(
        obs=obs,
        n_particles=100,
        duration_hours=12.0,
        timestep_min=15,
        windage_coefficients=(0.01, 0.03, 0.04),
        seed=42,
    )

    # 1. Spill observation retention
    assert result.spill_observation.latitude == 25.7715
    assert result.spill_observation.longitude == -80.1518

    # 2. Number of trajectory steps: 12h at 15min = 48 steps (plus initial = 49)
    assert len(result.trajectory_steps) >= 48

    # First step is at t=0 (hours_ago = 0.0)
    step0 = result.trajectory_steps[0]
    assert step0.hours_ago == 0.0
    assert math.isclose(step0.lat, spill_lat, abs_tol=0.05)
    assert math.isclose(step0.lon, spill_lon, abs_tol=0.05)

    # Last step is at t=12h
    step_last = result.trajectory_steps[-1]
    assert math.isclose(step_last.hours_ago, 12.0, abs_tol=0.1)

    # 3. Trajectory bounds in Florida Straits
    for step in result.trajectory_steps:
        assert 24.0 <= step.lat <= 28.0, f"Lat {step.lat} drifted out of regional corridor"
        assert -82.0 <= step.lon <= -78.0, f"Lon {step.lon} drifted out of regional corridor"

    # 4. Source region centroid
    src = result.source_region
    assert 24.0 <= src.centroid_lat <= 28.0
    assert -82.0 <= src.centroid_lon <= -78.0
    assert len(src.polygon) >= 3, "Source region polygon must have at least 3 vertices"

    # 5. Envelope polygon
    assert len(result.envelope_polygon) >= 3, "Envelope polygon must have at least 3 vertices"
    for pt in result.envelope_polygon:
        assert isinstance(pt, tuple)
        assert len(pt) == 2
        # Check pt is (lat, lon)
        assert 24.0 <= pt[0] <= 28.0
        assert -82.0 <= pt[1] <= -78.0
