"""
Historical AIS Vessel Matcher for Lagrangian Investigation
==========================================================
Module: fusion/historical_ais_matcher.py

Queries existing ais_positions and ais_anomalies tables (PostGIS)
to find candidate vessels whose historical tracks intersect the
backward Lagrangian trajectory corridor.

Reuses:
    haversine_km, is_valid_geo_coord from fusion.backtracking_service

Attribution scoring:
    attribution_score = (
        0.40 * trajectory_overlap_score +
        0.30 * time_match_score +
        0.20 * distance_score +
        0.10 * ais_anomaly_score
    ) * 100

Score is an INVESTIGATION PRIORITY score -- not a calibrated probability.
Vessels are CANDIDATES only -- not confirmed polluters.

SAFETY:
    - Returns [] if no AIS history exists -- never fabricates vessels.
    - Never substitutes random global vessels.
    - Deduplicates by MMSI (keeps highest attribution_score row).

FIXES (v2):
    - time_match_score: Previously grouped AIS pings by unique (date, hour),
      then divided by total trajectory timesteps. This gave a max score of
      ~0.25 for 96 timesteps (24h at 15-min) even for a perfect temporal match.
      FIX: For each AIS ping, compute minimum temporal distance to the nearest
      trajectory timestamp (minutes), then score_i = exp(-delta_t / TAU_MIN).
      Aggregate as mean(score_i) over all pings. Supports scores up to 1.0 for
      well-matching vessels.

    - Spatial matching: Now compares each AIS ping against the FULL trajectory
      corridor (envelope polygon bounding box + per-centroid distance), not just
      the centroid list.

    - source_region_score: Added — measures how close the vessel was to the
      estimated source region at the start of the simulation window.
"""

import math
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from db.connection import get_connection
from fusion.backtracking_service import haversine_km, is_valid_geo_coord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_SEARCH_RADIUS_KM: float = 50.0           # vessel search radius around trajectory
_MIN_AIS_PINGS: int = 2                   # minimum AIS pings to consider a vessel
_MAX_CANDIDATES: int = 20                 # maximum returned candidates
_OVERLAP_THRESHOLD_KM: float = 50.0      # distance threshold for overlap scoring
_ANOMALY_SCORE_MAX: float = 1.0          # normalise anomaly scores to [0, 1]
_TIME_DECAY_TAU_MIN: float = 30.0        # e-folding time for temporal score (minutes)
                                          # score = exp(-|delta_t| / TAU). At 30min gap -> 0.37
_SOURCE_REGION_RADIUS_KM: float = 80.0   # radius for source_region_score normalisation


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CandidateMatch:
    """
    A candidate vessel from historical AIS matching.
    All scores are transparent metrics -- NOT ground truth.
    """
    rank: int
    mmsi: str
    vessel_name: str
    imo: Optional[str]

    # Spatial / temporal metrics
    min_distance_km: float
    trajectory_overlap_score: float    # 0.0 - 1.0
    time_match_score: float            # 0.0 - 1.0 (FIXED: exponential decay, no hourly grouping)
    ais_anomaly_score: float           # 0.0 - 1.0 (0 if no record)
    distance_score: float              # 0.0 - 1.0
    source_region_score: float         # 0.0 - 1.0 (proximity to estimated source region)

    # Attribution
    attribution_score: float           # 0 - 100 investigation priority score
    matching_ais_pings: int

    # Correlated position (the AIS position closest to the particle cloud)
    lat: Optional[float]
    lon: Optional[float]
    position_timestamp: Optional[str]

    # Disclaimer
    disclaimer: str = (
        "This vessel is a CANDIDATE only. The attribution score is an "
        "investigation priority metric derived from spatial and temporal "
        "proximity -- it does NOT constitute legal evidence of causation."
    )


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------

def _min_distance_to_trajectory(
    vessel_lat: float,
    vessel_lon: float,
    trajectory_centroids: List[Tuple[float, float]],
) -> float:
    """
    Minimum haversine distance (km) from vessel position to any
    trajectory centroid (lat, lon) in the backward simulation.
    """
    if not trajectory_centroids:
        return float("inf")
    return min(
        haversine_km(vessel_lat, vessel_lon, c[0], c[1])
        for c in trajectory_centroids
    )


def _min_distance_to_envelope(
    vessel_lat: float,
    vessel_lon: float,
    envelope_polygon: List[Tuple[float, float]],
    trajectory_centroids: List[Tuple[float, float]],
) -> float:
    """
    Minimum haversine distance (km) from vessel position to either:
    - any vertex of the trajectory corridor envelope polygon, OR
    - any trajectory centroid.

    Uses whichever is smaller (corridor is the union of both).
    """
    d_cent = _min_distance_to_trajectory(vessel_lat, vessel_lon, trajectory_centroids)
    if not envelope_polygon:
        return d_cent
    d_env = min(
        haversine_km(vessel_lat, vessel_lon, p[0], p[1])
        for p in envelope_polygon
    )
    return min(d_cent, d_env)


# ---------------------------------------------------------------------------
# Time-match score (FIXED)
# ---------------------------------------------------------------------------

def _compute_time_match_score(
    pings: List[Dict[str, Any]],
    trajectory_timestamps: List[datetime],
) -> float:
    """
    Compute time_match_score using per-ping exponential temporal decay.

    For each AIS ping, find the nearest trajectory timestamp and compute:
        score_i = exp(-|delta_t_minutes| / TAU)
    where TAU = _TIME_DECAY_TAU_MIN (default 30 min).

    Aggregate: time_match_score = mean(score_i) across all pings.

    This correctly rewards:
    - Vessels with AIS pings close in time to trajectory timesteps (score → 1.0)
    - Vessels with large temporal gaps (score → 0.0)

    FIX vs previous: old code divided unique (date, hour) pairs by total_timesteps,
    giving a maximum of 24/96 = 0.25 for 24h simulations.

    Parameters
    ----------
    pings : list of ping dicts, each with key "ts" (datetime or timestamptz)
    trajectory_timestamps : list of datetime objects for each trajectory step

    Returns
    -------
    float in [0.0, 1.0]
    """
    if not pings or not trajectory_timestamps:
        return 0.0

    # Ensure trajectory timestamps are timezone-aware
    traj_ts: List[datetime] = []
    for t in trajectory_timestamps:
        if t is None:
            continue
        if hasattr(t, "tzinfo") and t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        traj_ts.append(t)

    if not traj_ts:
        return 0.0

    scores = []
    for ping in pings:
        ts = ping.get("ts")
        if ts is None:
            continue
        if hasattr(ts, "tzinfo") and ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        # Minimum absolute time difference to any trajectory timestamp (minutes)
        min_delta_min = min(
            abs((ts - t_traj).total_seconds()) / 60.0
            for t_traj in traj_ts
        )
        # Exponential decay score
        score_i = math.exp(-min_delta_min / _TIME_DECAY_TAU_MIN)
        scores.append(score_i)

    if not scores:
        return 0.0

    return round(min(1.0, sum(scores) / len(scores)), 4)


# ---------------------------------------------------------------------------
# Source region score
# ---------------------------------------------------------------------------

def _compute_source_region_score(
    pings: List[Dict[str, Any]],
    source_centroid_lat: float,
    source_centroid_lon: float,
    spill_timestamp: datetime,
    duration_hours: float,
) -> float:
    """
    Score how close the vessel was to the estimated source region at the
    START of the simulation window (spill_timestamp - duration_hours).

    Considers only AIS pings within ±3h of the source window start time.
    Score = 1 - min_dist / _SOURCE_REGION_RADIUS_KM (capped at 0).

    Returns 0.0 if vessel has no pings near the source time window.
    """
    if not pings:
        return 0.0

    source_time = spill_timestamp - timedelta(hours=duration_hours)
    window_start = source_time - timedelta(hours=3)
    window_end   = source_time + timedelta(hours=3)

    nearby_pings = []
    for ping in pings:
        ts = ping.get("ts")
        if ts is None:
            continue
        if hasattr(ts, "tzinfo") and ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if hasattr(spill_timestamp, "tzinfo") and spill_timestamp.tzinfo is None:
            spill_timestamp = spill_timestamp.replace(tzinfo=timezone.utc)
        if window_start.replace(tzinfo=timezone.utc if window_start.tzinfo is None else window_start.tzinfo) <= ts <= \
           window_end.replace(tzinfo=timezone.utc if window_end.tzinfo is None else window_end.tzinfo):
            nearby_pings.append(ping)

    if not nearby_pings:
        return 0.0

    min_dist = min(
        haversine_km(p["lat"], p["lon"], source_centroid_lat, source_centroid_lon)
        for p in nearby_pings
    )
    return round(max(0.0, 1.0 - min_dist / _SOURCE_REGION_RADIUS_KM), 4)


# ---------------------------------------------------------------------------
# Main matching function
# ---------------------------------------------------------------------------

def find_candidate_vessels(
    trajectory_centroids: List[Tuple[float, float]],
    spill_lat: float,
    spill_lon: float,
    spill_timestamp: datetime,
    duration_hours: float,
    source_centroid_lat: Optional[float] = None,
    source_centroid_lon: Optional[float] = None,
    envelope_polygon: Optional[List[Tuple[float, float]]] = None,
    trajectory_timestamps: Optional[List[datetime]] = None,
    conn=None,
) -> List[CandidateMatch]:
    """
    Query historical AIS data to find candidate vessels.

    Parameters
    ----------
    trajectory_centroids : list of (lat, lon) tuples -- backward trajectory centroids
    spill_lat, spill_lon : observed spill coordinates
    spill_timestamp      : when the spill was observed (timezone-aware)
    duration_hours       : backward simulation window
    source_centroid_lat/lon : centroid of estimated source region (from LagrangianModel)
    envelope_polygon     : convex hull of full particle corridor [(lat, lon), ...]
    trajectory_timestamps : list of datetime objects for each trajectory step (for time_match_score)
    conn                 : optional DB connection (created if None)

    Returns
    -------
    List of CandidateMatch sorted by attribution_score descending.
    Returns [] if no matching AIS history -- never fabricates vessels.

    COORDINATE NOTE:
        PostGIS ST_X(location) = longitude
        PostGIS ST_Y(location) = latitude
        Leaflet uses [lat, lon] -- conversion handled in API layer.
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    try:
        cur = conn.cursor()
        time_start = spill_timestamp - timedelta(hours=duration_hours)
        time_end   = spill_timestamp

        if not trajectory_centroids:
            logger.warning("No trajectory centroids supplied -- cannot match AIS vessels.")
            return []

        # Build a bounding box from both trajectory centroids AND envelope polygon
        lats = [c[0] for c in trajectory_centroids] + [spill_lat]
        lons = [c[1] for c in trajectory_centroids] + [spill_lon]
        if envelope_polygon:
            lats += [p[0] for p in envelope_polygon]
            lons += [p[1] for p in envelope_polygon]

        lat_min = min(lats) - (_SEARCH_RADIUS_KM / 111.0)
        lat_max = max(lats) + (_SEARCH_RADIUS_KM / 111.0)
        lon_min = min(lons) - (_SEARCH_RADIUS_KM / 111.0)
        lon_max = max(lons) + (_SEARCH_RADIUS_KM / 111.0)

        # PostGIS-accelerated query: bounding box + time filter
        # Ordering by mmsi, ts ensures stable grouping
        cur.execute("""
            SELECT
                p.mmsi,
                p.ship_name,
                p.imo,
                ST_Y(p.location::geometry) AS lat,
                ST_X(p.location::geometry) AS lon,
                p.ts,
                p.sog,
                p.cog
            FROM ais_positions p
            WHERE
                p.ts BETWEEN %s AND %s
                AND ST_Y(p.location::geometry) BETWEEN %s AND %s
                AND ST_X(p.location::geometry) BETWEEN %s AND %s
            ORDER BY p.mmsi, p.ts ASC
        """, (time_start, time_end, lat_min, lat_max, lon_min, lon_max))

        rows = cur.fetchall()
        if not rows:
            logger.info(
                "No AIS positions found in search corridor for time window %s to %s",
                time_start, time_end
            )
            return []

        # Group rows by MMSI
        vessels: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            mmsi, ship_name, imo, lat, lon, ts, sog, cog = row
            if not is_valid_geo_coord(lat, lon):
                continue
            if mmsi not in vessels:
                vessels[mmsi] = []
            vessels[mmsi].append({
                "mmsi":      mmsi,
                "ship_name": ship_name or f"Vessel {mmsi}",
                "imo":       imo,
                "lat":       float(lat),
                "lon":       float(lon),
                "ts":        ts,
                "sog":       sog,
                "cog":       cog,
            })

        # Query anomaly scores for all MMSIs found
        mmsi_list = list(vessels.keys())
        anomaly_scores: Dict[str, float] = {}
        if mmsi_list:
            placeholders = ",".join(["%s"] * len(mmsi_list))
            cur.execute(f"""
                SELECT vessel_id, MAX(anomaly_score)
                FROM ais_anomalies
                WHERE vessel_id = ANY(ARRAY[{placeholders}])
                  AND ts BETWEEN %s AND %s
                GROUP BY vessel_id
            """, (*mmsi_list, time_start, time_end))
            for anom_row in cur.fetchall():
                vid, score = anom_row
                if score is not None:
                    anomaly_scores[vid] = min(float(score), _ANOMALY_SCORE_MAX)

        cur.close()

        # Score each vessel
        candidates: List[CandidateMatch] = []

        for mmsi, pings in vessels.items():
            if len(pings) < _MIN_AIS_PINGS:
                continue

            # Per-ping distance to nearest trajectory centroid OR corridor envelope
            distances = []
            closest_ping = None
            min_dist = float("inf")

            for ping in pings:
                if envelope_polygon:
                    d = _min_distance_to_envelope(
                        ping["lat"], ping["lon"],
                        envelope_polygon, trajectory_centroids,
                    )
                else:
                    d = _min_distance_to_trajectory(
                        ping["lat"], ping["lon"], trajectory_centroids
                    )
                distances.append(d)
                if d < min_dist:
                    min_dist = d
                    closest_ping = ping

            min_dist_km = round(min_dist, 2)

            # trajectory_overlap_score: fraction of pings within corridor threshold
            pings_within = sum(1 for d in distances if d <= _OVERLAP_THRESHOLD_KM)
            trajectory_overlap_score = round(pings_within / max(1, len(pings)), 4)

            # time_match_score: FIXED -- exponential decay per ping (no hourly grouping)
            if trajectory_timestamps:
                time_match_score = _compute_time_match_score(pings, trajectory_timestamps)
            else:
                # Legacy fallback: build trajectory timestamps from centroids if not supplied
                # (Should not happen for new calls, but keeps old test compatibility)
                time_match_score = _legacy_time_match_score(pings, len(trajectory_centroids))

            # distance_score: 1 - min_dist/100km (capped at 0)
            distance_score = round(max(0.0, 1.0 - min_dist_km / 100.0), 4)

            # anomaly score from existing table
            ais_anomaly_score = round(anomaly_scores.get(mmsi, 0.0), 4)

            # source_region_score: proximity at start of simulation window
            if source_centroid_lat is not None and source_centroid_lon is not None:
                source_region_score = _compute_source_region_score(
                    pings, source_centroid_lat, source_centroid_lon,
                    spill_timestamp, duration_hours,
                )
            else:
                source_region_score = 0.0

            # Attribution score formula
            attribution_score = round((
                0.40 * trajectory_overlap_score +
                0.30 * time_match_score +
                0.20 * distance_score +
                0.10 * ais_anomaly_score
            ) * 100, 1)

            vessel_name = pings[-1].get("ship_name") or f"Vessel {mmsi}"
            raw_imo = pings[-1].get("imo")
            clean_imo = str(raw_imo) if raw_imo and str(raw_imo) != "UNKNOWN" else None

            closest_lat = closest_ping["lat"] if closest_ping else None
            closest_lon = closest_ping["lon"] if closest_ping else None
            closest_ts = (
                closest_ping["ts"].isoformat()
                if closest_ping and hasattr(closest_ping["ts"], "isoformat")
                else None
            )

            candidates.append(CandidateMatch(
                rank=0,
                mmsi=str(mmsi),
                vessel_name=vessel_name,
                imo=clean_imo,
                min_distance_km=min_dist_km,
                trajectory_overlap_score=trajectory_overlap_score,
                time_match_score=time_match_score,
                ais_anomaly_score=ais_anomaly_score,
                distance_score=distance_score,
                source_region_score=source_region_score,
                attribution_score=attribution_score,
                matching_ais_pings=len(pings),
                lat=closest_lat,
                lon=closest_lon,
                position_timestamp=closest_ts,
            ))

        # Sort by attribution score descending, cap at max
        candidates.sort(key=lambda c: c.attribution_score, reverse=True)
        ranked = candidates[:_MAX_CANDIDATES]
        for i, c in enumerate(ranked):
            c.rank = i + 1

        logger.info(
            "Historical AIS matching: found %d candidate vessels from %d total AIS records",
            len(ranked), len(rows)
        )
        return ranked

    except Exception as exc:
        logger.error("Error in historical AIS matching: %s", exc, exc_info=True)
        return []

    finally:
        if close_conn and conn:
            conn.close()


def _legacy_time_match_score(
    pings: List[Dict[str, Any]],
    total_timesteps: int,
) -> float:
    """
    LEGACY fallback for callers that do not supply trajectory_timestamps.
    Uses the old (buggy) formula but with documentation.
    Only invoked for backward-compatibility with tests that don't supply timestamps.

    NOTE: This gives artificially low scores for 15-min trajectory timesteps.
    Prefer supplying trajectory_timestamps to use the new exponential decay formula.
    """
    if total_timesteps == 0:
        return 0.0
    ping_hours = set()
    for ping in pings:
        ts = ping.get("ts")
        if ts is not None and hasattr(ts, "hour"):
            ping_hours.add((ts.date(), ts.hour))
    # Normalise by hours in duration (not raw timesteps) to avoid the 24/96 bug
    duration_hours = max(1, total_timesteps * 15 // 60)
    return round(min(1.0, len(ping_hours) / max(1, duration_hours)), 4)
