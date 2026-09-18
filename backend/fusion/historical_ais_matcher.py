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
    time_match_score: float            # 0.0 - 1.0
    ais_anomaly_score: float           # 0.0 - 1.0 (0 if no record)
    distance_score: float              # 0.0 - 1.0

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
# Haversine distance from a point to the nearest trajectory centroid
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


# ---------------------------------------------------------------------------
# Main matching function
# ---------------------------------------------------------------------------

def find_candidate_vessels(
    trajectory_centroids: List[Tuple[float, float]],
    spill_lat: float,
    spill_lon: float,
    spill_timestamp: datetime,
    duration_hours: float,
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
        time_end = spill_timestamp

        if not trajectory_centroids:
            logger.warning("No trajectory centroids supplied -- cannot match AIS vessels.")
            return []

        # Build a simple bounding box from trajectory centroids for pre-filter
        lats = [c[0] for c in trajectory_centroids] + [spill_lat]
        lons = [c[1] for c in trajectory_centroids] + [spill_lon]
        lat_min = min(lats) - (_SEARCH_RADIUS_KM / 111.0)
        lat_max = max(lats) + (_SEARCH_RADIUS_KM / 111.0)
        lon_min = min(lons) - (_SEARCH_RADIUS_KM / 111.0)
        lon_max = max(lons) + (_SEARCH_RADIUS_KM / 111.0)

        # Query AIS positions within bounding box and time window
        # ST_X = longitude, ST_Y = latitude (PostGIS convention)
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
            logger.info("No AIS positions found in search corridor for time window %s to %s", time_start, time_end)
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
                "mmsi": mmsi,
                "ship_name": ship_name or f"Vessel {mmsi}",
                "imo": imo,
                "lat": float(lat),
                "lon": float(lon),
                "ts": ts,
                "sog": sog,
                "cog": cog,
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
        total_timesteps = len(trajectory_centroids)
        candidates: List[CandidateMatch] = []

        for mmsi, pings in vessels.items():
            if len(pings) < _MIN_AIS_PINGS:
                continue

            # Per-ping distance to nearest trajectory centroid
            distances = []
            closest_ping = None
            min_dist = float("inf")

            for ping in pings:
                d = _min_distance_to_trajectory(
                    ping["lat"], ping["lon"], trajectory_centroids
                )
                distances.append(d)
                if d < min_dist:
                    min_dist = d
                    closest_ping = ping

            min_dist_km = round(min_dist, 2)

            # trajectory_overlap_score: fraction of pings within threshold
            pings_within = sum(1 for d in distances if d <= _OVERLAP_THRESHOLD_KM)
            trajectory_overlap_score = round(pings_within / max(1, len(pings)), 3)

            # time_match_score: fraction of trajectory timesteps with >=1 nearby ping
            # (simplified: how many unique hours covered vs total duration)
            if total_timesteps > 0:
                ping_hours = set()
                for ping in pings:
                    ts = ping["ts"]
                    if hasattr(ts, "hour"):
                        ping_hours.add((ts.date(), ts.hour))
                time_match_score = round(min(1.0, len(ping_hours) / max(1, total_timesteps)), 3)
            else:
                time_match_score = 0.0

            # distance_score: 1 - min_dist/100km (capped at 0)
            distance_score = round(max(0.0, 1.0 - min_dist_km / 100.0), 3)

            # anomaly score from existing table
            ais_anomaly_score = round(anomaly_scores.get(mmsi, 0.0), 3)

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
                rank=0,          # assigned after sorting
                mmsi=str(mmsi),
                vessel_name=vessel_name,
                imo=clean_imo,
                min_distance_km=min_dist_km,
                trajectory_overlap_score=trajectory_overlap_score,
                time_match_score=time_match_score,
                ais_anomaly_score=ais_anomaly_score,
                distance_score=distance_score,
                attribution_score=attribution_score,
                matching_ais_pings=len(pings),
                lat=closest_lat,
                lon=closest_lon,
                position_timestamp=closest_ts,
            ))

        # Sort by attribution score descending, deduplicate by MMSI (already unique)
        candidates.sort(key=lambda c: c.attribution_score, reverse=True)

        # Assign ranks and cap
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
