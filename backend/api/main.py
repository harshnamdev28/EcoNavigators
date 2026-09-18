"""
FastAPI Backend — Oil Spill Detection System
=============================================
Run with: uvicorn api.main:app --reload --port 8000
Docs at:  http://localhost:8000/docs

Changes vs prior version
-------------------------
* CORS uses CORS_ALLOWED_ORIGINS env var — no wildcard in production
* Alert acknowledgements persisted to database (not in-memory)
* Pipeline states returned at every analysis stage
* Slick centroid is geographic, not vessel position
* Timestamps never mutated
"""
import logging
import time
import uuid
from fastapi import FastAPI, HTTPException, APIRouter, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timezone, timedelta
import numpy as np

from config import api_cfg, sar_cfg, fusion_cfg, ais_cfg, PipelineState
from fusion.fusion_engine import fuse_db
from fusion.source_attribution import score_candidates, haversine_km
from db.auth import create_user, verify_user, generate_session_token, verify_session_token
from db.connection import get_connection

from ais_pipeline.behavior_anomaly_detector import gatekeep_anomaly
from sar_pipeline.sar_unet_detector import SARUNetDetector
from sar_pipeline.sar_service import (
    init_sar_model,
    analyze_sar_image,
    is_sar_model_loaded,
    get_sar_model,
)
from fusion.backtracking_service import (
    get_vessel_trajectory,
    simulate_lagrangian_backtrack,
    update_sar_characterization_cache,
)
from sar_pipeline.copernicus_service import (
    fetch_copernicus_satellite_image,
    parse_timestamp,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Oil Spill Detection API",
    description="Maritime oil-spill detection using AIS + Sentinel-1 SAR + UNet segmentation.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=api_cfg.CORS_ORIGINS,  # configurable via CORS_ALLOWED_ORIGINS env var
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter(prefix="/api/v1")


@app.on_event("startup")
def startup_event():
    """Pre-load standalone MiT-B2 SAR segmentation model on startup."""
    try:
        init_sar_model()
        logger.info("Standalone MiT-B2 UNet model loaded on startup.")
    except Exception as exc:
        logger.error("Could not preload MiT-B2 UNet model on startup: %s", exc)


NAV_STATUS = {
    0: "Underway", 1: "At anchor", 2: "Not under command",
    3: "Restricted manoeuvrability", 4: "Constrained by draught",
    5: "Moored", 6: "Aground", 7: "Engaged in fishing", 8: "Underway sailing",
}


# ---------- Auth (Available both at /api and /api/v1/auth) ----------

class SignupRequest(BaseModel):
    username: str
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


@app.post("/api/signup")
@router.post("/auth/signup")
def signup(payload: SignupRequest):
    try:
        user = create_user(payload.username, payload.email, payload.password)
        token = generate_session_token(user["id"], user["email"], user["username"])
        return {"message": "Signup successful", "user": user, "token": token}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Signup failed: {str(e)}")


@app.post("/api/login")
@router.post("/auth/login")
def login(payload: LoginRequest):
    user = verify_user(payload.email, payload.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = generate_session_token(user["id"], user["email"], user["username"])
    return {"message": "Login successful", "user": user, "token": token}


@router.get("/auth/me")
def get_current_user(token: Optional[str] = None):
    if not token:
        raise HTTPException(status_code=401, detail="Missing authorization token")
    payload = verify_session_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return {"user": payload}


# ---------- Incident Persistence & Identify Pipeline ----------

_incidents_store: Dict[str, Dict[str, Any]] = {}


def validate_coords(lat: Any, lon: Any) -> Tuple[Optional[float], Optional[float]]:
    """
    Validates geographic coordinates:
    - Rejects Null Island (0.0, 0.0) as unobserved/invalid.
    - Rejects values outside standard latitude [-90, 90] and longitude [-180, 180].
    - Preserves None without substituting default coordinates.
    """
    if lat is None or lon is None:
        return None, None
    try:
        f_lat = float(lat)
        f_lon = float(lon)
    except (ValueError, TypeError):
        return None, None

    # Strict check: 0.0, 0.0 is Null Island (invalid GPS / default reset)
    if abs(f_lat) < 1e-6 and abs(f_lon) < 1e-6:
        return None, None

    if not (-90.0 <= f_lat <= 90.0) or not (-180.0 <= f_lon <= 180.0):
        return None, None

    return f_lat, f_lon


def generate_event_key(mmsi: str, ts_val: Any, anomaly_type: str = "NORMAL") -> str:
    """
    Generates a deterministic unique event key for an AIS vessel event:
    {clean_mmsi}_{UTC timestamp formatted to second}_{clean anomaly_type}
    """
    clean_mmsi = str(mmsi).replace("MMSI_", "").strip()
    if isinstance(ts_val, str):
        try:
            dt = datetime.fromisoformat(ts_val.replace("Z", "+00:00"))
        except Exception:
            dt = datetime.now(timezone.utc)
    elif isinstance(ts_val, datetime):
        dt = ts_val
    else:
        dt = datetime.now(timezone.utc)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    utc_str = dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    clean_anom = str(anomaly_type).strip() if anomaly_type else "NORMAL"
    return f"{clean_mmsi}_{utc_str}_{clean_anom}"


class IdentifyRequest(BaseModel):
    mmsi: str
    timestamp: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None


@router.post("/identify")
def identify_vessel(payload: IdentifyRequest):
    """
    Real data-driven identify pipeline with deterministic idempotency:
    1. Validates MMSI and explicit coordinates (rejects invalid / Null Island 0,0).
    2. Retrieves real AIS position and metadata for vessel.
    3. Evaluates real AIS anomaly with configurable thresholds.
    4. Sets SAR status = DEFERRED (no Copernicus/U-Net calls).
    5. Determines canonical incident state (AIS_ANOMALY, AIS_NORMAL).
    6. Checks for existing equivalent incident before creating a new one (Idempotency).
    7. Creates or returns canonical Incident object.
    """
    clean_mmsi = str(payload.mmsi).replace("MMSI_", "").strip()
    if not clean_mmsi:
        raise HTTPException(status_code=400, detail="MMSI is required")

    conn = get_connection()
    cur = conn.cursor()

    # 1. Lookup vessel by MMSI in ais_positions
    cur.execute("""
        SELECT ship_name, ship_type, imo, sog, heading, draught, status, ts,
               ST_X(location::geometry) AS lon, ST_Y(location::geometry) AS lat
        FROM ais_positions WHERE mmsi = %s ORDER BY ts DESC LIMIT 1
    """, (clean_mmsi,))
    pos_row = cur.fetchone()

    # 2. Coordinates & Timestamp handling
    if payload.timestamp:
        vessel_ts = payload.timestamp
    elif pos_row and pos_row[7]:
        vessel_ts = pos_row[7].isoformat() if hasattr(pos_row[7], "isoformat") else str(pos_row[7])
    else:
        vessel_ts = datetime.now(timezone.utc).isoformat()

    # Validate payload coords if explicitly provided:
    if payload.lat is not None or payload.lon is not None:
        vessel_lat, vessel_lon = validate_coords(payload.lat, payload.lon)
    else:
        # Fall back to DB position ONLY if event timestamp matches pos_row timestamp (within 10 minutes)
        # or if payload.timestamp was omitted (meaning the event IS the latest recorded vessel position).
        # Never fabricate coordinates or use fallback locations from unrelated timestamps!
        pos_ts = pos_row[7] if pos_row and pos_row[7] else None
        coords_time_compatible = False
        if pos_ts and payload.timestamp:
            try:
                p_dt = datetime.fromisoformat(str(payload.timestamp).replace("Z", "+00:00"))
                r_dt = pos_ts if isinstance(pos_ts, datetime) else datetime.fromisoformat(str(pos_ts).replace("Z", "+00:00"))
                if p_dt.tzinfo is None:
                    p_dt = p_dt.replace(tzinfo=timezone.utc)
                if r_dt.tzinfo is None:
                    r_dt = r_dt.replace(tzinfo=timezone.utc)
                if abs((p_dt - r_dt).total_seconds()) <= 600:
                    coords_time_compatible = True
            except Exception:
                coords_time_compatible = False
        elif not payload.timestamp:
            coords_time_compatible = True

        if coords_time_compatible:
            db_lat = float(pos_row[9]) if pos_row and pos_row[9] is not None else None
            db_lon = float(pos_row[8]) if pos_row and pos_row[8] is not None else None
            vessel_lat, vessel_lon = validate_coords(db_lat, db_lon)
        else:
            vessel_lat, vessel_lon = None, None

    # 3. Canonical vessel metadata (never fabricate IMO)
    raw_imo = pos_row[2] if pos_row else None
    clean_imo = str(raw_imo) if raw_imo and str(raw_imo) != "UNKNOWN" and str(raw_imo) != "N/A" and not str(raw_imo).startswith("IMO-") else None
    ship_name = pos_row[0] if pos_row and pos_row[0] else f"Vessel {clean_mmsi}"
    ship_type = pos_row[1] if pos_row and pos_row[1] else "Commercial Vessel"

    # 4. Run real AIS anomaly detection using configured thresholds
    gatekeeper = gatekeep_anomaly(clean_mmsi, cur=cur)
    is_anomalous = bool(gatekeeper.get("anomalous", False))
    anomaly_score = float(gatekeeper.get("anomaly_score", 0.0))
    anomaly_type = gatekeeper.get("anomaly_type", "NORMAL")
    reasons = gatekeeper.get("reasons", [])

    # 5. Determine Incident State & SAR deferred status
    state = "AIS_ANOMALY" if is_anomalous else "AIS_NORMAL"

    sar_status = {
        "status": "DEFERRED",
        "reason": "SAR pipeline deferred for future integration",
    }

    # Generate deterministic event_key for DB idempotency
    event_key = generate_event_key(clean_mmsi, vessel_ts, anomaly_type)

    # 6. IDEMPOTENCY CHECK:
    # Check whether an equivalent active/recent incident already exists for this vessel event
    try:
        # Parse event timestamp to datetime for time comparison
        event_dt = None
        if vessel_ts:
            try:
                event_dt = datetime.fromisoformat(str(vessel_ts).replace("Z", "+00:00"))
            except Exception:
                pass

        cur.execute("""
            SELECT id, mmsi, timestamp, latitude, longitude, state, anomaly_detected, anomaly_score, sar_status, anomaly_type, event_key
            FROM incidents
            WHERE event_key = %s OR mmsi = %s
            ORDER BY created_at DESC
            LIMIT 20
        """, (event_key, clean_mmsi))
        existing_rows = cur.fetchall()

        for er in existing_rows:
            e_id, e_mmsi, e_ts, e_lat, e_lon, e_state, e_anom_det, e_anom_sc, e_sar, e_anom_type, e_ev_key = er

            matched = False
            is_explicit_zero = (payload.lat is not None and abs(float(payload.lat)) < 1e-6 and payload.lon is not None and abs(float(payload.lon)) < 1e-6)
            is_explicit_invalid = (payload.lat is not None or payload.lon is not None) and (vessel_lat is None and vessel_lon is None)

            # Check coordinates match within 0.01 deg tolerance (~1 km) or unobserved
            coords_matched = False
            if vessel_lat is None and e_lat is None:
                coords_matched = True
            elif vessel_lat is not None and e_lat is not None and vessel_lon is not None and e_lon is not None:
                if abs(float(e_lat) - float(vessel_lat)) <= 0.01 and abs(float(e_lon) - float(vessel_lon)) <= 0.01:
                    coords_matched = True
            elif not (is_explicit_zero or is_explicit_invalid) and (vessel_lat is None or vessel_lon is None) and (e_lat is not None and e_lon is not None):
                # Request omitted coords, existing has valid coords -> compatible
                coords_matched = True
            elif not (is_explicit_zero or is_explicit_invalid) and (vessel_lat is not None and vessel_lon is not None) and (e_lat is None or e_lon is None):
                # Canonical incident previously lacked coordinates, now valid coords arrived -> compatible
                coords_matched = True

            # If request provided explicit valid coordinates, they MUST match existing row's coords (if existing has coords)
            if e_ev_key == event_key:
                if vessel_lat is not None and e_lat is not None and not coords_matched:
                    # Explicit different coordinates -> different event
                    matched = False
                else:
                    matched = True
            else:
                # Check state match
                if e_state != state:
                    continue

                # Check timestamp match (exact string or within 10-minute tolerance)
                ts_matched = False
                if e_ts and event_dt:
                    e_dt = e_ts if isinstance(e_ts, datetime) else None
                    if e_dt is None:
                        try:
                            e_dt = datetime.fromisoformat(str(e_ts).replace("Z", "+00:00"))
                        except Exception:
                            pass
                    if e_dt:
                        if e_dt.tzinfo is None:
                            e_dt = e_dt.replace(tzinfo=timezone.utc)
                        if event_dt.tzinfo is None:
                            event_dt = event_dt.replace(tzinfo=timezone.utc)
                        if abs((e_dt - event_dt).total_seconds()) <= 600:  # 10 minute tolerance window
                            ts_matched = True
                elif str(e_ts) == str(vessel_ts):
                    ts_matched = True

                if ts_matched and coords_matched:
                    matched = True

            if matched:
                # Coordinate preservation and backfill:
                # 1. Existing coordinates from DB row (e_lat, e_lon)
                final_lat, final_lon = validate_coords(e_lat, e_lon)

                # 2. Check in-memory cache if DB coords were missing
                cached_existing = _incidents_store.get(e_id)
                if final_lat is None and cached_existing:
                    c_lat = cached_existing.get("location", {}).get("lat")
                    c_lon = cached_existing.get("location", {}).get("lon")
                    final_lat, final_lon = validate_coords(c_lat, c_lon)

                # 3. If incident lacks coordinates and incoming request has valid coordinates, update/backfill
                if (final_lat is None or final_lon is None) and vessel_lat is not None and vessel_lon is not None:
                    final_lat = vessel_lat
                    final_lon = vessel_lon
                    try:
                        cur.execute("""
                            UPDATE incidents
                            SET latitude = %s, longitude = %s
                            WHERE id = %s
                        """, (final_lat, final_lon, e_id))
                        conn.commit()
                    except Exception as upd_err:
                        logger.warning("[Identify] Failed to backfill coords on existing incident: %s", upd_err)

                # 4. Existing canonical incident found! Return without creating duplicate row
                # Never replace valid coordinates with NULL
                existing_inc = {
                    "incidentId": e_id,
                    "mmsi": clean_mmsi,
                    "timestamp": payload.timestamp if payload.timestamp else (e_ts.isoformat() if hasattr(e_ts, "isoformat") else str(e_ts)),
                    "lat": final_lat,
                    "lon": final_lon,
                    "latitude": final_lat,
                    "longitude": final_lon,
                    "location": {
                        "lat": final_lat,
                        "lon": final_lon,
                    },
                    "state": e_state,
                    "ais": {
                        "anomalyDetected": bool(e_anom_det),
                        "reason": "; ".join(reasons) if reasons else ("Normal commercial navigation confirmed" if not e_anom_det else "AIS anomaly detected"),
                        "anomalyType": e_anom_type or anomaly_type,
                        "confidence": float(e_anom_sc) if e_anom_sc is not None else anomaly_score,
                        "metrics": gatekeeper.get("metrics", {}),
                    },
                    "sar": sar_status,
                    "vessel": {
                        "mmsi": clean_mmsi,
                        "name": ship_name,
                        "type": ship_type,
                        "imo": clean_imo,
                        "speed": f"{pos_row[3]:.1f} kts" if pos_row and pos_row[3] is not None else "N/A",
                        "heading": f"{pos_row[4]:.0f}°" if pos_row and pos_row[4] is not None and pos_row[4] != 511 else "N/A",
                        "status": NAV_STATUS.get(pos_row[6], "Underway") if pos_row else "Unknown",
                    },
                    "backtracking": {
                        "status": "READY",
                    },
                }
                _incidents_store[e_id] = existing_inc
                _incidents_store[clean_mmsi] = existing_inc
                conn.close()
                return existing_inc

    except Exception as match_err:
        logger.warning("[Identify] Idempotency lookup warning: %s", match_err)

    # 7. Create real Incident ID only when no matching incident exists
    generated_id = f"INC-{clean_mmsi}-{uuid.uuid4().hex[:8].upper()}"

    # Persist to DB table with transactional concurrency protection (ON CONFLICT on event_key)
    persisted_id = generated_id
    persisted_lat = vessel_lat
    persisted_lon = vessel_lon
    try:
        cur.execute("""
            INSERT INTO incidents (
                id, mmsi, timestamp, latitude, longitude, state,
                anomaly_detected, anomaly_score, sar_status, anomaly_type, event_key
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (event_key) DO UPDATE SET
                latitude = COALESCE(EXCLUDED.latitude, incidents.latitude),
                longitude = COALESCE(EXCLUDED.longitude, incidents.longitude),
                state = EXCLUDED.state,
                anomaly_detected = EXCLUDED.anomaly_detected,
                anomaly_score = EXCLUDED.anomaly_score,
                sar_status = EXCLUDED.sar_status
            RETURNING id, latitude, longitude;
        """, (
            generated_id, clean_mmsi, vessel_ts, vessel_lat, vessel_lon,
            state, is_anomalous, anomaly_score, "DEFERRED", anomaly_type, event_key
        ))
        row = cur.fetchone()
        if row:
            persisted_id = row[0]
            persisted_lat = row[1]
            persisted_lon = row[2]
        conn.commit()
    except Exception as db_err:
        logger.warning("[Identify] DB persistence warning: %s", db_err)
    finally:
        conn.close()

    valid_persisted_lat, valid_persisted_lon = validate_coords(persisted_lat, persisted_lon)

    incident = {
        "incidentId": persisted_id,
        "mmsi": clean_mmsi,
        "timestamp": vessel_ts,
        "lat": valid_persisted_lat,
        "lon": valid_persisted_lon,
        "latitude": valid_persisted_lat,
        "longitude": valid_persisted_lon,
        "location": {
            "lat": valid_persisted_lat,
            "lon": valid_persisted_lon,
        },
        "state": state,
        "ais": {
            "anomalyDetected": is_anomalous,
            "reason": "; ".join(reasons) if reasons else ("Normal commercial navigation confirmed" if not is_anomalous else "AIS anomaly detected"),
            "anomalyType": anomaly_type,
            "confidence": anomaly_score,
            "metrics": gatekeeper.get("metrics", {}),
        },
        "sar": sar_status,
        "vessel": {
            "mmsi": clean_mmsi,
            "name": ship_name,
            "type": ship_type,
            "imo": clean_imo,
            "speed": f"{pos_row[3]:.1f} kts" if pos_row and pos_row[3] is not None else "N/A",
            "heading": f"{pos_row[4]:.0f}°" if pos_row and pos_row[4] is not None and pos_row[4] != 511 else "N/A",
            "status": NAV_STATUS.get(pos_row[6], "Underway") if pos_row else "Unknown",
        },
        "backtracking": {
            "status": "READY",
        },
    }

    # 8. Persist incident in-memory cache
    _incidents_store[persisted_id] = incident
    _incidents_store[clean_mmsi] = incident

    return incident


@router.get("/incidents/{incident_id}")
def get_incident(incident_id: str):
    clean_id = incident_id.strip()

    # 1. DB-first lookup
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT id, mmsi, timestamp, latitude, longitude, state, anomaly_detected, anomaly_score, sar_status, created_at, anomaly_type
            FROM incidents
            WHERE id = %s OR mmsi = %s
            ORDER BY created_at DESC LIMIT 1
        """, (clean_id, clean_id))
        row = cur.fetchone()
        conn.close()
        if row:
            cached = _incidents_store.get(row[0])
            db_lat, db_lon = validate_coords(row[3], row[4])
            cached_lat = cached.get("location", {}).get("lat") if cached and isinstance(cached.get("location"), dict) else None
            cached_lon = cached.get("location", {}).get("lon") if cached and isinstance(cached.get("location"), dict) else None
            valid_cached_lat, valid_cached_lon = validate_coords(cached_lat, cached_lon)

            # Never replace valid coordinates with NULL
            final_lat = db_lat if db_lat is not None else valid_cached_lat
            final_lon = db_lon if db_lon is not None else valid_cached_lon

            if cached:
                if not isinstance(cached.get("location"), dict):
                    cached["location"] = {}
                cached["location"]["lat"] = final_lat
                cached["location"]["lon"] = final_lon
                cached["lat"] = final_lat
                cached["lon"] = final_lon
                cached["latitude"] = final_lat
                cached["longitude"] = final_lon
                return cached
            return {
                "incidentId": row[0],
                "mmsi": row[1],
                "timestamp": row[2].isoformat() if hasattr(row[2], "isoformat") else str(row[2]),
                "lat": final_lat,
                "lon": final_lon,
                "latitude": final_lat,
                "longitude": final_lon,
                "location": {
                    "lat": final_lat,
                    "lon": final_lon,
                },
                "state": row[5],
                "ais": {
                    "anomalyDetected": bool(row[6]),
                    "reason": "AIS anomaly detected" if row[6] else "Normal commercial navigation confirmed",
                    "anomalyType": row[10] or ("AIS_ANOMALY" if row[6] else "NORMAL"),
                    "confidence": float(row[7]) if row[7] is not None else 0.0,
                    "metrics": {},
                },
                "sar": {
                    "status": row[8] or "DEFERRED",
                    "reason": "SAR pipeline deferred for future integration",
                },
                "vessel": {
                    "mmsi": row[1],
                    "name": f"Vessel {row[1]}",
                    "type": "Commercial Vessel",
                    "imo": None,
                    "speed": "N/A",
                    "heading": "N/A",
                    "status": "Unknown",
                },
                "backtracking": {
                    "status": "READY",
                },
                "created_at": row[9].isoformat() if hasattr(row[9], "isoformat") else str(row[9]),
            }
    except Exception as e:
        logger.warning("[get_incident] DB lookup warning: %s", e)

    # 2. In-memory fallback
    inc = _incidents_store.get(clean_id)
    if not inc:
        for item in _incidents_store.values():
            if item.get("mmsi") == clean_id:
                return item
        raise HTTPException(status_code=404, detail=f"Incident '{clean_id}' not found")
    return inc


@router.get("/incidents")
def list_incidents():
    seen = set()
    result = []

    # 1. DB-first lookup
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT id, mmsi, timestamp, latitude, longitude, state, anomaly_detected, anomaly_score, sar_status, created_at, anomaly_type
            FROM incidents
            ORDER BY created_at DESC LIMIT 100
        """)
        rows = cur.fetchall()
        conn.close()
        for row in rows:
            iid = row[0]
            if iid in seen:
                continue
            seen.add(iid)
            cached = _incidents_store.get(iid)
            db_lat, db_lon = validate_coords(row[3], row[4])
            cached_lat = cached.get("location", {}).get("lat") if cached and isinstance(cached.get("location"), dict) else None
            cached_lon = cached.get("location", {}).get("lon") if cached and isinstance(cached.get("location"), dict) else None
            valid_cached_lat, valid_cached_lon = validate_coords(cached_lat, cached_lon)

            # Never replace valid coordinates with NULL
            final_lat = db_lat if db_lat is not None else valid_cached_lat
            final_lon = db_lon if db_lon is not None else valid_cached_lon

            if cached:
                if not isinstance(cached.get("location"), dict):
                    cached["location"] = {}
                cached["location"]["lat"] = final_lat
                cached["location"]["lon"] = final_lon
                cached["lat"] = final_lat
                cached["lon"] = final_lon
                cached["latitude"] = final_lat
                cached["longitude"] = final_lon
                result.append(cached)
            else:
                result.append({
                    "incidentId": row[0],
                    "mmsi": row[1],
                    "timestamp": row[2].isoformat() if hasattr(row[2], "isoformat") else str(row[2]),
                    "lat": final_lat,
                    "lon": final_lon,
                    "latitude": final_lat,
                    "longitude": final_lon,
                    "location": {
                        "lat": final_lat,
                        "lon": final_lon,
                    },
                    "state": row[5],
                    "ais": {
                        "anomalyDetected": bool(row[6]),
                        "reason": "AIS anomaly detected" if row[6] else "Normal commercial navigation confirmed",
                        "anomalyType": row[10] or ("AIS_ANOMALY" if row[6] else "NORMAL"),
                        "confidence": float(row[7]) if row[7] is not None else 0.0,
                        "metrics": {},
                    },
                    "sar": {
                        "status": row[8] or "DEFERRED",
                        "reason": "SAR pipeline deferred for future integration",
                    },
                    "vessel": {
                        "mmsi": row[1],
                        "name": f"Vessel {row[1]}",
                        "type": "Commercial Vessel",
                        "imo": None,
                        "speed": "N/A",
                        "heading": "N/A",
                        "status": "Unknown",
                    },
                    "backtracking": {
                        "status": "READY",
                    },
                    "created_at": row[9].isoformat() if hasattr(row[9], "isoformat") else str(row[9]),
                })
    except Exception as e:
        logger.warning("[list_incidents] DB lookup warning: %s", e)

    # 2. Append any in-memory instances not in DB
    for inc in _incidents_store.values():
        iid = inc.get("incidentId")
        if iid and iid not in seen:
            seen.add(iid)
            loc = inc.get("location") if isinstance(inc.get("location"), dict) else {}
            lat_val = inc.get("lat") or inc.get("latitude") or loc.get("lat")
            lon_val = inc.get("lon") or inc.get("longitude") or loc.get("lon")
            v_lat, v_lon = validate_coords(lat_val, lon_val)
            inc["lat"] = v_lat
            inc["lon"] = v_lon
            inc["latitude"] = v_lat
            inc["longitude"] = v_lon
            result.append(inc)

    return result


# ---------- Shared helpers ----------

def _get_ais_anomalies(cur):
    cur.execute("""
        SELECT vessel_id, ST_X(location::geometry) AS lon, ST_Y(location::geometry) AS lat,
               ts, anomaly_type, anomaly_score
        FROM ais_anomalies
    """)
    rows = cur.fetchall()
    return [
        {"vessel_id": r[0], "lon": r[1], "lat": r[2], "timestamp": str(r[3]), "anomaly_type": r[4], "anomaly_score": r[5]}
        for r in rows
    ]


def _spill_status(spill_probability: float) -> str:
    if spill_probability is None:
        return "WARNING"
    if spill_probability > 0.7:
        return "CRITICAL"
    if spill_probability > 0.4:
        return "WARNING"
    return "RESOLVED"


def _risk_from_score(score: float) -> str:
    if score is None:
        return "Low"
    if score >= 0.7:
        return "High"
    if score >= 0.4:
        return "Medium"
    return "Low"


# ---------- 1. Dashboard ----------

@router.get("/dashboard/stats")
def get_dashboard_stats():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(DISTINCT mmsi) FROM ais_positions")
    active_vessels = cur.fetchone()[0]

    cur.execute("""
        SELECT COUNT(*) FROM (
            SELECT mmsi FROM ais_positions GROUP BY mmsi HAVING COUNT(*) > 1
        ) AS routed_vessels
    """)
    tracked_routes = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM sar_detections")
    detected_spills = cur.fetchone()[0]

    conn.close()
    return {
        "activeVessels": active_vessels,
        "trackedRoutes": tracked_routes,
        "detectedSpills": detected_spills,
    }


@router.get("/dashboard/candidate")
def get_dashboard_candidate():
    """Returns the single highest-confidence current alert's vessel, enriched
    with live position details, to power the dashboard's 'candidate vessel' card."""
    conn = get_connection()
    cur = conn.cursor()

    ais_list = _get_ais_anomalies(cur)
    if not ais_list:
        conn.close()
        raise HTTPException(status_code=404, detail="No AIS anomalies available yet")

    alerts = fuse_db(ais_list, cur)
    top = alerts[0]  # already sorted by combined_confidence descending
    mmsi = top["vessel_id"]

    cur.execute("""
        SELECT ship_name, ship_type, imo, sog, heading, draught, status, ts
        FROM ais_positions WHERE mmsi = %s ORDER BY ts DESC LIMIT 1
    """, (mmsi,))
    row = cur.fetchone()
    conn.close()

    ship_name = row[0] if row else mmsi
    ship_type = row[1] if row and row[1] else "Unknown"
    imo = row[2] if row and row[2] else "N/A"
    speed = f"{row[3]:.1f} kts" if row and row[3] is not None else "N/A"
    heading = f"{row[4]:.0f}°" if row and row[4] is not None and row[4] != 511 else "N/A"
    draft = f"{row[5]:.1f} m" if row and row[5] is not None else "N/A"
    status = NAV_STATUS.get(row[6], "Unknown") if row else "Unknown"

    return {
        "id": mmsi,
        "name": ship_name,
        "type": ship_type,
        "mmsi": mmsi,
        "imo": imo,
        "speed": speed,
        "heading": heading,
        "draft": draft,
        "status": status,
        "evidenceStrength": round(top["combined_confidence"] * 100),
        "lastFix": str(row[7]) if row else top["timestamp"],
        "coordinates": f"{top['location']['lat']:.4f}, {top['location']['lon']:.4f}",
        "lat": top['location']['lat'],
        "lng": top['location']['lon'],
    }


# ---------- 2. Oil Spills ----------

@router.get("/spills")
def list_spills():
    """
    Returns ONLY confirmed oil spill detections where:
    - A real slick was detected (not a NEGATIVE result)
    - A candidate vessel MMSI was attributed
    Filters out INCONCLUSIVE/NEGATIVE records and those with no vessel attribution.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT
            sd.id,
            sd.scene_id,
            sd.acquisition_time,
            ST_X(sd.aoi_location::geometry) AS lon,
            ST_Y(sd.aoi_location::geometry) AS lat,
            sd.spill_probability,
            sd.candidate_mmsi,
            sd.association_status,
            COALESCE(ap.ship_name, sd.candidate_mmsi) AS vessel_name,
            COALESCE(ap.ship_type, 'Unknown Vessel Type') AS vessel_type
        FROM sar_detections sd
        LEFT JOIN LATERAL (
            SELECT ship_name, ship_type FROM ais_positions
            WHERE mmsi = sd.candidate_mmsi
            ORDER BY ts DESC LIMIT 1
        ) ap ON TRUE
        WHERE
            sd.candidate_mmsi IS NOT NULL
            AND sd.association_status IN (
                'POSITIVE',
                'STRONGLY_ASSOCIATED',
                'SPATIALLY_COINCIDENT',
                'PLAUSIBLE_ASSOCIATION'
            )
        ORDER BY sd.acquisition_time DESC
    """)
    rows = cur.fetchall()
    conn.close()

    return [
        {
            "id":              f"Spill-{r[0]}",
            "coordinates":     f"LAT {r[4]:.4f} N, LON {r[3]:.4f} W",
            "lat":             r[4],
            "lng":             r[3],
            "status":          _spill_status(r[5]),
            "detectionTime":   str(r[2]),
            "sensorSource":    "Sentinel-1 SAR (MiT-B2 UNet)",
            "confidence":      f"{round((r[5] or 0) * 100)}%",
            "evidenceLevel":   round((r[5] or 0) * 5),
            "vesselMmsi":      r[6],
            "vesselName":      r[8],
            "vesselType":      r[9],
            "associationStatus": r[7],
        }
        for r in rows
    ]


@router.get("/spills/{spill_id}")
def get_spill_detail(spill_id: str):
    db_id = spill_id.replace("Spill-", "").replace("SPILL-", "")
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, scene_id, acquisition_time, ST_X(aoi_location::geometry) AS lon,
               ST_Y(aoi_location::geometry) AS lat, spill_probability
        FROM sar_detections WHERE id = %s
    """, (db_id,))
    row = cur.fetchone()
    if row is None:
        conn.close()
        raise HTTPException(status_code=404, detail=f"No spill found with id {spill_id}")

    spill_lat, spill_lon, spill_prob = row[4], row[3], row[5]

    # Nearby candidate vessels, reusing your existing source-attribution scoring
    ais_candidates = [
        {"vessel_id": a["vessel_id"], "lat": a["lat"], "lon": a["lon"], "anomaly_score": a["anomaly_score"]}
        for a in _get_ais_anomalies(cur)
    ]
    conn.close()

    ranked = score_candidates({"lat": spill_lat, "lon": spill_lon}, ais_candidates) if ais_candidates else []
    nearby_tracks = []
    for i, c in enumerate(ranked[:2]):
        clean_cand_mmsi = str(c["vessel_id"]).replace("MMSI_", "")
        nearby_tracks.append({
            "mmsi": clean_cand_mmsi,
            "imo": None,
            "type": "Commercial Vessel",
            "matchScore": c["probability_pct"],
            "lastPosTime": "N/A",
            "isPrimary": i == 0,
        })

    return {
        "id": f"Spill-{row[0]}",
        "coordinates": f"LAT {spill_lat:.4f} N, LON {spill_lon:.4f} W",
        "lat": spill_lat,
        "lng": spill_lon,
        "status": _spill_status(spill_prob),
        "detectionTime": str(row[2]),
        "estArea": None,
        "perimeterKm": None,
        "aspectRatio": None,
        "dampingContrast": None,
        "estimatedAge": None,
        "weatheringStage": None,
        "sensorSource": "Sentinel-1 SAR C-Band",
        "confidence": f"{round((spill_prob or 0) * 100)}% {'High' if (spill_prob or 0) > 0.7 else 'Medium'}",
        "evidenceLevel": max(1, round((spill_prob or 0) * 5)),
        "evidenceDescription": f"Sentinel-1 SAR detection record (Confidence: {round((spill_prob or 0) * 100)}%).",
        "areaSqNm": None,
        "nearbyTracks": nearby_tracks,
        "estimatedOriginStatus": "Modeling complete. Run backtrack to visualize trajectory." if nearby_tracks else "Awaiting candidate vessel data.",
        "polygonGeom": None,
    }


# ---------- 3. Monitored Vessels ----------

@router.get("/vessels")
def list_vessels(category: Optional[str] = None, risk: Optional[str] = None, search: Optional[str] = None):
    conn = get_connection()
    cur = conn.cursor()
    
    clean_search = search.strip() if search else None
    if clean_search:
        search_pattern = f"%{clean_search}%"
        cur.execute("""
            SELECT DISTINCT ON (mmsi) mmsi, ship_name, ship_type, sog, heading, status, imo,
                   ST_X(location::geometry) AS lon, ST_Y(location::geometry) AS lat
            FROM ais_positions
            WHERE LOWER(COALESCE(ship_name, '')) LIKE LOWER(%s)
               OR CAST(mmsi AS TEXT) LIKE %s
               OR CAST(COALESCE(imo, '') AS TEXT) LIKE %s
            ORDER BY mmsi, ts DESC
        """, (search_pattern, search_pattern, search_pattern))
    else:
        cur.execute("""
            SELECT DISTINCT ON (mmsi) mmsi, ship_name, ship_type, sog, heading, status, imo,
                   ST_X(location::geometry) AS lon, ST_Y(location::geometry) AS lat
            FROM ais_positions
            ORDER BY mmsi, ts DESC
        """)
    rows = cur.fetchall()

    # Join anomaly scores where available, for a risk estimate
    cur.execute("SELECT vessel_id, anomaly_score FROM ais_anomalies")
    scores = dict(cur.fetchall())
    conn.close()

    vessels = []
    for r in rows:
        mmsi, name, ship_type, sog, heading, status, imo, lon, lat = r
        score = scores.get(mmsi)
        vessel = {
            "id": mmsi,
            "name": name or "Unknown",
            "category": (ship_type or "PRODUCT TANKER").upper(),
            "mmsi": mmsi,
            "imo": imo or "N/A",
            "speed": f"{sog:.1f} kts" if sog is not None else "N/A",
            "heading": f"{heading:.0f}°" if heading is not None and heading != 511 else "N/A",
            "status": NAV_STATUS.get(status, "Underway"),
            "evidenceStrength": round((score or 0) * 100),
            "risk": _risk_from_score(score),
            "lastCoords": f"{lat:.4f}, {lon:.4f}" if lat is not None and lon is not None else "N/A",
            "lat": lat,
            "lng": lon,
        }
        if clean_search:
            s_low = clean_search.lower()
            if s_low not in (name or "").lower() and clean_search not in str(mmsi) and clean_search not in str(imo or ""):
                continue
        if risk and vessel["risk"].lower() != risk.lower():
            continue
        if category and category.lower() not in vessel["category"].lower():
            continue
        vessels.append(vessel)

    return vessels



@router.get("/vessels/{vessel_id}")
def get_vessel_detail_v1(vessel_id: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT ship_name, ship_type, sog, heading, draught, status, imo,
               ST_X(location::geometry), ST_Y(location::geometry)
        FROM ais_positions WHERE mmsi = %s ORDER BY ts DESC LIMIT 1
    """, (vessel_id,))
    row = cur.fetchone()
    conn.close()

    if row is None:
        raise HTTPException(status_code=404, detail=f"No vessel found with mmsi {vessel_id}")

    return {
        "id": vessel_id,
        "name": row[0] or "Unknown",
        "type": row[1] or "Unknown",
        "mmsi": vessel_id,
        "imo": row[6] or "N/A",
        "speed": f"{row[2]:.1f} kts" if row[2] is not None else "N/A",
        "heading": f"{row[3]:.0f}°" if row[3] is not None and row[3] != 511 else "N/A",
        "draft": f"{row[4]:.1f} m" if row[4] is not None else "N/A",
        "status": NAV_STATUS.get(row[5], "Underway"),
        "lastCoords": f"{row[8]:.4f}, {row[7]:.4f}",
    }


class OilSpillAnalyzeRequest(BaseModel):
    mmsi: str
    lat: Optional[float] = None
    lon: Optional[float] = None
    timestamp: Optional[str] = None


@router.post("/oil-spill/analyze")
def analyze_oil_spill(payload: OilSpillAnalyzeRequest):
    """
    Full oil-spill detection pipeline with pipeline state tracking.

    Stages
    ------
    1. AIS gatekeeper — reject normal vessels immediately
    2. Extract exact anomaly position (max speed-drop point) from AIS track
    3. Fetch Sentinel-1 SAR from Copernicus (24h search window)
    4. Preprocess via SARPreprocessor (shared with training)
    5. VanillaUNet segmentation
    6. Look-alike filtering & slick extraction
    7. Spatial + temporal fusion with vessel position
    8. Return structured incident result with pipeline state

    Temporal enforcement
    --------------------
    If |sar_ts - vessel_ts| > SAR_ASSOCIATION_WINDOW_MINUTES → TEMPORAL_MISMATCH.
    Timestamps are NEVER mutated — NO_SATELLITE_ACQUISITION returned if data unavailable.

    Slick geolocation
    -----------------
    slick_centroid is computed from pixel centroid + raster geotransform.
    vessel_location and slick_centroid are returned as SEPARATE fields.
    """
    t_start = time.time()
    clean_mmsi = payload.mmsi.replace("MMSI_", "").strip()
    logger.info("[Analyze] Starting pipeline for MMSI=%s", clean_mmsi)

    # ── Stage 1: AIS Gatekeeper ──────────────────────────────────────────────
    gatekeeper = gatekeep_anomaly(clean_mmsi)
    if not gatekeeper.get("anomalous", False):
        return {
            "incident_id":      f"INC_{clean_mmsi}_NORMAL",
            "pipeline_state":   PipelineState.AIS_NORMAL_VESSEL,
            "verdict":          "NEGATIVE",
            "confidence":       0.95,
            "reason":           "Normal navigation confirmed. Copernicus call skipped.",
            "satellite_triggered": False,
            "ais": {
                "mmsi":            clean_mmsi,
                "anomalous":       False,
                "anomaly_score":   gatekeeper.get("anomaly_score", 0.0),
                "anomaly_type":    gatekeeper.get("anomaly_type"),
                "reasons":         gatekeeper.get("reasons", []),
                "metrics":         gatekeeper.get("metrics", {}),
                "location":        gatekeeper.get("latest_location"),
            },
            "sar":              None,
            "fusion":           None,
            "backtracking":     {"status": "SKIPPED"},
            "final_status":     "NEGATIVE_NORMAL_VESSEL",
        }

    # ── Stage 2: Extract exact anomaly position ──────────────────────────────
    target_lat = payload.lat if payload.lat is not None else gatekeeper.get("latest_location", {}).get("lat", 0.0)
    target_lon = payload.lon if payload.lon is not None else gatekeeper.get("latest_location", {}).get("lon", 0.0)
    target_ts_raw = payload.timestamp if payload.timestamp is not None else gatekeeper.get("latest_timestamp")
    anomaly_description = "LATEST_AIS_POSITION"

    try:
        conn_tmp = get_connection()
        cur_tmp  = conn_tmp.cursor()
        cur_tmp.execute("""
            WITH ranked AS (
                SELECT ts,
                       ST_X(location::geometry) AS lon,
                       ST_Y(location::geometry) AS lat,
                       sog,
                       LAG(sog) OVER (ORDER BY ts) AS prev_sog
                FROM ais_positions
                WHERE mmsi = %s
                ORDER BY ts ASC
            )
            SELECT ts, lon, lat, sog, prev_sog,
                   COALESCE(prev_sog, 0) - sog AS speed_drop
            FROM ranked
            WHERE prev_sog IS NOT NULL
            ORDER BY speed_drop DESC
            LIMIT 1
        """, (clean_mmsi,))
        anomaly_row = cur_tmp.fetchone()
        conn_tmp.close()

        if anomaly_row and float(anomaly_row[5]) >= ais_cfg.SPEED_DROP_THRESHOLD_KNOTS:
            ts_anom, lon_anom, lat_anom, sog_after, sog_before, drop = anomaly_row
            if payload.lat is None:
                target_lat = float(lat_anom)
            if payload.lon is None:
                target_lon = float(lon_anom)
            if payload.timestamp is None:
                target_ts_raw = ts_anom.isoformat() if hasattr(ts_anom, "isoformat") else str(ts_anom)
            anomaly_description = (
                f"MAX_SPEED_DROP_POINT: {float(sog_before):.1f}->{float(sog_after):.1f}kt "
                f"(drop={float(drop):.1f}kt) @ lat={float(lat_anom):.5f} lon={float(lon_anom):.5f}"
            )
    except Exception as exc:
        logger.warning("[Analyze] Anomaly point query warning: %s", exc)

    vessel_ts = parse_timestamp(target_ts_raw)
    logger.info("[Analyze] Anomaly @ lat=%.4f lon=%.4f ts=%s (%s)",
                target_lat, target_lon, target_ts_raw, anomaly_description)

    # ── Stage 3: Copernicus SAR fetch ────────────────────────────────────────
    pipeline_state = PipelineState.SAR_SEARCHING
    sat_data = None
    sat_error = None

    try:
        t_sat = time.time()
        sat_data = fetch_copernicus_satellite_image(
            lat=target_lat,
            lon=target_lon,
            timestamp=target_ts_raw,
            satellite="sentinel-1",
            buffer_km=sar_cfg.SCENE_BUFFER_KM,
            search_window_hours=sar_cfg.SEARCH_WINDOW_HOURS,
            width=sar_cfg.IMAGE_WIDTH,
            height=sar_cfg.IMAGE_HEIGHT,
        )
        logger.info("[Analyze] SAR fetch complete in %.1fs", time.time() - t_sat)
        pipeline_state = PipelineState.SAR_ACQUIRED
    except RuntimeError as e:
        sat_error = str(e)
        if sat_error.startswith("NO_SATELLITE_ACQUISITION"):
            pipeline_state = PipelineState.NO_SATELLITE_ACQUISITION
            logger.info("[Analyze] %s", sat_error)
        else:
            pipeline_state = PipelineState.ERROR
            logger.error("[Analyze] SAR fetch error: %s", sat_error)

    # ── Stage 4-6: Preprocessing + Segmentation + Slick extraction ───────────
    detector = SARUNetDetector(
        association_window_minutes=sar_cfg.ASSOCIATION_WINDOW_MINUTES,
        max_spatial_distance_km=fusion_cfg.MAX_DISTANCE_KM,
        segmentation_threshold=sar_cfg.SEGMENTATION_THRESHOLD,
    )
    slicks = []
    sar_image_stats: dict = {}
    sat_acq_ts = None
    is_display_approx = False
    bbox = None

    if sat_data and sat_data.get("imageBase64"):
        import base64
        import io
        from PIL import Image

        pipeline_state = PipelineState.SAR_PROCESSING
        raw_b64    = sat_data["imageBase64"].split(",")[-1]
        img_bytes  = base64.b64decode(raw_b64)
        pil_img    = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_arr    = np.array(pil_img).astype(np.float32)
        bbox       = sat_data.get("bbox")

        t_preproc  = time.time()
        tensor, is_display_approx = detector.preprocess_from_rgb_jpeg(img_arr)
        if is_display_approx:
            pipeline_state = PipelineState.SAR_DISPLAY_APPROX

        t_infer = time.time()
        prob_map, mask = detector.segment(tensor)

        t_slick = time.time()
        # Pass bbox so slick centroids get geographic coords
        pix_res = sat_data.get("resolutionMetersPerPx", 31.25)
        slicks  = detector.extract_slicks(prob_map, mask, pixel_res_meters=float(pix_res), bbox=bbox)

        logger.info(
            "[Analyze] Preproc=%.2fs  Infer=%.2fs  Slick=%.2fs  found=%d",
            time.time() - t_preproc, t_infer - t_preproc, t_slick - t_infer, len(slicks),
        )

        # Parse SAR acquisition timestamp (no year mutation)
        sat_ts_raw = sat_data.get("targetTimestamp") or (sat_data.get("timeRange") or {}).get("from")
        sat_acq_ts = parse_timestamp(sat_ts_raw)

        sar_image_stats = {
            "image_shape":          list(img_arr.shape),
            "vv_channel_mean":      round(float(np.mean(img_arr[:, :, 0])), 2),
            "vh_channel_mean":      round(float(np.mean(img_arr[:, :, 1])), 2),
            "ratio_channel_mean":   round(float(np.mean(img_arr[:, :, 2])), 2),
            "prob_map_max":         round(float(prob_map.max()), 3),
            "prob_map_mean":        round(float(prob_map.mean()), 3),
            "mask_positive_pixels": int(np.sum(mask)),
            "data_quality":         "DISPLAY_APPROX (JPEG preview, not calibrated raster)",
            "preprocessing_note":   "SARPreprocessor v1 applied — no ImageNet normalization",
        }

        if slicks:
            pipeline_state = PipelineState.SLICK_DETECTED
        else:
            pipeline_state = PipelineState.NO_SLICK_DETECTED

    # ── Stage 7: Fusion & verdict ─────────────────────────────────────────────
    association_result = detector.associate_and_decide(
        slicks=slicks,
        vessel_mmsi=clean_mmsi,
        vessel_lat=target_lat,
        vessel_lon=target_lon,
        vessel_ts=vessel_ts,
        sar_acq_ts=sat_acq_ts,
        sat_available=(sat_data is not None and not sat_error.startswith("NO_SATELLITE") if sat_error else sat_data is not None),
        is_display_approx=is_display_approx,
    )
    association_result["sar_image_stats"] = sar_image_stats
    pipeline_state = PipelineState(association_result.get("pipeline_state", PipelineState.INCONCLUSIVE))

    # Cache characterization for backtracking
    _char = association_result.get("characterization")
    if _char:
        update_sar_characterization_cache(_char)

    # ── DB persist for POSITIVE detections ───────────────────────────────────
    if association_result["verdict"] == "POSITIVE":
        slick_lat = (association_result.get("slick_centroid") or {}).get("lat", target_lat)
        slick_lon = (association_result.get("slick_centroid") or {}).get("lon", target_lon)
        try:
            conn = get_connection()
            cur  = conn.cursor()
            cur.execute("""
                INSERT INTO sar_detections (
                    scene_id, acquisition_time, aoi_location, spill_probability,
                    segmentation_confidence, time_difference_minutes, association_status, candidate_mmsi
                ) VALUES (
                    %s, NOW(), ST_SetSRID(ST_MakePoint(%s, %s), 4326), %s, %s, %s, %s, %s
                )
            """, (
                f"S1_{clean_mmsi}_{int(time.time())}",
                slick_lon, slick_lat,
                association_result["confidence"],
                association_result["confidence"],
                association_result.get("time_difference_minutes", 0),
                association_result.get("association", "CONFIRMED"),
                clean_mmsi,
            ))
            conn.commit()
            conn.close()
            logger.info("[Analyze] POSITIVE detection persisted to sar_detections")
        except Exception as db_err:
            logger.warning("[Analyze] DB persist warning: %s", db_err)

    elapsed = round(time.time() - t_start, 2)
    logger.info("[Analyze] Complete in %.2fs — verdict=%s state=%s",
                elapsed, association_result["verdict"], pipeline_state)

    return {
        "incident_id":     f"INC_{clean_mmsi}_{int(time.time())}",
        "pipeline_state":  pipeline_state,
        "processing_time_s": elapsed,

        # AIS component
        "ais": {
            "mmsi":          clean_mmsi,
            "anomalous":     gatekeeper.get("anomalous"),
            "anomaly_score": gatekeeper.get("anomaly_score"),
            "anomaly_type":  gatekeeper.get("anomaly_type"),
            "reasons":       gatekeeper.get("reasons", []),
            "metrics":       gatekeeper.get("metrics", {}),
            "timestamp":     target_ts_raw,
            "location":      {"lat": target_lat, "lon": target_lon},
            "anomaly_point_description": anomaly_description,
        },

        # SAR component
        "sar": {
            "available":          sat_data is not None,
            "satellite":          sat_data.get("satellite") if sat_data else None,
            "data_type":          sat_data.get("data_type", "DISPLAY_JPEG") if sat_data else None,
            "acquisition_time":   sat_data.get("targetTimestamp") if sat_data else None,
            "time_range":         sat_data.get("timeRange") if sat_data else None,
            "bbox":               bbox,
            "center":             [target_lat, target_lon],
            "resolution_m_px":    sat_data.get("resolutionMetersPerPx") if sat_data else None,
            "imageBase64":        sat_data.get("imageBase64") if sat_data else None,
            "image_stats":        sar_image_stats,
            "slick_detected":     len(slicks) > 0,
            "slicks_count":       len(slicks),
            # DISTINCT from vessel_location:
            "slick_centroid":     association_result.get("slick_centroid"),
            "slick_polygon_geojson": association_result.get("slick_polygon_geojson"),
            "segmentation_confidence": association_result.get("confidence"),
            "attribution":        "Copernicus Sentinel Data — ESA",
            "error":              sat_error,
        },

        # Fusion component
        "fusion": {
            "distance_km":           association_result.get("distance_km"),
            "time_difference_minutes": association_result.get("time_difference_minutes"),
            "spatial_match":         association_result.get("spatial_match"),
            "temporal_match":        association_result.get("temporal_match"),
            "confidence":            association_result.get("fusion_confidence"),
            "status":                association_result.get("association"),
            "data_quality":          association_result.get("data_quality"),
        },

        # Vessel position used (DISTINCT from slick centroid)
        "vessel_location": {"lat": target_lat, "lon": target_lon},

        # Top-level verdict
        "verdict":      association_result["verdict"],
        "confidence":   association_result["confidence"],
        "reason":       association_result["reason"],
        "final_status": (
            "PROBABLE_OIL_SPILL" if association_result["verdict"] == "POSITIVE"
            else "POSSIBLE_SLICK" if association_result["verdict"] == "INCONCLUSIVE"
            else "CLEAN_SEA_SURFACE"
        ),

        # Characterization
        "characterization": association_result.get("characterization"),

        # Backtracking hint
        "backtracking": {
            "status": "SIMULATED_DRIFT",
            "note":   "Run /backtracking/analysis?spill_id=... for Lagrangian drift simulation.",
        },

        # Legacy alias fields (for backward compat with existing frontend)
        "ais_gatekeeper":    gatekeeper,
        "sar_detection":     association_result,
        "satellite_triggered": sat_data is not None,
        "satellite_recon": {
            "satellite":   sat_data.get("satellite") if sat_data else None,
            "timeRange":   sat_data.get("timeRange") if sat_data else None,
            "imageBase64": sat_data.get("imageBase64") if sat_data else None,
            "bbox":        bbox,
            "center":      [target_lat, target_lon],
        },
    }


@router.get("/vessels/{vessel_id}/track")
def get_vessel_track_v1(vessel_id: str, hours: int = 24):
    clean_id = vessel_id.replace("MMSI_", "")
    return get_vessel_trajectory(clean_id, hours=hours)


class BacktrackSimulationRequest(BaseModel):
    simulationHours: Optional[int] = 12
    windDriftFactor: Optional[float] = 0.03
    currentSpeedKnots: Optional[float] = 0.8
    currentDirDeg: Optional[float] = 45.0


@router.post("/spills/{spill_id}/backtrack")
def run_backtrack_simulation(spill_id: str, payload: Optional[BacktrackSimulationRequest] = None):
    hours = payload.simulationHours if payload and payload.simulationHours else 12
    return get_backtracking_analysis(spill_id=spill_id, hours=hours)


# ---------- 4. Backtracking & Forensic Analysis ----------

@router.get("/backtracking/analysis")
def get_backtracking_analysis_query(spill_id: Optional[str] = None, incidentId: Optional[str] = None, hours: int = 12):
    return get_backtracking_analysis(spill_id=spill_id, incident_id=incidentId, hours=hours)


@router.get("/backtracking/{spill_id}")
def get_backtracking_analysis_path(spill_id: str, hours: int = 12):
    return get_backtracking_analysis(spill_id=spill_id, hours=hours)


def get_backtracking_analysis(spill_id: Optional[str] = None, incident_id: Optional[str] = None, hours: int = 12):
    target_key = (incident_id or spill_id or "").strip()
    if target_key.lower() == "analysis":
        target_key = ""
    spill_key = target_key
    db_id = spill_key.replace("Spill-", "").replace("SPILL-", "")
    conn = get_connection()
    cur = conn.cursor()

    spill_lat = None
    spill_lon = None
    acquisition_time = None

    # 1. Lookup from real persisted incidents (DB first, then in-memory cache)
    if spill_key:
        try:
            cur.execute("""
                SELECT latitude, longitude, timestamp, mmsi
                FROM incidents
                WHERE id = %s OR mmsi = %s
                ORDER BY created_at DESC LIMIT 1
            """, (spill_key, spill_key))
            inc_row = cur.fetchone()
            if inc_row and inc_row[0] is not None and inc_row[1] is not None:
                spill_lat, spill_lon = float(inc_row[0]), float(inc_row[1])
                acquisition_time = inc_row[2].isoformat() if hasattr(inc_row[2], "isoformat") else str(inc_row[2])
        except Exception as e:
            conn.rollback()
            logger.warning("[Backtracking] DB incident lookup warning: %s", e)

        if (spill_lat is None or spill_lon is None) and spill_key in _incidents_store:
            inc = _incidents_store[spill_key]
            loc = inc.get("location") or {}
            spill_lat = loc.get("lat")
            spill_lon = loc.get("lon")
            acquisition_time = inc.get("timestamp")

    # 2. Lookup from sar_detections by numeric ID
    if (spill_lat is None or spill_lon is None) and db_id.isdigit():
        cur.execute("""
            SELECT ST_X(aoi_location::geometry), ST_Y(aoi_location::geometry), acquisition_time
            FROM sar_detections WHERE id = %s
        """, (int(db_id),))
        row = cur.fetchone()
        if row:
            spill_lon, spill_lat, acquisition_time = row

    # 3. Lookup from ais_positions if spill_key is an MMSI
    if (spill_lat is None or spill_lon is None) and spill_key:
        clean_mmsi = spill_key.replace("MMSI_", "")
        cur.execute("""
            SELECT ST_X(location::geometry), ST_Y(location::geometry), ts
            FROM ais_positions WHERE mmsi = %s ORDER BY ts DESC LIMIT 1
        """, (clean_mmsi,))
        row = cur.fetchone()
        if row:
            spill_lon, spill_lat, acquisition_time = row

    # If a specific target (incident or spill) was requested but NOT found, return INCONCLUSIVE (do not substitute random other spills)
    if spill_key and (spill_lat is None or spill_lon is None):
        conn.close()
        return {
            "spillId": spill_key,
            "incidentId": incident_id or (spill_key if spill_key.startswith("INC-") else None),
            "status": "INCONCLUSIVE",
            "detail": f"Spill or incident '{spill_key}' not found in active records.",
            "rankings": [],
            "forensicEvidence": {
                "proximityLevel": "LOW",
                "timeCorrelationLevel": "UNCONFIRMED",
                "trajectoryMatchLevel": "LOW",
                "cpa": "N/A",
                "intersectionArea": "N/A",
                "anomalyTimestamp": "N/A",
                "deltaT": "N/A",
                "headingVariance": "N/A",
                "speedProfile": "N/A",
            },
            "estimatedOriginPoint": None,
            "estimatedDischargeTime": None,
            "driftTrajectory": [],
            "forecastTrajectory": [],
            "spillCharacterization": {"status": "DEFERRED", "data_available": False},
            "coastalImpact": {},
            "hydrodynamics": {"status": "INCONCLUSIVE"},
        }

    # Only if NO specific spill_id was provided at all, fallback to latest SAR detection if any
    if not spill_key and (spill_lat is None or spill_lon is None):
        cur.execute("""
            SELECT ST_X(aoi_location::geometry), ST_Y(aoi_location::geometry), acquisition_time, id
            FROM sar_detections ORDER BY id DESC LIMIT 1
        """)
        row = cur.fetchone()
        if row:
            spill_lon, spill_lat, acquisition_time = row[0], row[1], row[2]
            spill_key = f"Spill-{row[3]}"

    # If still missing, return clean INCONCLUSIVE status (NEVER fabricate Florida coordinates)
    if spill_lat is None or spill_lon is None:
        conn.close()
        return {
            "spillId": spill_key or "UNKNOWN",
            "status": "INCONCLUSIVE",
            "detail": "No observed origin coordinates available.",
            "rankings": [],
            "forensicEvidence": {
                "proximityLevel": "LOW",
                "timeCorrelationLevel": "UNCONFIRMED",
                "trajectoryMatchLevel": "LOW",
                "cpa": "N/A",
                "intersectionArea": "N/A",
                "anomalyTimestamp": "N/A",
                "deltaT": "N/A",
                "headingVariance": "N/A",
                "speedProfile": "N/A",
            },
            "estimatedOriginPoint": None,
            "estimatedDischargeTime": None,
            "driftTrajectory": [],
            "forecastTrajectory": [],
            "spillCharacterization": {"status": "DEFERRED", "data_available": False},
            "coastalImpact": {},
            "hydrodynamics": {"status": "INCONCLUSIVE"},
        }

    sim = simulate_lagrangian_backtrack(
        spill_lat=spill_lat,
        spill_lon=spill_lon,
        spill_ts=acquisition_time,
        simulation_hours=hours,
        cur=cur
    )
    conn.close()

    variants = ["cyan", "amber", "muted"]
    rankings = []
    for i, c in enumerate(sim["candidate_rankings"]):
        clean_cand_imo = c.get("imo") if c.get("imo") and str(c.get("imo")) != "UNKNOWN" and not str(c.get("imo")).startswith("IMO-") else None
        raw_assoc = c.get("attribution_score") if c.get("attribution_score") is not None else c.get("probability_pct", 0.0)
        assoc_score = round(float(raw_assoc), 1)
        rankings.append({
            "rank": i + 1,
            "name": c["name"],
            "mmsi": c["mmsi"],
            "imo": clean_cand_imo,
            "associationScore": assoc_score,
            "attributionScore": assoc_score,
            "evidenceLevel": max(1, round(assoc_score / 20)),
            "variant": variants[i] if i < len(variants) else "muted",
            "anomalyScore": c.get("anomaly_score", 0.0),
            "anomalyType": c.get("anomaly_type", "NORMAL"),
            "minDistanceKm": c.get("min_distance_to_corridor_km", 0.0),
            "lat": c.get("lat"),
            "lng": c.get("lon") if c.get("lon") is not None else c.get("lng"),
            "lon": c.get("lon") if c.get("lon") is not None else c.get("lng"),
            "positionTimestamp": c.get("positionTimestamp"),
        })

    top_cand = rankings[0] if rankings else None
    forensic_evidence = {
        "cpa": f"{top_cand['minDistanceKm']:.2f} km" if top_cand else "N/A",
        "intersectionArea": f"{sim['estimated_origin']['uncertainty_radius_km']:.1f} km²",
        "proximityLevel": "HIGH" if top_cand and top_cand['minDistanceKm'] < 5 else ("MEDIUM" if top_cand and top_cand['minDistanceKm'] < 20 else "LOW"),
        "anomalyTimestamp": str(sim["estimated_origin"]["estimated_release_time"]),
        "deltaT": f"{sim['simulation_hours']} hours",
        "timeCorrelationLevel": "STRONG" if top_cand and top_cand['minDistanceKm'] < 10 else "UNCONFIRMED",
        "headingVariance": "HIGH" if top_cand and top_cand.get("anomalyType") == "ERRATIC_COURSE" else "MODERATE",
        "speedProfile": "SUDDEN_DROP" if top_cand and top_cand.get("anomalyType") == "SPEED_CHANGE" else "ABNORMAL",
        "trajectoryMatchLevel": "VERY_HIGH" if top_cand and top_cand["associationScore"] > 60 else ("HIGH" if top_cand and top_cand["associationScore"] > 30 else "MEDIUM"),
    }

    return {
        "spillId": target_key or spill_id,
        "incidentId": incident_id or (target_key if target_key.startswith("INC-") else None),
        "rankings": rankings,
        "forensicEvidence": forensic_evidence,
        "estimatedOriginPoint": [sim["estimated_origin"]["lat"], sim["estimated_origin"]["lon"]],
        "estimatedDischargeTime": sim["estimated_origin"]["estimated_release_time"],
        "driftTrajectory": sim["drift_trajectory"],
        "forecastTrajectory": sim.get("forecast_trajectory", []),
        "spillCharacterization": sim.get("spill_characterization", {}),
        "coastalImpact": sim.get("coastal_impact", {}),
        "hydrodynamics": sim["hydrodynamics"],
    }


# ---------- 5. Alerts ----------

def _is_alert_acknowledged(alert_id: str, cur) -> bool:
    """Check if an alert has been acknowledged in the database."""
    try:
        cur.execute(
            "SELECT 1 FROM alert_acknowledgements WHERE alert_id = %s LIMIT 1",
            (str(alert_id),),
        )
        return cur.fetchone() is not None
    except Exception:
        return False


@router.get("/alerts")
def get_alerts_v1():
    conn = get_connection()
    cur  = conn.cursor()
    ais_list = _get_ais_anomalies(cur)
    alerts   = fuse_db(ais_list, cur) if ais_list else []
    conn.close()

    severity_map = {
        "ALERT":              "critical",
        "WATCH":              "warning",
        "WATCH_NO_SAT_COVERAGE": "info",
    }

    # Re-open for acknowledgement check (connection already closed above)
    conn2 = get_connection()
    cur2  = conn2.cursor()

    # Pre-fetch real vessel attributes (imo, sog, ship_name) from ais_positions
    vessel_mmsis = [str(a["vessel_id"]).replace("MMSI_", "") for a in alerts]
    vessel_lookup = {}
    if vessel_mmsis:
        try:
            cur2.execute("""
                SELECT DISTINCT ON (mmsi) mmsi, imo, sog, ship_name
                FROM ais_positions
                WHERE mmsi = ANY(%s)
                ORDER BY mmsi, ts DESC
            """, (vessel_mmsis,))
            for r in cur2.fetchall():
                m_key = str(r[0])
                r_imo = r[1]
                clean_imo = str(r_imo) if r_imo and str(r_imo) != "UNKNOWN" and not str(r_imo).startswith("IMO-") else None
                sog_val = f"{r[2]:.1f} kn" if r[2] is not None else None
                vessel_lookup[m_key] = {
                    "imo": clean_imo,
                    "speed": sog_val,
                    "name": r[3] or f"Vessel {m_key}",
                }
        except Exception as e:
            logger.warning("[Alerts] Vessel lookup warning: %s", e)

    result = []
    for a in alerts:
        clean_mmsi = str(a["vessel_id"]).replace("MMSI_", "")
        v_info = vessel_lookup.get(clean_mmsi, {})
        if not _is_alert_acknowledged(a["vessel_id"], cur2):
            result.append({
                "id":             a["vessel_id"],
                "mmsi":           clean_mmsi,
                "title":          f"{a['anomaly_type'].replace('_', ' ').title()} — {v_info.get('name', f'Vessel {clean_mmsi}')}",
                "subtitle":       f"Confidence: {round(a['combined_confidence'] * 100)}%",
                "timestamp":      a["timestamp"],
                "severity":       severity_map.get(a["status"], "info"),
                "coordinates":    f"{a['location']['lat']:.4f}, {a['location']['lon']:.4f}",
                "lat":            a["location"]["lat"],
                "lng":            a["location"]["lon"],
                "actionLabel":    "Identify" if a["status"] == "ALERT" else "View Evidence",
                "matchConfidence": round(a["combined_confidence"] * 100),
                "imo":            v_info.get("imo"),
                "speed":          v_info.get("speed"),
                "data_source":    "AIS_SAR_FUSION",
            })
    conn2.close()
    return result


@router.post("/alerts/{alert_id}/acknowledge")
def acknowledge_alert(alert_id: str):
    """Persist alert acknowledgement to database — survives server restarts."""
    conn = get_connection()
    cur  = conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO alert_acknowledgements (alert_id, acknowledged_at)
            VALUES (%s, NOW())
            ON CONFLICT (alert_id) DO UPDATE SET acknowledged_at = NOW()
            """,
            (str(alert_id),),
        )
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.warning("[Alerts] Could not persist acknowledgement for %s: %s", alert_id, e)
    finally:
        conn.close()
    return {"success": True, "alert_id": alert_id}


# ---------- 6. Manual Incident Reporting ----------

class ReportRequest(BaseModel):
    reporterName: str
    contactEmail: Optional[str] = None
    lat: float
    lng: float
    estimatedSizeSqKm: Optional[float] = None
    spillAppearance: Optional[str] = None
    notes: Optional[str] = None


@router.post("/reports")
def report_spill_incident(payload: ReportRequest):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO spill_reports (reporter_name, contact_email, location, estimated_size_sq_km, spill_appearance, notes)
        VALUES (%s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326), %s, %s, %s)
        RETURNING id
    """, (
        payload.reporterName, payload.contactEmail, payload.lng, payload.lat,
        payload.estimatedSizeSqKm, payload.spillAppearance, payload.notes,
    ))
    new_id = cur.fetchone()[0]
    conn.commit()
    conn.close()
    return {"reportId": new_id, "status": "received"}


# ---------- 8. Copernicus Satellite Reconnaissance ----------

@router.get("/satellite/recon")
def get_satellite_recon(
    lat: float,
    lon: float,
    timestamp: Optional[str] = None,
    satellite: str = "sentinel-1",
    buffer_km: float = 8.0,
):
    """
    Fetches real Sentinel-1 SAR (Radar) or Sentinel-2 Optical satellite imagery
    from Copernicus Data Space Ecosystem for the given coordinates and timestamp.
    """
    from sar_pipeline.copernicus_service import fetch_copernicus_satellite_image

    try:
        data = fetch_copernicus_satellite_image(
            lat=lat,
            lon=lon,
            timestamp=timestamp,
            satellite=satellite,
            buffer_km=buffer_km,
        )
        return data
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/satellite/vessel/{vessel_id}")
def get_vessel_satellite_recon(vessel_id: str, satellite: str = "sentinel-1"):
    """
    Finds the anomalous vessel's location and detection timestamp and fetches
    the corresponding Copernicus satellite image.
    """
    from sar_pipeline.copernicus_service import fetch_copernicus_satellite_image

    conn = get_connection()
    cur = conn.cursor()

    clean_id = vessel_id.replace("MMSI_", "")

    # Check if vessel has a recorded anomaly
    cur.execute("""
        SELECT vessel_id, ST_X(location::geometry) AS lon, ST_Y(location::geometry) AS lat,
               ts, anomaly_type, anomaly_score
        FROM ais_anomalies
        WHERE vessel_id = %s OR vessel_id = %s
        ORDER BY ts DESC
        LIMIT 1
    """, (clean_id, f"MMSI_{clean_id}"))
    anomaly_row = cur.fetchone()

    if anomaly_row:
        v_id, lon, lat, ts, a_type, a_score = anomaly_row
        vessel_name = f"Vessel {v_id}"
        cur.execute("SELECT ship_name FROM ais_positions WHERE mmsi = %s AND ship_name IS NOT NULL LIMIT 1", (clean_id,))
        static_row = cur.fetchone()
        if static_row and static_row[0]:
            vessel_name = static_row[0]

        conn.close()

        sat_result = fetch_copernicus_satellite_image(
            lat=lat,
            lon=lon,
            timestamp=ts,
            satellite=satellite,
            buffer_km=8.0,
        )

        return {
            "vesselId": v_id,
            "vesselName": vessel_name,
            "isAnomaly": True,
            "anomalyType": a_type,
            "anomalyScore": a_score,
            "coordinates": {"lat": lat, "lon": lon},
            "detectionTimestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
            **sat_result,
        }

    # Fallback: check latest position in ais_positions
    cur.execute("""
        SELECT mmsi, ST_X(location::geometry) AS lon, ST_Y(location::geometry) AS lat, ts, ship_name
        FROM ais_positions
        WHERE mmsi = %s
        ORDER BY ts DESC
        LIMIT 1
    """, (clean_id,))
    pos_row = cur.fetchone()
    conn.close()

    if not pos_row:
        raise HTTPException(status_code=404, detail=f"Vessel {vessel_id} not found")

    mmsi, lon, lat, ts, ship_name = pos_row
    sat_result = fetch_copernicus_satellite_image(
        lat=lat,
        lon=lon,
        timestamp=ts,
        satellite=satellite,
        buffer_km=8.0,
    )

    return {
        "vesselId": mmsi,
        "vesselName": ship_name or f"Vessel {mmsi}",
        "isAnomaly": False,
        "coordinates": {"lat": lat, "lon": lon},
        "detectionTimestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
        **sat_result,
    }


@router.get("/satellite/anomalies")
def get_anomalies_for_satellite():
    """
    Returns anomalous vessels with coordinates and timestamps, ready for Copernicus satellite recon.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT a.vessel_id, ST_X(a.location::geometry) AS lon, ST_Y(a.location::geometry) AS lat,
               a.ts, a.anomaly_type, a.anomaly_score,
               (SELECT ship_name FROM ais_positions WHERE mmsi = a.vessel_id OR mmsi = REPLACE(a.vessel_id, 'MMSI_', '') LIMIT 1) AS ship_name
        FROM ais_anomalies a
        ORDER BY a.ts DESC
        LIMIT 20
    """)
    rows = cur.fetchall()
    conn.close()

    return [
        {
            "vesselId": r[0],
            "vesselName": r[6] or f"Vessel {r[0]}",
            "lon": r[1],
            "lat": r[2],
            "timestamp": r[3].isoformat() if hasattr(r[3], "isoformat") else str(r[3]),
            "anomalyType": r[4],
            "anomalyScore": r[5],
        }
        for r in rows
    ]


@app.post("/api/sar/analyze")
@router.post("/sar/analyze")
async def analyze_sar_image_endpoint(
    image: UploadFile = File(...),
    threshold: Optional[float] = None,
):
    """
    Standalone SAR oil-spill segmentation analysis using authoritative MiT-B2 + U-Net.
    Accepts image file (PNG, JPG, JPEG, TIF, TIFF).
    Returns segmentation metrics, status, and 4 visual outputs as base64 Data URIs.
    Completely independent of AIS streaming and external APIs.
    """
    filename = image.filename or "uploaded_image.png"
    try:
        content = await image.read()
        kwargs = {}
        if threshold is not None:
            kwargs["threshold"] = threshold

        result = analyze_sar_image(content, filename=filename, **kwargs)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as exc:
        logger.error("SAR analysis failed for '%s': %s", filename, exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="SAR image analysis failed due to an internal processing error.",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Historical Oil Spill Investigation  (Lagrangian particle backtracking)
# ─────────────────────────────────────────────────────────────────────────────
from fusion.lagrangian_model import (
    LagrangianModel,
    MockEnvironmentalProvider,
    SpillObservation,
    parse_spill_timestamp,
)
from fusion.historical_ais_matcher import find_candidate_vessels
import json as _json


@router.post("/historical-investigation")
async def run_historical_investigation(
    latitude: float = Form(...),
    longitude: float = Form(...),
    timestamp: str = Form(...),
    duration_hours: int = Form(default=24, ge=1, le=48),
    windage: float = Form(default=0.03, ge=0.005, le=0.10),
    n_particles: int = Form(default=500, ge=10, le=1000),
    uncertainty_radius_m: float = Form(default=5000.0, ge=500.0, le=100000.0),
    data_mode: str = Form(default="DEMO"),
    file: Optional[UploadFile] = File(default=None),
):
    """
    POST /api/v1/historical-investigation

    Runs Lagrangian particle backtracking from an observed oil-spill location
    and matches historical AIS vessel data to produce attribution candidates.

    Steps:
      1. Validate inputs
      2. If image uploaded: run MIT-B2 U-Net segmentation to get spill polygon
         (IMAGE_DERIVED geometry). Otherwise: use supplied lat/lon (POINT_ONLY).
      3. Run Lagrangian backward simulation (windage ensemble 0.01/0.03/0.04)
      4. Match historical AIS against particle corridor
      5. Persist to DB
      6. Return full response

    data_mode:
      "DEMO" -> MockEnvironmentalProvider (always works, labeled as DEMO)
      "LIVE" -> NOAA ERDDAP HYCOM (real data; never silently falls back to mock)

    Results are CANDIDATE vessels only -- NOT confirmed polluters.
    """
    import os as _os
    from datetime import datetime as _dt, timezone as _tz

    # -- 1. Validate inputs ------------------------------------------------
    if not (-90.0 <= latitude <= 90.0):
        raise HTTPException(status_code=422, detail=f"Invalid latitude: {latitude}")
    if not (-180.0 <= longitude <= 180.0):
        raise HTTPException(status_code=422, detail=f"Invalid longitude: {longitude}")

    data_mode_upper = data_mode.strip().upper()
    if data_mode_upper not in ("DEMO", "LIVE"):
        raise HTTPException(status_code=422, detail="data_mode must be 'DEMO' or 'LIVE'")

    try:
        spill_ts = parse_spill_timestamp(timestamp)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # -- 2. Image segmentation via MIT-B2 U-Net ---------------------------
    image_filename: Optional[str] = None
    image_bytes: Optional[bytes] = None
    segmentation_result = None
    image_analysis_response: Optional[dict] = None

    if file is not None:
        image_filename = file.filename
        try:
            image_bytes = await file.read()
        except Exception as exc:
            logger.warning("Failed to read uploaded image: %s", exc)
            image_bytes = None

    if image_bytes:
        try:
            from sar_pipeline.mit_b2_inference import analyze_oil_spill
            segmentation_result = analyze_oil_spill(image_bytes)

            image_analysis_response = {
                "model":                  segmentation_result.model_name,
                "detected":               segmentation_result.detected,
                "geometrySource":         segmentation_result.geometry_source,
                "spillPixelCount":        segmentation_result.spill_pixel_count,
                "totalPixels":            segmentation_result.total_pixels,
                "spillFraction":          segmentation_result.spill_fraction,
                "spillAreaKm2":           segmentation_result.spill_area_km2,
                "centroidRel":            list(segmentation_result.centroid_rel) if segmentation_result.centroid_rel else None,
                "boundingBoxRel":         list(segmentation_result.bounding_box_rel) if segmentation_result.bounding_box_rel else None,
                "segmentationConfidence": segmentation_result.segmentation_confidence,
                "maskPngBase64":          segmentation_result.mask_png_base64,
                "reason":                 segmentation_result.reason,
                "error":                  segmentation_result.error,
                # spillPolygonGeo is populated in step 3 after geographic conversion;
                # set placeholder here so the key always exists in the response.
                "spillPolygonGeo":        None,
            }

            if segmentation_result.error:
                logger.warning("[Investigation] MIT-B2 inference error: %s", segmentation_result.error)

        except Exception as exc:
            logger.error("[Investigation] MIT-B2 inference failed: %s", exc, exc_info=True)
            image_analysis_response = {
                "model":    "MIT-B2-U-Net",
                "detected": False,
                "error":    str(exc),
            }

    # -- 3. Build SpillObservation ----------------------------------------
    spill_polygon: Optional[list] = None
    geometry_source = "POINT_ONLY"

    if segmentation_result and segmentation_result.detected and segmentation_result.polygon_rel:
        from sar_pipeline.mit_b2_inference import segmentation_polygon_to_geographic
        try:
            geo_polygon = segmentation_polygon_to_geographic(
                segmentation_result.polygon_rel,
                spill_lat=latitude,
                spill_lon=longitude,
            )
            spill_polygon = geo_polygon
            geometry_source = "IMAGE_DERIVED"
            logger.info(
                "[Investigation] MIT-B2 detected spill. Geometry: IMAGE_DERIVED, "
                "polygon vertices: %d, area: %s km2",
                len(geo_polygon),
                segmentation_result.spill_area_km2,
            )
            # Populate spillPolygonGeo in imageAnalysis so frontend can render
            # the geographic polygon on the map.
            # geo_polygon is [(lat, lon), ...] — we serialize as {latitude, longitude}
            # so the Leaflet layer can use [p.latitude, p.longitude] without confusion.
            if image_analysis_response is not None:
                image_analysis_response["spillPolygonGeo"] = [
                    {"latitude": round(float(lat_p), 6), "longitude": round(float(lon_p), 6)}
                    for lat_p, lon_p in geo_polygon
                ]
        except Exception as exc:
            logger.warning("[Investigation] Polygon geo-conversion failed: %s", exc)
            spill_polygon = None

    try:
        obs = SpillObservation(
            latitude=latitude,
            longitude=longitude,
            timestamp=spill_ts,
            image_filename=image_filename,
            polygon=spill_polygon,
            uncertainty_radius_m=uncertainty_radius_m,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # -- 4. Select environmental data provider ----------------------------
    env_error: Optional[str] = None
    try:
        from fusion.environmental_providers import get_provider
        from fusion.lagrangian_model import EnvironmentalDataUnavailable
        provider = get_provider(data_mode_upper)
    except Exception as exc:
        if data_mode_upper == "LIVE":
            raise HTTPException(
                status_code=503,
                detail=(
                    f"LIVE environmental data unavailable: {exc}. "
                    "Switch to data_mode=DEMO or configure environmental data credentials."
                ),
            )
        provider = MockEnvironmentalProvider()
        env_error = str(exc)

    lag_model = LagrangianModel(provider)

    # -- 5. Run Lagrangian backward simulation ----------------------------
    windage_ensemble = (0.01, round(windage, 3), round(min(0.04, windage + 0.01), 3))
    try:
        result = lag_model.run_backward(
            obs=obs,
            n_particles=n_particles,
            duration_hours=float(duration_hours),
            timestep_min=15,
            windage_coefficients=windage_ensemble,
            seed=None,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Lagrangian simulation failed: {str(exc)}",
        )

    # -- 6. Query historical AIS (with fixed time_match_score) ------------
    trajectory_timestamps = []
    for step in result.trajectory_steps:
        try:
            ts_step = _dt.fromisoformat(step.timestamp.replace("Z", "+00:00"))
            if ts_step.tzinfo is None:
                ts_step = ts_step.replace(tzinfo=_tz.utc)
            trajectory_timestamps.append(ts_step)
        except Exception:
            pass

    trajectory_centroids = [
        (step.lat, step.lon) for step in result.trajectory_steps
    ]

    ais_candidates = []
    ais_error: Optional[str] = None
    try:
        ais_candidates = find_candidate_vessels(
            trajectory_centroids=trajectory_centroids,
            spill_lat=latitude,
            spill_lon=longitude,
            spill_timestamp=spill_ts,
            duration_hours=float(duration_hours),
            source_centroid_lat=result.source_region.centroid_lat,
            source_centroid_lon=result.source_region.centroid_lon,
            envelope_polygon=result.envelope_polygon,
            trajectory_timestamps=trajectory_timestamps,
        )
    except Exception as exc:
        ais_error = str(exc)
        logger.warning("[Investigation] AIS matching failed: %s", exc)

    # -- 7. Save investigation to DB -------------------------------------
    try:
        conn = get_connection()
        cur = conn.cursor()

        traj_summary = [
            {
                "step":      s.step,
                "hours_ago": s.hours_ago,
                "ts":        s.timestamp,
                "lat":       s.lat,
                "lon":       s.lon,
            }
            for s in result.trajectory_steps
        ]

        source_polygon_pts = result.source_region.polygon
        source_wkt: Optional[str] = None
        if len(source_polygon_pts) >= 3:
            ring = ", ".join(
                f"{lon_p} {lat_p}" for lat_p, lon_p in source_polygon_pts
            )
            first_lat, first_lon = source_polygon_pts[0]
            ring += f", {first_lon} {first_lat}"
            source_wkt = f"POLYGON(({ring}))"

        candidates_json_list = [
            {
                "rank":                   c.rank,
                "mmsi":                   c.mmsi,
                "vesselName":             c.vessel_name,
                "imo":                    c.imo,
                "minDistanceKm":          c.min_distance_km,
                "trajectoryOverlapScore": c.trajectory_overlap_score,
                "timeMatchScore":         c.time_match_score,
                "aisAnomalyScore":        c.ais_anomaly_score,
                "distanceScore":          c.distance_score,
                "sourceRegionScore":      c.source_region_score,
                "attributionScore":       c.attribution_score,
                "matchingAisPings":       c.matching_ais_pings,
                "lat":                    c.lat,
                "lon":                    c.lon,
                "positionTimestamp":      c.position_timestamp,
            }
            for c in ais_candidates
        ]

        img_analysis_db = None
        if image_analysis_response:
            img_analysis_db = {k: v for k, v in image_analysis_response.items()
                               if k != "maskPngBase64"}

        try:
            cur.execute("""
                INSERT INTO historical_investigations
                    (spill_lat, spill_lon, spill_timestamp, image_filename,
                     duration_hours, particle_count, timestep_minutes, windage,
                     windage_ensemble, data_mode, provider_name, status,
                     trajectory_summary, source_region, candidates_json,
                     segmentation_detected, geometry_source, spill_area_km2,
                     spill_pixel_count, spill_fraction, segmentation_confidence,
                     image_analysis_json, uncertainty_radius_m)
                VALUES
                    (%s, %s, %s, %s,
                     %s, %s, %s, %s,
                     %s, %s, %s, %s,
                     %s, %s::geography, %s,
                     %s, %s, %s,
                     %s, %s, %s,
                     %s, %s)
                RETURNING id
            """, (
                latitude, longitude, spill_ts, image_filename,
                duration_hours, n_particles, 15, windage,
                _json.dumps(list(result.windage_coefficients)),
                result.data_mode, result.provider_name, "COMPLETED",
                _json.dumps(traj_summary),
                f"SRID=4326;{source_wkt}" if source_wkt else None,
                _json.dumps(candidates_json_list),
                segmentation_result.detected if segmentation_result else None,
                geometry_source,
                segmentation_result.spill_area_km2 if segmentation_result else None,
                segmentation_result.spill_pixel_count if segmentation_result else None,
                segmentation_result.spill_fraction if segmentation_result else None,
                segmentation_result.segmentation_confidence if segmentation_result else None,
                _json.dumps(img_analysis_db) if img_analysis_db else None,
                uncertainty_radius_m,
            ))
        except Exception as db_exc:
            conn.rollback()
            logger.warning("[Investigation] Falling back to 005-schema insert: %s", db_exc)
            cur.execute("""
                INSERT INTO historical_investigations
                    (spill_lat, spill_lon, spill_timestamp, image_filename,
                     duration_hours, particle_count, timestep_minutes, windage,
                     windage_ensemble, data_mode, provider_name, status,
                     trajectory_summary, source_region, candidates_json)
                VALUES
                    (%s, %s, %s, %s,
                     %s, %s, %s, %s,
                     %s, %s, %s, %s,
                     %s, %s::geography, %s)
                RETURNING id
            """, (
                latitude, longitude, spill_ts, image_filename,
                duration_hours, n_particles, 15, windage,
                _json.dumps(list(result.windage_coefficients)),
                result.data_mode, result.provider_name, "COMPLETED",
                _json.dumps(traj_summary),
                f"SRID=4326;{source_wkt}" if source_wkt else None,
                _json.dumps(candidates_json_list),
            ))

        row = cur.fetchone()
        saved_id = str(row[0]) if row else result.investigation_id
        conn.commit()
        cur.close()
        conn.close()
        investigation_id = saved_id
    except Exception as exc:
        print(f"[WARNING] Could not save historical investigation to DB: {exc}")
        investigation_id = result.investigation_id

    # -- 8. Build response -----------------------------------------------
    response = {
        "investigationId": investigation_id,
        "dataMode":        result.data_mode,
        "disclaimer": (
            "Results are CANDIDATE vessels only -- NOT confirmed polluters. "
            "The attribution score is an investigation priority metric, "
            "not a legally valid probability."
        ),
        "spillObservation": {
            "latitude":             latitude,
            "longitude":            longitude,
            "timestamp":            spill_ts.isoformat(),
            "imageFilename":        image_filename,
            "geometrySource":       geometry_source,
            "segmentationDetected": segmentation_result.detected if segmentation_result else False,
            "uncertaintyRadiusM":   uncertainty_radius_m,
        },
        "imageAnalysis": image_analysis_response,
        "model": {
            "type":                "lagrangian_particle_backtracking",
            "description":        "Lagrangian particle-based oil-spill transport and backtracking model",
            "particles":          n_particles,
            "timestepMinutes":    15,
            "durationHours":      duration_hours,
            "windageCoefficients": result.windage_coefficients,
        },
        "trajectory": {
            "timesteps": [
                {
                    "step":          s.step,
                    "hoursAgo":      s.hours_ago,
                    "timestamp":     s.timestamp,
                    "lat":           s.lat,
                    "lon":           s.lon,
                    "particleCount": s.particle_count,
                    "uOilMs":        s.u_oil,
                    "vOilMs":        s.v_oil,
                }
                for s in result.trajectory_steps
            ],
            "envelopePolygon": [
                {"lat": p[0], "lon": p[1]} for p in result.envelope_polygon
            ],
        },
        "sourceRegion": {
            "centroidLat":        result.source_region.centroid_lat,
            "centroidLon":        result.source_region.centroid_lon,
            "polygon":            [
                {"lat": p[0], "lon": p[1]} for p in result.source_region.polygon
            ],
            "uncertaintyNote":    result.source_region.uncertainty_note,
            "windageConsistency": result.source_region.windage_consistency,
        },
        "environment": {
            "dataMode":  result.data_mode,
            "provider":  result.provider_name,
            "note":      result.provider_note,
            "envError":  env_error,
        },
        "candidates":       candidates_json_list,
        "aisMatchingError": ais_error,
    }
    return response


@router.get("/health")
def health_v1():
    return {
        "status": "OPERATIONAL",
        "sar_model_loaded": is_sar_model_loaded(),
    }


app.include_router(router)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "sar_model_loaded": is_sar_model_loaded(),
    }