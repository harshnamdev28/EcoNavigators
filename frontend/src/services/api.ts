/**
 * EcoNavigators Centralized Backend API Client
 * Connects Next.js Frontend to FastAPI Backend.
 *
 * Rules:
 * - In LIVE mode: NEVER substitutes mock data. Unreachable backend throws an error.
 * - In DEMO mode: Explicitly serves demo/mock fixtures and flags the data as simulated.
 */

import type { OperationalStats, CandidateVessel } from '@/types/dashboard';
import type { SpillDetail, SpillSummary } from '@/types/spill';
import type { TrackedVessel, VesselDetail } from '@/types/vessel';
import type { AlertItem, AlertAcknowledgeResponse } from '@/types/alert';
import type {
  BacktrackingAnalysisResponse,
  CandidateRanking,
  ForensicCorrelationData,
  BacktrackResult,
} from '@/types/backtracking';
import type { SatelliteReconData, SatelliteAnomalyItem } from '@/types/satellite';
import type { Incident } from '@/types/incident';

import { mockOperationalStats, mockCandidateVessel } from '@/data/demo/mockDashboard';
import { mockSpillDetail } from '@/data/demo/mockSpills';
import { mockTrackedVessels } from '@/data/demo/mockVessels';
import { mockAlerts } from '@/data/demo/mockAlerts';
import { mockCandidateRankings, mockForensicEvidence } from '@/data/demo/mockBacktracking';

import { mapSpillDetail, mapSpillSummary } from './adapters/spillAdapter';
import { mapTrackedVessel, mapVesselDetail } from './adapters/vesselAdapter';
import { mapBacktrackingResponse } from './adapters/backtrackAdapter';
import { getAppMode } from '@/utils/appMode';

export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000/api/v1';
const AUTH_BASE_URL = API_BASE_URL.replace(/\/api\/v1\/?$/, '');

export class ApiError extends Error {
  status: number;
  data: any;

  constructor(message: string, status: number, data?: any) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.data = data;
  }
}

/**
 * Base HTTP fetcher.
 * In LIVE mode, errors are strictly thrown. Mock data is NEVER returned.
 */
export async function apiFetch<T>(
  endpoint: string,
  options?: RequestInit,
  demoFallback?: () => T
): Promise<T> {
  const mode = getAppMode();

  // If in DEMO mode and demoFallback is provided, use it directly when requested
  if (mode === 'demo' && demoFallback) {
    return demoFallback();
  }

  try {
    const res = await fetch(`${API_BASE_URL}${endpoint}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...(options?.headers || {}),
      },
    });

    if (!res.ok) {
      let errorBody: any = null;
      try {
        errorBody = await res.json();
      } catch {
        // non-json response
      }
      const message =
        errorBody?.detail ||
        errorBody?.message ||
        `Backend request failed with status ${res.status}: ${res.statusText}`;
      throw new ApiError(message, res.status, errorBody);
    }

    return (await res.json()) as T;
  } catch (err: any) {
    // In LIVE mode: STRICTLY throw. Never fabricate substitute values.
    if (mode === 'live') {
      if (err instanceof ApiError) {
        throw err;
      }
      throw new ApiError(
        `Unable to connect to maritime backend service at ${API_BASE_URL}. Verify FastAPI server is running.`,
        0,
        err
      );
    }

    // In DEMO mode: fallback to demo fixtures if available
    if (demoFallback) {
      console.warn(`[DEMO MODE] Backend unavailable for ${endpoint}. Serving isolated demo data.`);
      return demoFallback();
    }
    throw err;
  }
}

// ==========================================
// 1. Dashboard Operations
// ==========================================
export async function getOperationalStats(): Promise<OperationalStats> {
  return apiFetch<OperationalStats>(
    '/dashboard/stats',
    undefined,
    () => mockOperationalStats
  );
}

export async function getCandidateVessel(): Promise<CandidateVessel> {
  return apiFetch<CandidateVessel>(
    '/dashboard/candidate',
    undefined,
    () => mockCandidateVessel
  );
}

// ==========================================
// 2. Oil Spills
// ==========================================
export async function getSpillsList(): Promise<SpillSummary[]> {
  const rawList = await apiFetch<any[]>(
    '/spills',
    undefined,
    () => [mockSpillDetail]
  );
  return rawList.map(mapSpillSummary);
}

export async function getSpillDetail(spillId: string): Promise<SpillDetail> {
  const raw = await apiFetch<any>(
    `/spills/${encodeURIComponent(spillId)}`,
    undefined,
    () => mockSpillDetail
  );
  return mapSpillDetail(raw);
}

export async function executeBacktrack(
  spillId: string,
  hours = 24,
  windFactor = 0.03
): Promise<BacktrackResult> {
  return apiFetch<BacktrackResult>(
    `/spills/${encodeURIComponent(spillId)}/backtrack`,
    {
      method: 'POST',
      body: JSON.stringify({ simulationHours: hours, windDriftFactor: windFactor }),
    },
    () => ({
      spillId,
      estimatedOriginPoint: [28.45, -89.72],
      estimatedOriginTimestamp: '2023-10-24 08:42Z',
      confidenceScore: 92,
      driftTrajectory: [],
      status: 'COMPLETED',
    })
  );
}

// ==========================================
// 3. Monitored Vessels
// ==========================================
export async function getTrackedVessels(filters?: {
  category?: string;
  risk?: string;
  search?: string;
}): Promise<TrackedVessel[]> {
  const params = new URLSearchParams();
  if (filters?.category) params.append('category', filters.category);
  if (filters?.risk) params.append('risk', filters.risk);
  if (filters?.search) params.append('search', filters.search);
  const qs = params.toString() ? `?${params.toString()}` : '';

  const raw = await apiFetch<any[]>(
    `/vessels${qs}`,
    undefined,
    () => mockTrackedVessels
  );
  return raw.map(mapTrackedVessel);
}

export async function getVesselDetail(vesselId: string): Promise<VesselDetail> {
  const raw = await apiFetch<any>(
    `/vessels/${encodeURIComponent(vesselId)}`,
    undefined,
    () => ({ ...mockTrackedVessels[0], trajectory: [] })
  );
  return mapVesselDetail(raw);
}

export async function getVesselTrack(
  vesselId: string,
  hours = 24
): Promise<{ mmsi: string; track: { lat: number; lon: number; timestamp: string }[] }> {
  return apiFetch<{ mmsi: string; track: { lat: number; lon: number; timestamp: string }[] }>(
    `/vessels/${encodeURIComponent(vesselId)}/track?hours=${hours}`,
    undefined,
    () => ({ mmsi: vesselId, track: [] })
  );
}

// ==========================================
// 4. Backtracking & Forensic Analysis
// ==========================================
export async function getBacktrackingAnalysis(
  spillId: string,
  hours = 12,
  incidentId?: string
): Promise<BacktrackingAnalysisResponse> {
  const isInc = incidentId || (spillId && spillId.startsWith('INC-'));
  const queryParam = isInc
    ? `incidentId=${encodeURIComponent(incidentId || spillId)}`
    : `spill_id=${encodeURIComponent(spillId || '')}`;
  const raw = await apiFetch<any>(
    `/backtracking/analysis?${queryParam}&hours=${hours}`,
    undefined,
    () => ({
      spillId,
      incidentId: isInc ? (incidentId || spillId) : undefined,
      rankings: mockCandidateRankings,
      forensicEvidence: mockForensicEvidence,
      estimatedOriginPoint: [28.45, -89.72],
      estimatedDischargeTime: '2023-10-24 08:42Z',
      driftTrajectory: [],
      forecastTrajectory: [],
    })
  );
  return mapBacktrackingResponse(raw);
}

// ==========================================
// 5. Alerts
// ==========================================
export async function getAlerts(): Promise<AlertItem[]> {
  return apiFetch<AlertItem[]>(
    '/alerts',
    undefined,
    () => mockAlerts
  );
}

export async function acknowledgeAlert(alertId: string): Promise<boolean> {
  const res = await apiFetch<AlertAcknowledgeResponse>(
    `/alerts/${encodeURIComponent(alertId)}/acknowledge`,
    { method: 'POST' },
    () => ({ id: alertId, success: true, message: 'Acknowledged in Demo Mode' })
  );
  return Boolean(res?.success);
}

// ==========================================
// 6. Manual Incident Reporting
// ==========================================
export async function reportSpillIncident(reportData: {
  reporterName: string;
  contactEmail?: string;
  lat: number;
  lng: number;
  estimatedSizeSqKm?: number;
  spillAppearance?: string;
  notes?: string;
}): Promise<any> {
  return apiFetch('/reports', {
    method: 'POST',
    body: JSON.stringify(reportData),
  });
}

// ==========================================
// 7. System Health
// ==========================================
export async function getSystemHealth(): Promise<any> {
  return apiFetch(
    '/health',
    undefined,
    () => ({ status: 'OPERATIONAL', mode: 'DEMO' })
  );
}

// ==========================================
// 8. Copernicus Satellite Reconnaissance
// ==========================================
export async function getSatelliteRecon(
  lat: number,
  lon: number,
  timestamp?: string,
  satellite: 'sentinel-1' | 'sentinel-2' = 'sentinel-1',
  bufferKm = 8
): Promise<SatelliteReconData> {
  if (isNaN(lat) || isNaN(lon)) {
    throw new ApiError('COORDINATES UNAVAILABLE for satellite reconnaissance', 400);
  }
  const params = new URLSearchParams({
    lat: lat.toString(),
    lon: lon.toString(),
    satellite,
    buffer_km: bufferKm.toString(),
  });
  if (timestamp) params.append('timestamp', timestamp);
  return apiFetch<SatelliteReconData>(`/satellite/recon?${params.toString()}`);
}

export async function getVesselSatelliteRecon(
  vesselId: string,
  satellite: 'sentinel-1' | 'sentinel-2' = 'sentinel-1'
): Promise<SatelliteReconData> {
  return apiFetch<SatelliteReconData>(
    `/satellite/vessel/${encodeURIComponent(vesselId)}?satellite=${satellite}`
  );
}

export async function getSatelliteAnomalies(): Promise<SatelliteAnomalyItem[]> {
  return apiFetch<SatelliteAnomalyItem[]>('/satellite/anomalies', undefined, () => []);
}

// ==========================================
// 9. Centralized Authentication
// ==========================================
export async function loginUser(
  email: string,
  password: string
): Promise<{ token: string; user: any }> {
  try {
    const res = await fetch(`${AUTH_BASE_URL}/api/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });

    let data: any = null;
    try {
      data = await res.json();
    } catch {
      // non-JSON response
    }

    if (!res.ok) {
      if (res.status === 401 || res.status === 403) {
        throw new ApiError(data?.detail || 'Invalid email or password.', res.status, data);
      }
      if (res.status === 422) {
        throw new ApiError(
          data?.detail?.[0]?.msg || 'Validation error. Please check your credentials.',
          res.status,
          data
        );
      }
      throw new ApiError(
        data?.detail || `Authentication server error (${res.status}).`,
        res.status,
        data
      );
    }

    return data;
  } catch (err: any) {
    if (err instanceof ApiError) throw err;
    throw new ApiError(
      `Unable to connect to authentication server at ${AUTH_BASE_URL}.`,
      0,
      err
    );
  }
}

export async function signupUser(
  username: string,
  email: string,
  password: string
): Promise<{ token: string; user: any }> {
  try {
    const res = await fetch(`${AUTH_BASE_URL}/api/signup`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, email, password }),
    });

    let data: any = null;
    try {
      data = await res.json();
    } catch {
      // non-JSON response
    }

    if (!res.ok) {
      if (res.status === 422) {
        throw new ApiError(
          data?.detail?.[0]?.msg || 'Validation error. Please check your inputs.',
          res.status,
          data
        );
      }
      throw new ApiError(
        data?.detail || `Failed to create account (${res.status}).`,
        res.status,
        data
      );
    }

    return data;
  } catch (err: any) {
    if (err instanceof ApiError) throw err;
    throw new ApiError(
      `Unable to connect to authentication server at ${AUTH_BASE_URL}.`,
      0,
      err
    );
  }
}

// ==========================================
// 10. Incident Management & Vessel Identification
// ==========================================
export async function identifyVessel(
  mmsi: string,
  lat?: number,
  lon?: number,
  timestamp?: string
): Promise<Incident> {
  const cleanMmsi = String(mmsi).replace(/^MMSI_?/i, '').trim();
  const payload: Record<string, any> = { mmsi: cleanMmsi };
  if (lat !== undefined && lat !== null) payload.lat = lat;
  if (lon !== undefined && lon !== null) payload.lon = lon;
  if (timestamp) payload.timestamp = timestamp;

  return apiFetch<Incident>(
    '/identify',
    {
      method: 'POST',
      body: JSON.stringify(payload),
    },
    () => ({
      incidentId: `INC-${cleanMmsi}-DEMO`,
      mmsi: cleanMmsi,
      timestamp: timestamp || new Date().toISOString(),
      location: {
        lat: lat !== undefined && lat !== null ? lat : 28.45,
        lon: lon !== undefined && lon !== null ? lon : -89.72,
      },
      state: 'AIS_ANOMALY',
      status: 'AIS_ANOMALY',
      ais: {
        mmsi: cleanMmsi,
        anomalyDetected: true,
        reason: 'Simulated speed drop anomaly (Demo Mode)',
        anomalyType: 'SPEED_DROP',
        confidence: 0.85,
        lat: lat !== undefined && lat !== null ? lat : 28.45,
        lon: lon !== undefined && lon !== null ? lon : -89.72,
      },
      sar: {
        status: 'DEFERRED',
        reason: 'SAR pipeline deferred for future integration (Demo Mode)',
      },
      vessel: {
        mmsi: cleanMmsi,
        name: `Demo Vessel ${cleanMmsi}`,
        type: 'PRODUCT TANKER',
        imo: null,
        speed: '12.4 kts',
        heading: '142°',
        status: 'Underway',
      },
      backtracking: {
        status: 'READY',
      },
    })
  );
}

export async function getIncident(incidentId: string): Promise<Incident> {
  const cleanId = incidentId.trim();
  const cleanMmsi = cleanId.replace(/^INC-/, '').split('-')[0] || '368091590';
  return apiFetch<Incident>(
    `/incidents/${encodeURIComponent(cleanId)}`,
    undefined,
    () => ({
      incidentId: cleanId,
      mmsi: cleanMmsi,
      timestamp: new Date().toISOString(),
      state: 'AIS_ANOMALY',
      status: 'AIS_ANOMALY',
      location: {
        lat: 28.45,
        lon: -89.72,
      },
      sar: {
        status: 'DEFERRED',
        reason: 'SAR pipeline deferred for future integration (Demo Mode)',
      },
      ais: {
        anomalyDetected: true,
        reason: 'AIS anomaly detected in demo mode',
        anomalyType: 'SPEED_DROP',
        confidence: 0.85,
      },
      backtracking: {
        status: 'READY',
      },
    })
  );
}

export async function getIncidents(): Promise<Incident[]> {
  return apiFetch<Incident[]>(
    '/incidents',
    undefined,
    () => []
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Historical Oil Spill Investigation (Lagrangian particle backtracking)
// ─────────────────────────────────────────────────────────────────────────────
import type { InvestigationResult } from '@/types/investigation';

/**
 * POST /api/v1/historical-investigation
 *
 * Accepts multipart form data. Runs Lagrangian backtracking and AIS matching.
 * If an image is uploaded, runs MIT-B2 U-Net segmentation to derive the spill
 * polygon (IMAGE_DERIVED). Otherwise uses supplied lat/lon (POINT_ONLY).
 * Always returns structured result even if AIS matching finds no candidates.
 * NEVER falls back to fabricated vessels.
 *
 * @param params.dataMode  "DEMO" (mock env data) | "LIVE" (NOAA ERDDAP HYCOM)
 */
export async function runHistoricalInvestigation(params: {
  latitude: number;
  longitude: number;
  timestamp: string;
  durationHours: number;
  windage: number;
  nParticles: number;
  uncertaintyRadiusM?: number;
  dataMode?: 'DEMO' | 'LIVE';
  file?: File;
}): Promise<InvestigationResult> {
  const formData = new FormData();
  formData.append('latitude', String(params.latitude));
  formData.append('longitude', String(params.longitude));
  formData.append('timestamp', params.timestamp);
  formData.append('duration_hours', String(params.durationHours));
  formData.append('windage', String(params.windage));
  formData.append('n_particles', String(params.nParticles));
  formData.append('uncertainty_radius_m', String(params.uncertaintyRadiusM ?? 5000));
  formData.append('data_mode', params.dataMode ?? 'DEMO');
  if (params.file) {
    formData.append('file', params.file);
  }

  const res = await fetch(`${API_BASE_URL}/historical-investigation`, {
    method: 'POST',
    body: formData,
    // Do NOT set Content-Type — browser must set multipart boundary automatically
  });

  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const err = await res.json();
      detail = err?.detail || detail;
    } catch {
      // ignore JSON parse error
    }
    throw new ApiError(`Historical investigation failed: ${detail}`, res.status);
  }

  return res.json() as Promise<InvestigationResult>;
}
