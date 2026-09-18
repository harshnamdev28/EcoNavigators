// Types for Historical Oil Spill Investigation feature
// All map coordinates follow Leaflet convention: [lat, lon]

export interface InvestigationRequest {
  latitude: number;
  longitude: number;
  timestamp: string;
  durationHours: number;
  windage: number;
  nParticles: number;
  uncertaintyRadiusM: number;
  dataMode: 'DEMO' | 'LIVE';
  file?: File;
}

export interface TrajectoryTimestep {
  step: number;
  hoursAgo: number;
  timestamp: string;
  lat: number;
  lon: number;
  particleCount: number;
  uOilMs: number;
  vOilMs: number;
}

export interface CoordPoint {
  lat: number;
  lon: number;
}

export interface InvestigationTrajectory {
  timesteps: TrajectoryTimestep[];
  envelopePolygon: CoordPoint[];
}

export interface SourceRegion {
  centroidLat: number;
  centroidLon: number;
  polygon: CoordPoint[];
  uncertaintyNote: string;
  windageConsistency: string | null;
}

export interface EnvironmentInfo {
  dataMode: 'DEMO' | 'LIVE';
  provider: string;
  note: string;
  envError?: string | null;
}

export interface ImageAnalysisResult {
  model: string;                                        // "MIT-B2-U-Net"
  detected: boolean;
  geometrySource: 'IMAGE_DERIVED' | 'POINT_ONLY';
  spillPixelCount?: number;
  totalPixels?: number;
  spillFraction?: number;
  spillAreaKm2?: number | null;
  centroidRel?: [number, number] | null;
  boundingBoxRel?: [number, number, number, number] | null;
  segmentationConfidence?: number | null;
  maskPngBase64?: string | null;                        // base64 RGBA PNG for overlay
  reason?: string;
  error?: string | null;
  // Geographic polygon for map overlay — present when geometrySource === 'IMAGE_DERIVED'
  // Coordinates follow Leaflet convention: each point is {latitude, longitude}
  spillPolygonGeo?: Array<{ latitude: number; longitude: number }> | null;
}

export interface InvestigationCandidate {
  rank: number;
  mmsi: string;
  vesselName: string;
  imo: string | null;
  minDistanceKm: number;
  trajectoryOverlapScore: number;
  timeMatchScore: number;
  aisAnomalyScore: number;
  distanceScore: number;
  sourceRegionScore: number;
  attributionScore: number;
  matchingAisPings: number;
  lat: number | null;
  lon: number | null;
  positionTimestamp: string | null;
}

export interface InvestigationResult {
  investigationId: string;
  dataMode: 'DEMO' | 'LIVE';
  disclaimer: string;
  spillObservation: {
    latitude: number;
    longitude: number;
    timestamp: string;
    imageFilename: string | null;
    geometrySource: 'IMAGE_DERIVED' | 'POINT_ONLY';
    segmentationDetected: boolean;
    uncertaintyRadiusM: number;
  };
  imageAnalysis: ImageAnalysisResult | null;
  model: {
    type: string;
    description: string;
    particles: number;
    timestepMinutes: number;
    durationHours: number;
    windageCoefficients: number[];
  };
  trajectory: InvestigationTrajectory;
  sourceRegion: SourceRegion;
  environment: EnvironmentInfo;
  candidates: InvestigationCandidate[];
  aisMatchingError: string | null;
}
