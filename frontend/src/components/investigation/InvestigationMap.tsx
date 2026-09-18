'use client';
import React, { useEffect, useRef } from 'react';
import {
  MapContainer, TileLayer, CircleMarker, Polygon, Polyline,
  Marker, Tooltip, LayersControl, useMap,
} from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { InvestigationResult, InvestigationCandidate } from '@/types/investigation';

const { BaseLayer } = LayersControl;

// ---------------------------------------------------------------------------
// Coordinate safety: ALL Leaflet positions use [lat, lon] tuples
// PostGIS returns ST_X=lon, ST_Y=lat — conversion is done in API layer
// ---------------------------------------------------------------------------

interface InvestigationMapProps {
  result: InvestigationResult | null;
  spillLat: number | null;
  spillLon: number | null;
  activeCandidate: InvestigationCandidate | null;
}

// Fit map to result bounds
function FitBounds({ result }: { result: InvestigationResult | null }) {
  const map = useMap();
  useEffect(() => {
    if (!result) return;
    const pts: [number, number][] = [];

    // Spill location
    pts.push([result.spillObservation.latitude, result.spillObservation.longitude]);

    // Trajectory centroids (sample every 4th to keep bounds tight)
    result.trajectory.timesteps
      .filter((_, i) => i % 4 === 0)
      .forEach((s) => pts.push([s.lat, s.lon]));

    // Source region centroid
    pts.push([result.sourceRegion.centroidLat, result.sourceRegion.centroidLon]);

    if (pts.length > 0) {
      const bounds = L.latLngBounds(pts);
      map.fitBounds(bounds, { padding: [40, 40] });
    }
  }, [result, map]);
  return null;
}

// Candidate vessel marker icon
function candidateIcon(isActive: boolean, rank: number): L.DivIcon {
  const color = isActive ? '#f43f5e' : rank === 1 ? '#fbbf24' : '#00d7b2';
  return L.divIcon({
    className: '',
    html: `<div style="
      width:28px; height:28px; border-radius:50%;
      background: ${color}22; border: 2px solid ${color};
      display:flex; align-items:center; justify-content:center;
      font-size:10px; font-weight:700; color:${color};
      font-family:monospace; box-shadow: 0 0 8px ${color}44;
    ">#${rank}</div>`,
    iconSize: [28, 28],
    iconAnchor: [14, 14],
  });
}

export default function InvestigationMap({
  result,
  spillLat,
  spillLon,
  activeCandidate,
}: InvestigationMapProps) {
  // Default center — Florida / Gulf of Mexico
  const defaultCenter: [number, number] = [spillLat ?? 25.77, spillLon ?? -80.15];
  const defaultZoom = spillLat ? 7 : 5;

  // Build Leaflet-format arrays: ALL use [lat, lon]
  const spillPos: [number, number] | null =
    result ? [result.spillObservation.latitude, result.spillObservation.longitude] : null;

  // Trajectory envelope polygon — already [lat, lon] from API
  const envelopePositions: [number, number][] =
    result?.trajectory.envelopePolygon.map((p) => [p.lat, p.lon] as [number, number]) ?? [];

  // Source region polygon — already [lat, lon] from API
  const sourcePositions: [number, number][] =
    result?.sourceRegion.polygon.map((p) => [p.lat, p.lon] as [number, number]) ?? [];

  // Trajectory centroid path
  const trajectoryPath: [number, number][] =
    result?.trajectory.timesteps.map((s) => [s.lat, s.lon] as [number, number]) ?? [];

  return (
    <MapContainer
      center={defaultCenter}
      zoom={defaultZoom}
      style={{ width: '100%', height: '100%', background: '#0b1723' }}
      zoomControl={true}
    >
      <LayersControl position="topright">
        <BaseLayer checked name="Dark Tactical">
          <TileLayer
            url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
            attribution="&copy; CartoDB"
            maxZoom={19}
          />
        </BaseLayer>
        <BaseLayer name="Satellite">
          <TileLayer
            url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
            attribution="&copy; Esri"
            maxZoom={19}
          />
        </BaseLayer>
        <BaseLayer name="Ocean">
          <TileLayer
            url="https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}"
            attribution="&copy; Esri"
            maxZoom={13}
          />
        </BaseLayer>
      </LayersControl>

      <FitBounds result={result} />

      {/* 1. Trajectory corridor envelope — orange fill */}
      {envelopePositions.length >= 3 && (
        <Polygon
          positions={envelopePositions}
          pathOptions={{
            color: '#f97316',
            fillColor: '#f97316',
            fillOpacity: 0.12,
            weight: 1.5,
            dashArray: '4 6',
          }}
        >
          <Tooltip sticky>
            <div style={{ fontFamily: 'monospace', fontSize: '11px' }}>
              <strong>BACKWARD TRAJECTORY CORRIDOR</strong><br />
              Lagrangian particle ensemble envelope<br />
              {result?.model.particles} particles · {result?.model.durationHours}h
            </div>
          </Tooltip>
        </Polygon>
      )}

      {/* 2. Trajectory centroid path — dashed orange line */}
      {trajectoryPath.length >= 2 && (
        <Polyline
          positions={trajectoryPath}
          pathOptions={{
            color: '#f97316',
            weight: 2,
            dashArray: '6 4',
            opacity: 0.7,
          }}
        />
      )}

      {/* 3. Source region — purple fill */}
      {sourcePositions.length >= 3 && (
        <Polygon
          positions={sourcePositions}
          pathOptions={{
            color: '#c026d3',
            fillColor: '#c026d3',
            fillOpacity: 0.18,
            weight: 2,
          }}
        >
          <Tooltip sticky>
            <div style={{ fontFamily: 'monospace', fontSize: '11px' }}>
              <strong style={{ color: '#c026d3' }}>CANDIDATE SOURCE REGION</strong><br />
              Centroid: {result?.sourceRegion.centroidLat.toFixed(4)}°N,{' '}
              {result?.sourceRegion.centroidLon.toFixed(4)}°E<br />
              <span style={{ color: '#999', fontSize: '10px' }}>
                NOT a confirmed spill origin.
              </span>
            </div>
          </Tooltip>
        </Polygon>
      )}

      {/* 4. Observed spill location — red circle */}
      {spillPos && (
        <CircleMarker
          center={spillPos}
          radius={14}
          pathOptions={{
            color: '#f43f5e',
            fillColor: '#f43f5e',
            fillOpacity: 0.25,
            weight: 2.5,
          }}
        >
          <Tooltip permanent direction="top" offset={[0, -16]}>
            <div style={{ fontFamily: 'monospace', fontSize: '10px', fontWeight: 700, color: '#f43f5e' }}>
              OBSERVED SPILL<br />
              {spillPos[0].toFixed(4)}°N, {spillPos[1].toFixed(4)}°E
            </div>
          </Tooltip>
        </CircleMarker>
      )}

      {/* 5. Candidate vessel positions */}
      {result?.candidates
        .filter((c) => c.lat !== null && c.lon !== null)
        .map((c) => {
          // c.lat and c.lon come from API (already lat/lon)
          const pos: [number, number] = [c.lat as number, c.lon as number];
          const isActive = activeCandidate?.mmsi === c.mmsi;
          return (
            <Marker
              key={c.mmsi}
              position={pos}
              icon={candidateIcon(isActive, c.rank)}
              zIndexOffset={isActive ? 1000 : 0}
            >
              <Tooltip>
                <div style={{ fontFamily: 'monospace', fontSize: '11px', minWidth: '180px' }}>
                  <div style={{ fontWeight: 700, marginBottom: '4px' }}>
                    #{c.rank} {c.vesselName}
                  </div>
                  <div>MMSI: {c.mmsi}</div>
                  {c.imo && <div>IMO: {c.imo}</div>}
                  <div style={{ color: '#00d7b2', marginTop: '4px' }}>
                    Score: <strong>{c.attributionScore.toFixed(0)}</strong>
                  </div>
                  <div>Min dist: {c.minDistanceKm.toFixed(1)} km</div>
                  <div>AIS pings: {c.matchingAisPings}</div>
                  {c.positionTimestamp && (
                    <div style={{ color: '#888', fontSize: '10px', marginTop: '2px' }}>
                      {new Date(c.positionTimestamp).toISOString().slice(0, 16)}Z
                    </div>
                  )}
                  <div style={{ color: '#888', fontSize: '9px', marginTop: '4px', borderTop: '1px solid #333', paddingTop: '3px' }}>
                    CANDIDATE ONLY — not confirmed
                  </div>
                </div>
              </Tooltip>
            </Marker>
          );
        })}
    </MapContainer>
  );
}
