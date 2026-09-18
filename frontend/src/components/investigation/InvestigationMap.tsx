'use client';
import React, { useEffect } from 'react';
import {
  MapContainer, TileLayer, CircleMarker, Polygon, Polyline,
  Marker, Tooltip, LayersControl, useMap,
} from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { InvestigationResult, InvestigationCandidate } from '@/types/investigation';

const { BaseLayer } = LayersControl;

// ---------------------------------------------------------------------------
// Geographical bounds: exactly one world copy, Web Mercator boundaries
// ---------------------------------------------------------------------------
const WORLD_BOUNDS: L.LatLngBoundsExpression = [
  [-85.05112878, -180],
  [85.05112878, 180],
];
const WHOLE_MAP_CENTER: [number, number] = [20.0, 0.0];
const WHOLE_MAP_ZOOM = 2;

// ---------------------------------------------------------------------------
// Coordinate safety: ALL Leaflet positions use [lat, lon] tuples.
//
// Convention (enforced throughout):
//   PostGIS  : ST_X(location) = longitude,  ST_Y(location) = latitude
//   API JSON : { latitude, longitude }   or  { lat, lon }
//   GeoJSON  : [longitude, latitude]
//   Leaflet  : [latitude, longitude]
//
// This component NEVER performs coordinate swapping.  All [lat, lon] tuples
// arrive from the API already in Leaflet-ready order.
// ---------------------------------------------------------------------------

interface InvestigationMapProps {
  result: InvestigationResult | null;
  spillLat: number | null;
  spillLon: number | null;
  activeCandidate: InvestigationCandidate | null;
}

// ---------------------------------------------------------------------------
// Map Controller: handles invalidateSize, initial rendering, and ResizeObserver
// ---------------------------------------------------------------------------
function InvestigationMapController({
  center,
  zoom,
}: {
  center: [number, number];
  zoom: number;
}) {
  const map = useMap();

  useEffect(() => {
    map.invalidateSize();

    const t1 = setTimeout(() => {
      map.invalidateSize();
      map.setView(center, zoom);
    }, 120);

    const t2 = setTimeout(() => {
      map.invalidateSize();
    }, 350);

    const container = map.getContainer();
    if (!container) {
      return () => {
        clearTimeout(t1);
        clearTimeout(t2);
      };
    }

    let resizeObserver: ResizeObserver | null = null;
    if (typeof ResizeObserver !== 'undefined') {
      resizeObserver = new ResizeObserver(() => {
        map.invalidateSize();
      });
      resizeObserver.observe(container);
    }

    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
      if (resizeObserver) {
        resizeObserver.disconnect();
      }
    };
  }, [map]);

  useEffect(() => {
    map.flyTo(center, zoom, { animate: true, duration: 1.0 });
    map.invalidateSize();
  }, [center, zoom, map]);

  return null;
}

// ---------------------------------------------------------------------------
// FitBounds — called whenever result changes; uses actual data geometry
// ---------------------------------------------------------------------------
function FitBounds({ result }: { result: InvestigationResult | null }) {
  const map = useMap();

  useEffect(() => {
    if (!result) return;

    map.invalidateSize();
    const pts: [number, number][] = [];

    // 1. Observed spill — authoritative position
    const obs = result.spillObservation;
    if (
      typeof obs.latitude === 'number' &&
      typeof obs.longitude === 'number' &&
      !isNaN(obs.latitude) &&
      !isNaN(obs.longitude)
    ) {
      pts.push([obs.latitude, obs.longitude]);
    }

    // 2. Trajectory centroids (sample every 3rd to keep bounds representative)
    result.trajectory.timesteps
      .filter((_, i) => i % 3 === 0)
      .forEach((s) => {
        if (
          typeof s.lat === 'number' &&
          typeof s.lon === 'number' &&
          !isNaN(s.lat) &&
          !isNaN(s.lon)
        ) {
          pts.push([s.lat, s.lon]);
        }
      });

    // 3. Source region centroid and vertices
    if (
      typeof result.sourceRegion.centroidLat === 'number' &&
      typeof result.sourceRegion.centroidLon === 'number' &&
      !isNaN(result.sourceRegion.centroidLat) &&
      !isNaN(result.sourceRegion.centroidLon)
    ) {
      pts.push([result.sourceRegion.centroidLat, result.sourceRegion.centroidLon]);
    }

    // 4. Candidate vessel positions & track points
    result.candidates.forEach((c) => {
      if (
        c.lat !== null &&
        c.lon !== null &&
        typeof c.lat === 'number' &&
        typeof c.lon === 'number' &&
        !isNaN(c.lat) &&
        !isNaN(c.lon)
      ) {
        pts.push([c.lat, c.lon]);
      }
      if (Array.isArray(c.track)) {
        c.track.forEach((pt) => {
          if (
            typeof pt.lat === 'number' &&
            typeof pt.lon === 'number' &&
            !isNaN(pt.lat) &&
            !isNaN(pt.lon)
          ) {
            pts.push([pt.lat, pt.lon]);
          }
        });
      }
    });

    // 5. Image-derived spill polygon if present
    const geo = (result.imageAnalysis as any)?.spillPolygonGeo;
    if (Array.isArray(geo)) {
      geo.forEach((p: any) => {
        if (
          typeof p.latitude === 'number' &&
          typeof p.longitude === 'number' &&
          !isNaN(p.latitude) &&
          !isNaN(p.longitude)
        ) {
          pts.push([p.latitude, p.longitude]);
        }
      });
    }

    if (pts.length >= 2) {
      const bounds = L.latLngBounds(pts);
      if (bounds.isValid()) {
        map.fitBounds(bounds, { padding: [50, 50], maxZoom: 13 });
      }
    } else if (pts.length === 1) {
      map.setView(pts[0], 8);
    }
  }, [result, map]);

  return null;
}

// ---------------------------------------------------------------------------
// Candidate vessel marker icon
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Main map component
// ---------------------------------------------------------------------------
export default function InvestigationMap({
  result,
  spillLat,
  spillLon,
  activeCandidate,
}: InvestigationMapProps) {
  // Map center: use actual spill coordinates if available, otherwise world center.
  // NEVER hardcode a geographic location as permanent fallback.
  const hasSpillCoords =
    spillLat !== null &&
    spillLon !== null &&
    typeof spillLat === 'number' &&
    typeof spillLon === 'number' &&
    !isNaN(spillLat) &&
    !isNaN(spillLon);

  const initialCenter: [number, number] = hasSpillCoords
    ? [spillLat as number, spillLon as number]
    : WHOLE_MAP_CENTER;
  const initialZoom = hasSpillCoords ? 7 : WHOLE_MAP_ZOOM;

  // Observed spill marker position — lat/lon from API, directly to Leaflet
  const spillPos: [number, number] | null = result
    ? [result.spillObservation.latitude, result.spillObservation.longitude]
    : hasSpillCoords
    ? [spillLat as number, spillLon as number]
    : null;

  // Trajectory corridor envelope — API returns {lat, lon} objects
  const envelopePositions: [number, number][] =
    result?.trajectory.envelopePolygon
      .filter((p) => typeof p.lat === 'number' && typeof p.lon === 'number')
      .map((p) => [p.lat, p.lon] as [number, number]) ?? [];

  // Source region polygon — API returns {lat, lon} objects
  const sourcePositions: [number, number][] =
    result?.sourceRegion.polygon
      .filter((p) => typeof p.lat === 'number' && typeof p.lon === 'number')
      .map((p) => [p.lat, p.lon] as [number, number]) ?? [];

  // Trajectory centroid path — from trajectory timesteps
  const trajectoryPath: [number, number][] =
    result?.trajectory.timesteps
      .filter((s) => typeof s.lat === 'number' && typeof s.lon === 'number')
      .map((s) => [s.lat, s.lon] as [number, number]) ?? [];

  // IMAGE_DERIVED spill polygon — geographic coordinates returned by API
  const imageSpillPolygonPositions: [number, number][] = (() => {
    const geo = (result?.imageAnalysis as any)?.spillPolygonGeo;
    if (!Array.isArray(geo) || geo.length < 3) return [];
    return geo
      .filter(
        (p: any) =>
          typeof p.latitude === 'number' &&
          typeof p.longitude === 'number' &&
          !isNaN(p.latitude) &&
          !isNaN(p.longitude)
      )
      .map((p: any) => [p.latitude, p.longitude] as [number, number]);
  })();

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%', overflow: 'hidden' }}>
      <MapContainer
        center={initialCenter}
        zoom={initialZoom}
        minZoom={2}
        maxBounds={WORLD_BOUNDS}
        maxBoundsViscosity={1.0}
        worldCopyJump={false}
        style={{ width: '100%', height: '100%', position: 'relative', overflow: 'hidden', background: '#070e17' }}
        zoomControl={true}
      >
        <InvestigationMapController center={initialCenter} zoom={initialZoom} />

        <LayersControl position="topright">
          <BaseLayer checked name="Dark Tactical">
            <TileLayer
              url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
              attribution="&copy; OpenStreetMap contributors &copy; CARTO"
              noWrap={true}
              bounds={WORLD_BOUNDS}
              maxZoom={18}
            />
          </BaseLayer>
          <BaseLayer name="3D Satellite">
            <TileLayer
              url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
              attribution="&copy; Esri &mdash; Earthstar Geographics, Maxar"
              noWrap={true}
              bounds={WORLD_BOUNDS}
              maxZoom={18}
            />
          </BaseLayer>
          <BaseLayer name="3D Ocean">
            <TileLayer
              url="https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}"
              attribution="&copy; Esri &mdash; GEBCO, NOAA, National Geographic"
              noWrap={true}
              bounds={WORLD_BOUNDS}
              maxZoom={13}
            />
          </BaseLayer>
        </LayersControl>

        {/* Fit bounds to actual data once result arrives */}
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
                Centroid: {result?.sourceRegion.centroidLat.toFixed(4)}°,{' '}
                {result?.sourceRegion.centroidLon.toFixed(4)}°<br />
                <span style={{ color: '#999', fontSize: '10px' }}>
                  NOT a confirmed spill origin.
                </span>
              </div>
            </Tooltip>
          </Polygon>
        )}

        {/* 4. IMAGE_DERIVED spill polygon — cyan fill (only when geographic polygon available) */}
        {imageSpillPolygonPositions.length >= 3 &&
          result?.spillObservation.geometrySource === 'IMAGE_DERIVED' && (
          <Polygon
            positions={imageSpillPolygonPositions}
            pathOptions={{
              color: '#06b6d4',
              fillColor: '#06b6d4',
              fillOpacity: 0.20,
              weight: 2,
              dashArray: '3 4',
            }}
          >
            <Tooltip sticky>
              <div style={{ fontFamily: 'monospace', fontSize: '11px' }}>
                <strong style={{ color: '#06b6d4' }}>MIT-B2 SEGMENTED SPILL</strong><br />
                Image-derived geographic polygon<br />
                <span style={{ color: '#999', fontSize: '10px' }}>
                  Approximate — no satellite georeferencing.
                </span>
              </div>
            </Tooltip>
          </Polygon>
        )}

        {/* 5. Separate Historical AIS Vessel Tracks per Candidate (Phase 10) */}
        {result?.candidates.map((c) => {
          if (!c.track || c.track.length < 2) return null;
          const trackPositions = c.track
            .filter(
              (pt) =>
                typeof pt.lat === 'number' &&
                typeof pt.lon === 'number' &&
                !isNaN(pt.lat) &&
                !isNaN(pt.lon)
            )
            .map((pt) => [pt.lat, pt.lon] as [number, number]);
          if (trackPositions.length < 2) return null;

          const isActive = activeCandidate?.mmsi === c.mmsi;
          const isRank1 = c.rank === 1;
          const trackColor = isActive ? '#f43f5e' : isRank1 ? '#fbbf24' : '#00d7b2';

          return (
            <Polyline
              key={`candidate-track-${c.mmsi}`}
              positions={trackPositions}
              pathOptions={{
                color: trackColor,
                weight: isActive ? 3 : isRank1 ? 2.5 : 1.5,
                dashArray: isActive ? undefined : '4 4',
                opacity: isActive ? 0.95 : 0.6,
              }}
            >
              <Tooltip sticky>
                <div style={{ fontFamily: 'monospace', fontSize: '11px' }}>
                  <strong style={{ color: trackColor }}>
                    #{c.rank} {c.vesselName} (Track)
                  </strong><br />
                  MMSI: {c.mmsi}<br />
                  Pings: {trackPositions.length}<br />
                  Attribution score: {c.attributionScore.toFixed(0)}
                </div>
              </Tooltip>
            </Polyline>
          );
        })}

        {/* 6. Observed spill location — red circle */}
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
                {spillPos[0].toFixed(4)}°, {spillPos[1].toFixed(4)}°
              </div>
            </Tooltip>
          </CircleMarker>
        )}

        {/* 7. Candidate vessel positions */}
        {result?.candidates
          .filter(
            (c) =>
              c.lat !== null &&
              c.lon !== null &&
              typeof c.lat === 'number' &&
              typeof c.lon === 'number' &&
              !isNaN(c.lat) &&
              !isNaN(c.lon)
          )
          .map((c) => {
            // c.lat and c.lon come from the API which extracts:
            //   ST_Y(location) = latitude, ST_X(location) = longitude
            // Leaflet receives [latitude, longitude] — no swapping needed.
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
    </div>
  );
}
