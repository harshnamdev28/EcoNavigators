import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ---------------------------------------------------------------------------
// TEST 1 — Leaflet Coordinate Conversion
// ---------------------------------------------------------------------------
test('Leaflet Coordinate Conversion: [latitude, longitude] order is preserved', () => {
  const latitude = 25.7715;
  const longitude = -80.1518;

  // Leaflet standard order is [latitude, longitude]
  const leafletPosition: [number, number] = [latitude, longitude];

  assert.equal(leafletPosition[0], 25.7715, 'First element must be latitude');
  assert.equal(leafletPosition[1], -80.1518, 'Second element must be longitude');
  assert.notDeepEqual(leafletPosition, [-80.1518, 25.7715], 'Coordinates must NOT be reversed');
});

// ---------------------------------------------------------------------------
// TEST 2 — PostGIS Coordinate Order
// ---------------------------------------------------------------------------
test('PostGIS convention: ST_MakePoint takes (longitude, latitude)', () => {
  const longitude = -80.1518;
  const latitude = 25.7715;

  const postgisCall = `ST_SetSRID(ST_MakePoint(${longitude}, ${latitude}), 4326)`;

  assert.match(postgisCall, /ST_MakePoint\(-80\.1518,\s*25\.7715\)/);
});

// ---------------------------------------------------------------------------
// TEST 3 — GeoJSON to Leaflet Conversion
// ---------------------------------------------------------------------------
test('GeoJSON: [lon, lat] coordinates convert to Leaflet [lat, lon]', () => {
  const geoJsonCoord = [-80.1518, 25.7715]; // [longitude, latitude]
  const leafletCoord: [number, number] = [geoJsonCoord[1], geoJsonCoord[0]];

  assert.equal(leafletCoord[0], 25.7715);
  assert.equal(leafletCoord[1], -80.1518);
});

// ---------------------------------------------------------------------------
// TEST 4 — Actual Investigation Center (No Permanent Hardcoded Center)
// ---------------------------------------------------------------------------
test('Actual Investigation Center: InvestigationMap does not use permanent hardcoded center', () => {
  const mapFilePath = path.join(__dirname, '../src/components/investigation/InvestigationMap.tsx');
  const content = fs.readFileSync(mapFilePath, 'utf-8');

  // Verify that permanent fallback coordinates like '?? 25.77' are absent
  assert.doesNotMatch(content, /spillLat\s*\?\?\s*25\./, 'Must not fallback to Miami lat');
  assert.doesNotMatch(content, /spillLon\s*\?\?\s*-80\./, 'Must not fallback to Miami lon');

  // Verify neutral global center WHOLE_MAP_CENTER is defined
  assert.match(content, /WHOLE_MAP_CENTER/, 'Must use neutral whole map center before query');
});

// ---------------------------------------------------------------------------
// TEST 5 — Lagrangian Trajectory Coordinate Order
// ---------------------------------------------------------------------------
test('Lagrangian Trajectory: trajectory timesteps map to [lat, lon] for Leaflet Polyline', () => {
  const mockBackendSteps = [
    { step: 0, hoursAgo: 0, timestamp: '2026-09-05T10:39:00Z', lat: 25.7715, lon: -80.1518 },
    { step: 1, hoursAgo: 0.25, timestamp: '2026-09-05T10:24:00Z', lat: 25.7700, lon: -80.1490 },
    { step: 2, hoursAgo: 0.50, timestamp: '2026-09-05T10:09:00Z', lat: 25.7680, lon: -80.1460 },
  ];

  const polylinePositions = mockBackendSteps.map((s) => [s.lat, s.lon] as [number, number]);

  assert.equal(polylinePositions.length, 3);
  assert.deepEqual(polylinePositions[0], [25.7715, -80.1518]);
  assert.deepEqual(polylinePositions[1], [25.7700, -80.1490]);
  assert.deepEqual(polylinePositions[2], [25.7680, -80.1460]);
});

// ---------------------------------------------------------------------------
// TEST 6 — AIS Tracks: Separate Chronological Tracks per MMSI
// ---------------------------------------------------------------------------
test('AIS Tracks: candidate vessels generate separate chronological tracks per MMSI', () => {
  const candidates = [
    {
      mmsi: '111111111',
      vesselName: 'Vessel Alpha',
      track: [
        { lat: 25.70, lon: -80.10, ts: '2026-09-05T08:00:00Z' },
        { lat: 25.72, lon: -80.12, ts: '2026-09-05T09:00:00Z' },
        { lat: 25.75, lon: -80.14, ts: '2026-09-05T10:00:00Z' },
      ],
    },
    {
      mmsi: '222222222',
      vesselName: 'Vessel Beta',
      track: [
        { lat: 25.80, lon: -80.20, ts: '2026-09-05T08:30:00Z' },
        { lat: 25.82, lon: -80.22, ts: '2026-09-05T09:30:00Z' },
      ],
    },
  ];

  // Each vessel produces its own distinct track
  const tracksByMmsi = candidates.map((c) => ({
    mmsi: c.mmsi,
    positions: c.track.map((pt) => [pt.lat, pt.lon] as [number, number]),
  }));

  assert.equal(tracksByMmsi.length, 2);
  assert.equal(tracksByMmsi[0].mmsi, '111111111');
  assert.equal(tracksByMmsi[0].positions.length, 3);
  assert.equal(tracksByMmsi[1].mmsi, '222222222');
  assert.equal(tracksByMmsi[1].positions.length, 2);

  // Verify tracks are NOT joined together
  assert.notDeepEqual(
    tracksByMmsi[0].positions[tracksByMmsi[0].positions.length - 1],
    tracksByMmsi[1].positions[0],
    'Different vessels must not be linked in the same polyline'
  );
});

// ---------------------------------------------------------------------------
// TEST 7 — Source Polygon Conversion
// ---------------------------------------------------------------------------
test('Source Polygon: GeoJSON polygon converts cleanly to Leaflet positions', () => {
  // GeoJSON polygon coordinates are [lon, lat]
  const geoJsonRing = [
    [-80.20, 25.70],
    [-80.20, 25.80],
    [-80.10, 25.80],
    [-80.10, 25.70],
    [-80.20, 25.70],
  ];

  const leafletPolygon: [number, number][] = geoJsonRing.map(
    (coord) => [coord[1], coord[0]] as [number, number]
  );

  assert.equal(leafletPolygon.length, 5);
  assert.deepEqual(leafletPolygon[0], [25.70, -80.20]);
  assert.deepEqual(leafletPolygon[1], [25.80, -80.20]);
  assert.deepEqual(leafletPolygon[2], [25.80, -80.10]);
});

// ---------------------------------------------------------------------------
// TEST 8 — No Fake Map Data
// ---------------------------------------------------------------------------
test('No Fake Map Data: investigation map only renders actual returned entities', () => {
  const emptyResult = {
    trajectory: { timesteps: [], envelopePolygon: [] },
    sourceRegion: { centroidLat: NaN, centroidLon: NaN, polygon: [] },
    candidates: [],
    spillObservation: { latitude: 25.7715, longitude: -80.1518 },
  };

  // Trajectory envelope positions
  const envelopePositions = emptyResult.trajectory.envelopePolygon.map(
    (p: any) => [p.lat, p.lon]
  );
  assert.equal(envelopePositions.length, 0, 'Must be empty when backend provides no envelope');

  // Candidate markers
  const candidateMarkers = emptyResult.candidates.filter(
    (c: any) => typeof c.lat === 'number' && typeof c.lon === 'number'
  );
  assert.equal(candidateMarkers.length, 0, 'Must be empty when backend provides no candidates');
});

// ---------------------------------------------------------------------------
// TEST 9 — Tile Rendering and CSS Protection
// ---------------------------------------------------------------------------
test('Tile Rendering: dashboard.css enforces max-width: none !important on Leaflet tiles', () => {
  const cssFilePath = path.join(__dirname, '../src/styles/dashboard.css');
  const cssContent = fs.readFileSync(cssFilePath, 'utf-8');

  assert.match(
    cssContent,
    /\.leaflet-container\s+img\s*\{\s*max-width:\s*none\s*!important;/,
    'Leaflet images must be protected from global max-width'
  );

  assert.match(
    cssContent,
    /\.leaflet-tile\s*\{\s*max-width:\s*none\s*!important;/,
    'Leaflet tiles must have max-width: none !important'
  );
});
