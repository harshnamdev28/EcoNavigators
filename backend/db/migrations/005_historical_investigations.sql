-- Migration 005: Historical Oil Spill Investigation Table
-- Created for: Lagrangian particle backtracking investigation feature
-- NOTE: Particle positions are NOT stored per-row.
--       Only compact trajectory summaries (JSONB) and source region polygon are stored.

CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS historical_investigations (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Observed spill input
    spill_lat           DOUBLE PRECISION NOT NULL,
    spill_lon           DOUBLE PRECISION NOT NULL,
    spill_timestamp     TIMESTAMPTZ NOT NULL,
    image_filename      TEXT,

    -- Model parameters
    duration_hours      INTEGER NOT NULL DEFAULT 24,
    particle_count      INTEGER NOT NULL DEFAULT 500,
    timestep_minutes    INTEGER NOT NULL DEFAULT 15,
    windage             FLOAT NOT NULL DEFAULT 0.03,
    windage_ensemble    JSONB,               -- e.g. [0.01, 0.03, 0.04]

    -- Data provenance
    data_mode           TEXT NOT NULL DEFAULT 'DEMO',   -- 'DEMO' or 'LIVE'
    provider_name       TEXT,

    -- Status
    status              TEXT NOT NULL DEFAULT 'COMPLETED',

    -- Results (compact JSONB -- NOT per-particle rows)
    -- trajectory_summary: array of {step, hours_ago, ts, lat, lon} centroids
    trajectory_summary  JSONB,

    -- Source region as PostGIS polygon for spatial queries
    -- COORDINATE ORDER: ST_MakePoint(lon, lat) -- PostGIS convention
    source_region       GEOGRAPHY(Polygon, 4326),

    -- Top candidates as JSONB array
    candidates_json     JSONB,

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Spatial index on source region polygon
CREATE INDEX IF NOT EXISTS idx_hist_inv_source_region
    ON historical_investigations USING GIST (source_region);

-- Index on spill timestamp for time-range queries
CREATE INDEX IF NOT EXISTS idx_hist_inv_spill_ts
    ON historical_investigations (spill_timestamp DESC);

-- Index on created_at for recent queries
CREATE INDEX IF NOT EXISTS idx_hist_inv_created_at
    ON historical_investigations (created_at DESC);

-- Composite index for spatial lookup by approximate location
CREATE INDEX IF NOT EXISTS idx_hist_inv_lat_lon
    ON historical_investigations (spill_lat, spill_lon);
