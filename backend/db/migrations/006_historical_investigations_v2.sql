-- Migration 006: Add image segmentation columns to historical_investigations
-- This migration extends the table created by 005_historical_investigations.sql.
-- Uses IF NOT EXISTS / IF EXISTS guards so it is idempotent.

-- Image segmentation result from MIT-B2 U-Net
ALTER TABLE historical_investigations
    ADD COLUMN IF NOT EXISTS segmentation_detected   BOOLEAN,
    ADD COLUMN IF NOT EXISTS geometry_source         TEXT DEFAULT 'POINT_ONLY',
    ADD COLUMN IF NOT EXISTS spill_area_km2          FLOAT,
    ADD COLUMN IF NOT EXISTS spill_pixel_count       INTEGER,
    ADD COLUMN IF NOT EXISTS spill_fraction          FLOAT,
    ADD COLUMN IF NOT EXISTS segmentation_confidence FLOAT,
    ADD COLUMN IF NOT EXISTS image_analysis_json     JSONB,

-- Input uncertainty
    ADD COLUMN IF NOT EXISTS uncertainty_radius_m    FLOAT DEFAULT 5000.0;

-- Index on geometry_source for filtering queries
CREATE INDEX IF NOT EXISTS idx_hist_inv_geometry_source
    ON historical_investigations (geometry_source);

COMMENT ON COLUMN historical_investigations.segmentation_detected   IS 'True if MIT-B2 U-Net detected a spill in the uploaded image';
COMMENT ON COLUMN historical_investigations.geometry_source          IS 'IMAGE_DERIVED: polygon from segmentation; POINT_ONLY: user-supplied lat/lon';
COMMENT ON COLUMN historical_investigations.spill_area_km2           IS 'Estimated spill area from pixel count (approximate)';
COMMENT ON COLUMN historical_investigations.image_analysis_json      IS 'Full SpillSegmentationResult as JSON for audit';
