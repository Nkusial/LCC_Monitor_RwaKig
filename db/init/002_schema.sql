CREATE TABLE IF NOT EXISTS aoi (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    geom geometry(Polygon, 4326) NOT NULL
);

CREATE TABLE IF NOT EXISTS sentinel_scenes (
    id TEXT PRIMARY KEY,
    collection TEXT NOT NULL,
    acquired_at TIMESTAMPTZ NOT NULL,
    cloud_cover DOUBLE PRECISION,
    footprint geometry(Polygon, 4326),
    assets JSONB NOT NULL DEFAULT '{}'::jsonb,
    properties JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id UUID PRIMARY KEY,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'created',
    parameters JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS change_polygons (
    id UUID PRIMARY KEY,
    run_id UUID REFERENCES pipeline_runs(id),
    change_type TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    area_m2 DOUBLE PRECISION NOT NULL CHECK (area_m2 >= 0),
    pre_date DATE,
    post_date DATE,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    geom geometry(Polygon, 4326) NOT NULL,
    properties JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_aoi_geom ON aoi USING gist (geom);
CREATE INDEX IF NOT EXISTS idx_sentinel_scenes_footprint ON sentinel_scenes USING gist (footprint);
CREATE INDEX IF NOT EXISTS idx_change_polygons_geom ON change_polygons USING gist (geom);
