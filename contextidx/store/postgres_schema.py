"""PostgreSQL schema definitions for the contextidx internal metadata store.

Mirrors the SQLite schema in schema.py, ported to native Postgres types:
JSONB for scope/payload, TIMESTAMPTZ for datetimes, SERIAL for WAL seq.
"""

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS context_units (
    id           TEXT PRIMARY KEY,
    content      TEXT NOT NULL,
    embedding    TEXT,
    scope        JSONB NOT NULL,
    confidence   DOUBLE PRECISION NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    decay_rate   DOUBLE PRECISION NOT NULL CHECK (decay_rate > 0),
    decay_model  TEXT NOT NULL CHECK (decay_model IN ('exponential', 'linear', 'step')),
    version      INTEGER NOT NULL DEFAULT 1,
    source       TEXT NOT NULL,
    superseded_by TEXT,
    created_at   TIMESTAMPTZ NOT NULL,
    expires_at   TIMESTAMPTZ,
    archived_at  TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS context_graph (
    from_id      TEXT NOT NULL REFERENCES context_units(id),
    to_id        TEXT NOT NULL REFERENCES context_units(id),
    relationship TEXT NOT NULL CHECK (relationship IN ('supersedes', 'relates_to', 'version_of', 'caused_by')),
    created_at   TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (from_id, to_id, relationship)
);

CREATE TABLE IF NOT EXISTS decay_state (
    unit_id              TEXT PRIMARY KEY REFERENCES context_units(id),
    current_score        DOUBLE PRECISION NOT NULL,
    last_updated         TIMESTAMPTZ NOT NULL,
    reinforcement_count  INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS wal (
    seq          SERIAL PRIMARY KEY,
    unit_id      TEXT NOT NULL,
    operation    TEXT NOT NULL,
    store_target TEXT NOT NULL,
    payload      JSONB NOT NULL,
    written_at   TIMESTAMPTZ NOT NULL,
    applied_at   TIMESTAMPTZ,
    status       TEXT NOT NULL CHECK (status IN ('pending', 'applied', 'failed'))
);

CREATE TABLE IF NOT EXISTS sync_checkpoints (
    store_name     TEXT PRIMARY KEY,
    last_synced_at TIMESTAMPTZ NOT NULL,
    units_synced   INTEGER NOT NULL
);
"""

INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_context_units_created_at ON context_units(created_at);
CREATE INDEX IF NOT EXISTS idx_context_units_superseded_by ON context_units(superseded_by);
CREATE INDEX IF NOT EXISTS idx_context_graph_to_id ON context_graph(to_id, relationship);
CREATE INDEX IF NOT EXISTS idx_wal_status ON wal(status, written_at);
"""
