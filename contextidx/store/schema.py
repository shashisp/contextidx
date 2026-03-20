"""SQL schema definitions for the contextidx internal metadata store."""

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS context_units (
    id           TEXT PRIMARY KEY,
    content      TEXT NOT NULL,
    embedding    TEXT,            -- JSON array of floats, NULL if not stored
    scope        TEXT NOT NULL,  -- JSON
    confidence   REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    decay_rate   REAL NOT NULL CHECK (decay_rate > 0),
    decay_model  TEXT NOT NULL CHECK (decay_model IN ('exponential', 'linear', 'step')),
    version      INTEGER NOT NULL DEFAULT 1,
    source       TEXT NOT NULL,
    superseded_by TEXT,
    created_at   TEXT NOT NULL,   -- ISO 8601
    expires_at   TEXT,            -- ISO 8601 or NULL
    archived_at  TEXT             -- ISO 8601 or NULL
);

CREATE TABLE IF NOT EXISTS context_graph (
    from_id      TEXT NOT NULL REFERENCES context_units(id),
    to_id        TEXT NOT NULL REFERENCES context_units(id),
    relationship TEXT NOT NULL CHECK (relationship IN ('supersedes', 'relates_to', 'version_of', 'caused_by')),
    created_at   TEXT NOT NULL,
    PRIMARY KEY (from_id, to_id, relationship)
);

CREATE TABLE IF NOT EXISTS decay_state (
    unit_id              TEXT PRIMARY KEY REFERENCES context_units(id),
    current_score        REAL NOT NULL,
    last_updated         TEXT NOT NULL,
    reinforcement_count  INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS wal (
    seq          INTEGER PRIMARY KEY AUTOINCREMENT,
    unit_id      TEXT NOT NULL,
    operation    TEXT NOT NULL,
    store_target TEXT NOT NULL,
    payload      TEXT NOT NULL,  -- JSON
    written_at   TEXT NOT NULL,
    applied_at   TEXT,
    status       TEXT NOT NULL CHECK (status IN ('pending', 'applied', 'failed'))
);

CREATE TABLE IF NOT EXISTS sync_checkpoints (
    store_name     TEXT PRIMARY KEY,
    last_synced_at TEXT NOT NULL,
    units_synced   INTEGER NOT NULL
);
"""

INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_context_units_created_at ON context_units(created_at);
CREATE INDEX IF NOT EXISTS idx_context_units_superseded_by ON context_units(superseded_by);
CREATE INDEX IF NOT EXISTS idx_context_graph_to_id ON context_graph(to_id, relationship);
CREATE INDEX IF NOT EXISTS idx_wal_status ON wal(status, written_at);
"""
