#!/usr/bin/env bash
#
# Set up a local PostgreSQL database with pgvector for contextidx.
#
# Prerequisites:
#   - PostgreSQL running locally (e.g. Postgres.app)
#   - pgvector extension available (bundled with Postgres.app)
#
# Usage:
#   ./scripts/setup_pgvector.sh            # create DB with defaults
#   ./scripts/setup_pgvector.sh --drop     # drop and recreate
#
# After running, add to your .env:
#   CONTEXTIDX_BACKEND=postgresql://localhost:5432/contextidx

set -euo pipefail

DB_NAME="${CONTEXTIDX_DB_NAME:-contextidx}"
DB_USER="${CONTEXTIDX_DB_USER:-$(whoami)}"
DB_HOST="${CONTEXTIDX_DB_HOST:-localhost}"
DB_PORT="${CONTEXTIDX_DB_PORT:-5432}"
TABLE_NAME="contextidx_vectors"
DIMENSIONS=1536
DROP=false

if [[ "${1:-}" == "--drop" ]]; then
    DROP=true
fi

echo "=== contextidx pgvector setup ==="
echo "  Host:  $DB_HOST:$DB_PORT"
echo "  User:  $DB_USER"
echo "  DB:    $DB_NAME"
echo ""

if $DROP; then
    echo "Dropping database '$DB_NAME' if it exists..."
    psql -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -d postgres \
        -c "DROP DATABASE IF EXISTS $DB_NAME;" 2>/dev/null || true
fi

DB_EXISTS=$(psql -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -d postgres \
    -tAc "SELECT 1 FROM pg_database WHERE datname = '$DB_NAME';" 2>/dev/null || echo "")

if [[ "$DB_EXISTS" == "1" ]]; then
    echo "Database '$DB_NAME' already exists."
else
    echo "Creating database '$DB_NAME'..."
    psql -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -d postgres \
        -c "CREATE DATABASE $DB_NAME;"
    echo "Database created."
fi

echo "Enabling pgvector extension..."
psql -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -d "$DB_NAME" \
    -c "CREATE EXTENSION IF NOT EXISTS vector;"

echo "Creating vectors table..."
psql -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -d "$DB_NAME" <<SQL
CREATE TABLE IF NOT EXISTS $TABLE_NAME (
    id        TEXT PRIMARY KEY,
    embedding vector($DIMENSIONS),
    metadata  JSONB DEFAULT '{}'::jsonb,
    content   TEXT DEFAULT ''
);

-- Add columns for existing tables being upgraded
DO \$\$ BEGIN
    ALTER TABLE $TABLE_NAME ADD COLUMN IF NOT EXISTS content TEXT DEFAULT '';
EXCEPTION WHEN duplicate_column THEN NULL;
END \$\$;

DO \$\$ BEGIN
    ALTER TABLE $TABLE_NAME
        ADD COLUMN IF NOT EXISTS search_vector tsvector
        GENERATED ALWAYS AS (to_tsvector('english', coalesce(content, ''))) STORED;
EXCEPTION WHEN duplicate_column THEN NULL;
END \$\$;

-- HNSW index for fast approximate nearest neighbor search
CREATE INDEX IF NOT EXISTS idx_${TABLE_NAME}_hnsw
    ON $TABLE_NAME
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- GIN index on metadata for fast scope filtering
CREATE INDEX IF NOT EXISTS idx_${TABLE_NAME}_metadata
    ON $TABLE_NAME
    USING gin (metadata jsonb_path_ops);

-- GIN index on tsvector for full-text (BM25) search
CREATE INDEX IF NOT EXISTS idx_${TABLE_NAME}_search_vector
    ON $TABLE_NAME
    USING GIN (search_vector);
SQL

ROW_COUNT=$(psql -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -d "$DB_NAME" \
    -tAc "SELECT COUNT(*) FROM $TABLE_NAME;")

echo ""
echo "=== Setup complete ==="
echo "  Table:   $TABLE_NAME ($ROW_COUNT rows)"
echo "  Index:   HNSW (cosine, m=16, ef_construction=64)"
echo "  Dims:    $DIMENSIONS"
echo ""
echo "Connection string:"
echo "  postgresql://$DB_USER@$DB_HOST:$DB_PORT/$DB_NAME"
echo ""
echo "Add to your .env:"
echo "  CONTEXTIDX_BACKEND=postgresql://$DB_USER@$DB_HOST:$DB_PORT/$DB_NAME"
