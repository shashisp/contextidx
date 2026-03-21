#!/usr/bin/env bash
#
# Run memorybench with the contextidx provider.
#
# Prerequisites:
#   - OPENAI_API_KEY set in environment
#   - memorybench cloned and set up (see memorybench-provider/README.md)
#   - pip install contextidx[server]
#
# Usage:
#   ./scripts/run_memorybench.sh [benchmark] [extra memorybench args...]
#
# Examples:
#   ./scripts/run_memorybench.sh locomo
#   ./scripts/run_memorybench.sh locomo -l 10
#   ./scripts/run_memorybench.sh longmemeval -j sonnet-4

set -euo pipefail

BENCHMARK="${1:-locomo}"
shift || true

CTXIDX_PORT="${CONTEXTIDX_PORT:-8741}"
MEMORYBENCH_DIR="${MEMORYBENCH_DIR:-../memorybench}"
CTXIDX_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: OPENAI_API_KEY is not set" >&2
    exit 1
fi

if [ ! -d "$MEMORYBENCH_DIR" ]; then
    echo "ERROR: memorybench directory not found at $MEMORYBENCH_DIR" >&2
    echo "Set MEMORYBENCH_DIR or clone memorybench alongside ctxidx." >&2
    exit 1
fi

cleanup() {
    if [ -n "${SERVER_PID:-}" ]; then
        echo "Stopping contextidx server (PID $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "Starting contextidx server on port $CTXIDX_PORT..."
OPENAI_API_KEY="$OPENAI_API_KEY" \
CONTEXTIDX_DETECTION="${CONTEXTIDX_DETECTION:-semantic}" \
CONTEXTIDX_HALF_LIFE="${CONTEXTIDX_HALF_LIFE:-30}" \
CONTEXTIDX_STORE_PATH="${CONTEXTIDX_STORE_PATH:-.contextidx/memorybench.db}" \
    uvicorn contextidx.server:app \
        --host 127.0.0.1 \
        --port "$CTXIDX_PORT" \
        --log-level info &
SERVER_PID=$!

echo "Waiting for server to be ready..."
for i in $(seq 1 30); do
    if curl -sf "http://127.0.0.1:$CTXIDX_PORT/health" >/dev/null 2>&1; then
        echo "Server ready."
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "ERROR: Server failed to start within 30 seconds" >&2
        exit 1
    fi
    sleep 1
done

echo "Running memorybench: benchmark=$BENCHMARK provider=contextidx $*"
cd "$MEMORYBENCH_DIR"
CONTEXTIDX_BASE_URL="http://127.0.0.1:$CTXIDX_PORT" \
OPENAI_API_KEY="$OPENAI_API_KEY" \
    bun run src/index.ts run \
        -p contextidx \
        -b "$BENCHMARK" \
        "$@"

echo "Done. Results are in $MEMORYBENCH_DIR/data/runs/"
