# contextidx memorybench provider

TypeScript provider adapter for [memorybench](https://github.com/supermemoryai/memorybench) that benchmarks contextidx against mem0, zep, and supermemory.

## Setup

### 1. Clone memorybench and install dependencies

```bash
git clone https://github.com/supermemoryai/memorybench.git
cd memorybench
bun install
```

### 2. Copy the provider into memorybench

```bash
cp -r /path/to/ctxidx/memorybench-provider/contextidx memorybench/src/providers/contextidx
```

### 3. Register the provider

Edit `memorybench/src/providers/index.ts` to add:

```typescript
import { ContextIdxProvider } from "./contextidx"

// Add to providers map:
const providers: Record<ProviderName, new () => Provider> = {
  // ... existing providers ...
  contextidx: ContextIdxProvider,
}
```

Edit `memorybench/src/types/provider.ts` to add `"contextidx"` to the `ProviderName` type:

```typescript
export type ProviderName = "supermemory" | "mem0" | "zep" | "filesystem" | "rag" | "contextidx"
```

Edit `memorybench/src/utils/config.ts` to add the config case:

```typescript
case "contextidx":
  return { apiKey: "", baseUrl: process.env.CONTEXTIDX_BASE_URL || "http://localhost:8741" }
```

### 4. Start the contextidx server

```bash
cd /path/to/ctxidx
OPENAI_API_KEY=sk-... uvicorn contextidx.server:app --host 0.0.0.0 --port 8741
```

### 5. Run memorybench

```bash
cd memorybench
OPENAI_API_KEY=sk-... bun run src/index.ts run -p contextidx -b locomo
```

## Configuration

Environment variables for the contextidx server:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | (required) | OpenAI API key for embeddings |
| `CONTEXTIDX_HALF_LIFE` | `30` | Half-life in days |
| `CONTEXTIDX_DETECTION` | `semantic` | Conflict detection mode |
| `CONTEXTIDX_STRATEGY` | `LAST_WRITE_WINS` | Conflict resolution strategy |
| `CONTEXTIDX_STORE_PATH` | `.contextidx/memorybench.db` | SQLite store path |
| `CONTEXTIDX_BACKEND` | `memory` | `memory` or a pgvector DSN |
| `CONTEXTIDX_RECENCY_BIAS` | (none) | Recency bias 0-1 |
