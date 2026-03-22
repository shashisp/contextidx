/**
 * contextidx provider for memorybench.
 *
 * Communicates with the contextidx FastAPI server over HTTP.
 * The server must be running before memorybench invokes this provider.
 *
 * Environment:
 *   CONTEXTIDX_BASE_URL  (default: http://localhost:8741)
 */

import type {
  Provider,
  ProviderConfig,
  IngestOptions,
  IngestResult,
  SearchOptions,
  IndexingProgressCallback,
} from "../../src/types/provider"
import type { UnifiedSession } from "../../src/types/unified"
import { CONTEXTIDX_PROMPTS } from "./prompts"

export class ContextIdxProvider implements Provider {
  name = "contextidx"
  prompts = CONTEXTIDX_PROMPTS
  concurrency = {
    default: 20,
    ingest: 5,
  }

  private baseUrl: string = "http://localhost:8741"

  async initialize(config: ProviderConfig): Promise<void> {
    this.baseUrl = config.baseUrl || this.baseUrl

    const resp = await fetch(`${this.baseUrl}/health`)
    if (!resp.ok) {
      throw new Error(
        `contextidx server not reachable at ${this.baseUrl}: ${resp.status}`
      )
    }
    const data = (await resp.json()) as { status: string; version: string }
    console.log(
      `[contextidx] Connected to server v${data.version} at ${this.baseUrl}`
    )
  }

  async ingest(
    sessions: UnifiedSession[],
    options: IngestOptions
  ): Promise<IngestResult> {
    const body = {
      sessions: sessions.map((s) => ({
        sessionId: s.sessionId,
        messages: s.messages.map((m) => ({
          role: m.role,
          content: m.content,
          timestamp: m.timestamp,
          speaker: m.speaker,
        })),
        metadata: s.metadata,
      })),
      containerTag: options.containerTag,
      metadata: options.metadata,
    }

    const resp = await fetch(`${this.baseUrl}/ingest`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })

    if (!resp.ok) {
      const text = await resp.text()
      throw new Error(`Ingest failed (${resp.status}): ${text}`)
    }

    const result = (await resp.json()) as { documentIds: string[] }
    return { documentIds: result.documentIds }
  }

  async awaitIndexing(
    result: IngestResult,
    _containerTag: string,
    onProgress?: IndexingProgressCallback
  ): Promise<void> {
    // contextidx indexes synchronously during ingest
    onProgress?.({
      completedIds: result.documentIds,
      failedIds: [],
      total: result.documentIds.length,
    })
  }

  async search(query: string, options: SearchOptions): Promise<unknown[]> {
    const body = {
      query,
      containerTag: options.containerTag,
      limit: Math.max(options.limit || 10, 40),
      threshold: options.threshold,
    }

    const resp = await fetch(`${this.baseUrl}/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })

    if (!resp.ok) {
      const text = await resp.text()
      throw new Error(`Search failed (${resp.status}): ${text}`)
    }

    const data = (await resp.json()) as {
      results: Array<{
        content: string
        score: number | null
        metadata: Record<string, unknown>
      }>
    }
    return data.results
  }

  async clear(containerTag: string): Promise<void> {
    const resp = await fetch(
      `${this.baseUrl}/clear/${encodeURIComponent(containerTag)}`,
      { method: "DELETE" }
    )
    if (!resp.ok) {
      const text = await resp.text()
      throw new Error(`Clear failed (${resp.status}): ${text}`)
    }
  }
}

export default ContextIdxProvider
