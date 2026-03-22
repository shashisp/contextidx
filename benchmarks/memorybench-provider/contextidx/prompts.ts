/**
 * Custom prompts for the contextidx memorybench provider.
 *
 * Each context entry contains conversation excerpts with speaker labels
 * and session date headers.  The prompt instructs the model to resolve
 * relative dates and combine facts across entries for multi-hop questions.
 */

import type { ProviderPrompts } from "../../src/types/prompts"

function parseSessionDate(content: string): Date | null {
  const match = content.match(/\[Session date:\s*(.+?)\]/)
  if (!match) return null
  try {
    return new Date(match[1])
  } catch {
    return null
  }
}

export const CONTEXTIDX_PROMPTS: ProviderPrompts = {
  answerPrompt: (
    question: string,
    context: unknown[],
    questionDate?: string
  ): string => {
    const typed = context as Array<{ content: string; score?: number; metadata?: Record<string, unknown> }>

    // Sort entries chronologically (oldest first) so the LLM sees the
    // natural progression of events -- critical for temporal reasoning.
    const sorted = [...typed].sort((a, b) => {
      const da = parseSessionDate(a.content)
      const db = parseSessionDate(b.content)
      if (da && db) return da.getTime() - db.getTime()
      if (da) return -1
      if (db) return 1
      return 0
    })

    const entries = sorted
      .map((c, i) => {
        const score = c.score != null ? ` [relevance: ${c.score.toFixed(2)}]` : ""
        return `--- Entry ${i + 1}${score} ---\n${c.content}`
      })
      .join("\n\n")

    const dateSection = questionDate
      ? `\nThe question is being asked on: ${questionDate}\nUse this date to resolve any relative time references (e.g. "recently", "a few months ago").`
      : ""

    return `You are answering a question about people based on their conversation history.
${dateSection}

INSTRUCTIONS:
1. CHRONOLOGICAL CONTEXT: Entries below are sorted oldest-first. Each begins with a "[Session date: ...]" header.
2. SPEAKER ATTRIBUTION: Speakers are labeled (e.g. "Caroline:", "Melanie:"). Before answering, identify which speaker the question is about and ONLY attribute their statements to them.
3. TEMPORAL RESOLUTION: Resolve ALL relative time references using the session date. Examples:
   - "yesterday" in an 8 May 2023 session = 7 May 2023
   - "last week" in a 25 May 2023 session = week of 14-20 May 2023
   - "next month" in a May 2023 session = June 2023
   - "last year" in a 2023 session = 2022
4. MULTI-ENTRY SYNTHESIS: Combine facts from ALL entries. For list questions ("what does X do?", "what books has X read?"), scan every entry and compile a complete list.
5. INFERENCE: Draw reasonable inferences from context clues. For example, if someone mentions "that tough breakup", infer they are currently single. If someone says "I started looking into counseling after getting support", infer counseling was motivated by receiving support.
6. CONFLICT RESOLUTION: When the same fact appears in multiple sessions, prefer the most recent session's version.
7. ALWAYS ANSWER: Provide your best answer based on available context. Even if the evidence is indirect or requires inference, give a concrete answer. Do NOT say "not enough information" unless the context is truly completely silent on the topic.

Context:
${entries || "(no relevant context found)"}

Question: ${question}

Answer concisely and specifically. Use exact names, dates, and details from the context.`
  },
}
