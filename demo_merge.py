"""Demo: current merge strategy vs new merge strategies.

Run: .venv/bin/python demo_merge.py

=== RESULTS ===

============================================================
  SCENARIO 1: User changes jobs
============================================================

Old context: Works as a data analyst at TechCorp (conf=0.9, 180 days ago)
New context: Works as a senior engineer at StartupXYZ (conf=0.85, 2 days ago)

  [CONCAT (original)]
    content:      'Works as a senior engineer at StartupXYZ [MERGED: Works as a data analyst at TechCorp]'
    confidence:   0.700
    needs_review: True
    superseded:   2 unit(s)

  [RECENCY_WEIGHTED]
    content:      'Works as a senior engineer at StartupXYZ'
    confidence:   1.000
    needs_review: False
    superseded:   2 unit(s)

  [LLM_SUMMARIZED]
    content:      'Previously: Works as a data analyst at TechCorp. Currently: Works as a senior engineer at StartupXYZ'
    confidence:   0.850
    needs_review: False
    superseded:   2 unit(s)


============================================================
  SCENARIO 2: Three diet changes over a year
============================================================

Old 1: Follows a strict keto diet (conf=0.7, 365 days)
Old 2: Switched to vegetarian eating (conf=0.8, 90 days)
New:   Now eating Mediterranean diet (conf=0.85, 1 day)

  [CONCAT (original)]
    content:      'Now eating Mediterranean diet [MERGED: Follows a strict keto diet] [MERGED: Switched to vegetarian eating]'
    confidence:   0.627
    needs_review: True
    superseded:   3 unit(s)

  [RECENCY_WEIGHTED]
    content:      'Now eating Mediterranean diet'
    confidence:   1.000
    needs_review: False
    superseded:   3 unit(s)

  [LLM_SUMMARIZED]
    content:      'Previously: Follows a strict keto diet, Switched to vegetarian eating. Currently: Now eating Mediterranean diet'
    confidence:   0.850
    needs_review: False
    superseded:   3 unit(s)


============================================================
  SCENARIO 3: LLM merge graceful fallback
============================================================

LLM merge_fn raises RuntimeError...
  [LLM_SUMMARIZED (fallback)]
    content:      'Works as a senior engineer at StartupXYZ'
    confidence:   1.000
    needs_review: False
    superseded:   2 unit(s)

  ^ Gracefully fell back to RECENCY_WEIGHTED!

=== KEY TAKEAWAYS ===

CONCAT (original):
  - Produces gibberish like '[MERGED: ...]' that confuses downstream LLMs
  - Confidence drops arbitrarily (avg * 0.8)
  - Always flags needs_review=True

RECENCY_WEIGHTED (new):
  - Clean content — newest fact wins, no tags
  - Confidence boosted (topic seen before = more trust)
  - No review needed — deterministic resolution

LLM_SUMMARIZED (new):
  - Best quality — LLM produces a clean summary preserving relevant history
  - Gracefully falls back to RECENCY_WEIGHTED if LLM fails
  - Confidence preserved from the new unit
"""
import asyncio
from datetime import datetime, timedelta, timezone

from contextidx.core.context_unit import ContextUnit
from contextidx.core.conflict_resolver import ConflictResolver


def make_unit(content: str, days_ago: int = 0, confidence: float = 0.8) -> ContextUnit:
    return ContextUnit(
        content=content,
        scope={"user_id": "u1"},
        confidence=confidence,
        timestamp=datetime.now(timezone.utc) - timedelta(days=days_ago),
    )


def divider(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def show_result(label: str, result):
    print(f"  [{label}]")
    print(f"    content:      '{result.winner.content}'")
    print(f"    confidence:   {result.winner.confidence:.3f}")
    print(f"    needs_review: {result.needs_review}")
    print(f"    superseded:   {len(result.superseded)} unit(s)")
    print()


async def main():
    # ── Scenario 1: User changes jobs ──────────────────────────────────

    divider("SCENARIO 1: User changes jobs")

    old_job = make_unit("Works as a data analyst at TechCorp", days_ago=180, confidence=0.9)
    new_job = make_unit("Works as a senior engineer at StartupXYZ", days_ago=2, confidence=0.85)

    print("Old context:", old_job.content, f"(conf={old_job.confidence}, 180 days ago)")
    print("New context:", new_job.content, f"(conf={new_job.confidence}, 2 days ago)")
    print()

    # CONCAT (original behavior)
    r1 = ConflictResolver(strategy="MERGE", merge_strategy="CONCAT")
    show_result("CONCAT (original)", r1.resolve(new_job, [old_job]))

    # RECENCY_WEIGHTED
    r2 = ConflictResolver(strategy="MERGE", merge_strategy="RECENCY_WEIGHTED")
    show_result("RECENCY_WEIGHTED", r2.resolve(new_job, [old_job]))

    # LLM_SUMMARIZED
    async def mock_llm_merge(new_content: str, old_contents: list[str]) -> str:
        # In production this would call an LLM. Here we simulate a clean summary.
        old = ", ".join(old_contents)
        return f"Previously: {old}. Currently: {new_content}"

    r3 = ConflictResolver(
        strategy="MERGE", merge_strategy="LLM_SUMMARIZED", merge_fn=mock_llm_merge,
    )
    show_result("LLM_SUMMARIZED", await r3.aresolve(new_job, [old_job]))

    # ── Scenario 2: Multiple diet changes ──────────────────────────────

    divider("SCENARIO 2: Three diet changes over a year")

    old_diet_1 = make_unit("Follows a strict keto diet", days_ago=365, confidence=0.7)
    old_diet_2 = make_unit("Switched to vegetarian eating", days_ago=90, confidence=0.8)
    new_diet = make_unit("Now eating Mediterranean diet", days_ago=1, confidence=0.85)

    old_diets = [old_diet_1, old_diet_2]

    print("Old 1:", old_diet_1.content, f"(conf={old_diet_1.confidence}, 365 days)")
    print("Old 2:", old_diet_2.content, f"(conf={old_diet_2.confidence}, 90 days)")
    print("New:  ", new_diet.content, f"(conf={new_diet.confidence}, 1 day)")
    print()

    r_concat = ConflictResolver(strategy="MERGE", merge_strategy="CONCAT")
    show_result("CONCAT (original)", r_concat.resolve(new_diet, old_diets))

    r_recency = ConflictResolver(strategy="MERGE", merge_strategy="RECENCY_WEIGHTED", confidence_boost=0.1)
    show_result("RECENCY_WEIGHTED", r_recency.resolve(new_diet, old_diets))

    r_llm = ConflictResolver(
        strategy="MERGE", merge_strategy="LLM_SUMMARIZED", merge_fn=mock_llm_merge,
    )
    show_result("LLM_SUMMARIZED", await r_llm.aresolve(new_diet, old_diets))

    # ── Scenario 3: LLM merge fallback on error ───────────────────────

    divider("SCENARIO 3: LLM merge graceful fallback")

    async def failing_llm(new_content: str, old_contents: list[str]) -> str:
        raise RuntimeError("OpenAI API timeout")

    r_fail = ConflictResolver(
        strategy="MERGE", merge_strategy="LLM_SUMMARIZED", merge_fn=failing_llm,
    )
    print("LLM merge_fn raises RuntimeError...")
    show_result("LLM_SUMMARIZED (fallback)", await r_fail.aresolve(new_job, [old_job]))
    print("  ^ Gracefully fell back to RECENCY_WEIGHTED!")


asyncio.run(main())
