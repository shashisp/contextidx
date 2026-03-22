"""Demo: CONCAT (default) vs LLM_SUMMARIZED merge strategy.

Run: python examples/demo_merge.py
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
    async def mock_llm_merge(new_content: str, old_contents: list[str]) -> str:
        old = ", ".join(old_contents)
        return f"Previously: {old}. Currently: {new_content}"

    # ── Scenario 1: User changes jobs ──

    divider("SCENARIO 1: User changes jobs")

    old_job = make_unit("Works as a data analyst at TechCorp", days_ago=180, confidence=0.9)
    new_job = make_unit("Works as a senior engineer at StartupXYZ", days_ago=2, confidence=0.85)

    print("Old context:", old_job.content, f"(conf={old_job.confidence}, 180 days ago)")
    print("New context:", new_job.content, f"(conf={new_job.confidence}, 2 days ago)")
    print()

    r1 = ConflictResolver(strategy="MERGE", merge_strategy="CONCAT")
    show_result("CONCAT (default)", r1.resolve(new_job, [old_job]))

    r2 = ConflictResolver(strategy="MERGE", merge_fn=mock_llm_merge)
    show_result("LLM_SUMMARIZED", await r2.aresolve(new_job, [old_job]))

    # ── Scenario 2: Multiple diet changes ──

    divider("SCENARIO 2: Three diet changes over a year")

    old_diet_1 = make_unit("Follows a strict keto diet", days_ago=365, confidence=0.7)
    old_diet_2 = make_unit("Switched to vegetarian eating", days_ago=90, confidence=0.8)
    new_diet = make_unit("Now eating Mediterranean diet", days_ago=1, confidence=0.85)
    old_diets = [old_diet_1, old_diet_2]

    print("Old 1:", old_diet_1.content, f"(conf={old_diet_1.confidence}, 365 days)")
    print("Old 2:", old_diet_2.content, f"(conf={old_diet_2.confidence}, 90 days)")
    print("New:  ", new_diet.content, f"(conf={new_diet.confidence}, 1 day)")
    print()

    r3 = ConflictResolver(strategy="MERGE", merge_strategy="CONCAT")
    show_result("CONCAT (default)", r3.resolve(new_diet, old_diets))

    r4 = ConflictResolver(strategy="MERGE", merge_fn=mock_llm_merge)
    show_result("LLM_SUMMARIZED", await r4.aresolve(new_diet, old_diets))

    # ── Scenario 3: LLM merge graceful fallback ──

    divider("SCENARIO 3: LLM merge graceful fallback")

    async def failing_llm(new_content: str, old_contents: list[str]) -> str:
        raise RuntimeError("OpenAI API timeout")

    r_fail = ConflictResolver(strategy="MERGE", merge_fn=failing_llm)
    print("LLM merge_fn raises RuntimeError...")
    show_result("LLM_SUMMARIZED (fallback)", await r_fail.aresolve(new_job, [old_job]))
    print("  ^ Gracefully fell back to CONCAT")


asyncio.run(main())
