"""LangChain integration for contextidx.

Install extras:
    pip install contextidx[langchain]

Usage::

    from contextidx import ContextIdx
    from contextidx.backends.pgvector import PGVectorBackend
    from contextidx.integrations.langchain import ContextIdxMemory
    from langchain.chains import ConversationChain
    from langchain_openai import ChatOpenAI

    async def main():
        ctx = ContextIdx(backend=PGVectorBackend(...))
        await ctx.ainitialize()

        memory = ContextIdxMemory(ctx=ctx, scope={"session": "abc123"})
        chain = ConversationChain(llm=ChatOpenAI(), memory=memory)
        response = await chain.arun("What did I say about the project timeline?")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import Field

try:
    from langchain.memory.chat_memory import BaseChatMemory
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.outputs import LLMResult
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "LangChain is not installed. Run: pip install contextidx[langchain]"
    ) from exc

if TYPE_CHECKING:
    from contextidx.contextidx import ContextIdx


class ContextIdxMemory(BaseChatMemory):
    """LangChain ``BaseChatMemory`` backed by contextidx temporal storage.

    Stores every human/AI turn as a ``ContextUnit`` and retrieves the most
    relevant past context on each ``load_memory_variables()`` call.

    Parameters
    ----------
    ctx:
        An initialised ``ContextIdx`` instance.
    scope:
        Scope dict used to isolate memory per session/user/thread.
        Example: ``{"session": "abc123", "user": "u_42"}``.
    memory_key:
        The key under which retrieved context is returned to the chain.
        Defaults to ``"history"``.
    top_k:
        Number of context units to retrieve for each query.
    human_prefix:
        Prefix applied to human messages when stored as content.
    ai_prefix:
        Prefix applied to AI messages when stored as content.
    """

    ctx: Any = Field(...)  # ContextIdx — typed as Any to avoid circular import
    scope: dict[str, str] = Field(default_factory=dict)
    memory_key: str = "history"
    top_k: int = 5
    human_prefix: str = "Human"
    ai_prefix: str = "AI"

    class Config:
        arbitrary_types_allowed = True

    @property
    def memory_variables(self) -> list[str]:
        return [self.memory_key]

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Synchronous wrapper — prefer ``aload_memory_variables`` in async chains."""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            # Running inside an existing event loop (e.g. Jupyter) — cannot use run()
            raise RuntimeError(
                "ContextIdxMemory.load_memory_variables() cannot be called from an async "
                "context. Use `await memory.aload_memory_variables(inputs)` instead."
            )
        except RuntimeError as exc:
            if "cannot be called" in str(exc):
                raise
            return asyncio.run(self.aload_memory_variables(inputs))

    async def aload_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Retrieve relevant past context units for the current query."""
        query = inputs.get("input") or inputs.get("question") or ""
        if not query:
            return {self.memory_key: []}

        units = await self.ctx.aretrieve(query, top_k=self.top_k, scope=self.scope)
        messages = []
        for unit in units:
            # Re-hydrate as the appropriate message type based on stored prefix
            if unit.content.startswith(f"{self.human_prefix}: "):
                messages.append(HumanMessage(content=unit.content[len(self.human_prefix) + 2:]))
            elif unit.content.startswith(f"{self.ai_prefix}: "):
                messages.append(AIMessage(content=unit.content[len(self.ai_prefix) + 2:]))
            else:
                messages.append(HumanMessage(content=unit.content))

        return {self.memory_key: messages}

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """Synchronous wrapper — prefer ``asave_context`` in async chains."""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            raise RuntimeError(
                "ContextIdxMemory.save_context() cannot be called from an async context. "
                "Use `await memory.asave_context(inputs, outputs)` instead."
            )
        except RuntimeError as exc:
            if "cannot be called" in str(exc):
                raise
            asyncio.run(self.asave_context(inputs, outputs))

    async def asave_context(
        self, inputs: dict[str, Any], outputs: dict[str, str]
    ) -> None:
        """Store the human input and AI response as separate context units."""
        human_text = inputs.get("input") or inputs.get("question") or ""
        ai_text = outputs.get("output") or outputs.get("answer") or ""

        store_ops = []
        if human_text:
            store_ops.append(
                self.ctx.astore(
                    f"{self.human_prefix}: {human_text}",
                    scope=self.scope,
                    metadata={"role": "human"},
                )
            )
        if ai_text:
            store_ops.append(
                self.ctx.astore(
                    f"{self.ai_prefix}: {ai_text}",
                    scope=self.scope,
                    metadata={"role": "ai"},
                )
            )

        import asyncio
        await asyncio.gather(*store_ops)

    async def aclear(self) -> None:
        """Delete all stored context units for this memory's scope."""
        await self.ctx.aclear(scope=self.scope)

    def clear(self) -> None:
        """Synchronous clear — prefer ``aclear`` in async contexts."""
        import asyncio

        try:
            asyncio.get_running_loop()
            raise RuntimeError(
                "ContextIdxMemory.clear() cannot be called from an async context. "
                "Use `await memory.aclear()` instead."
            )
        except RuntimeError as exc:
            if "cannot be called" in str(exc):
                raise
            asyncio.run(self.aclear())

    # LangChain v0.2+ callback hooks (optional, no-ops are fine)
    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:  # noqa: ANN401
        pass
