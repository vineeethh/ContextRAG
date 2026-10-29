from __future__ import annotations

from collections.abc import AsyncIterator
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from backend.models import Chunk, ContextWindow, QueryResult
from backend.observers.langfuse_tracing import (
    TracedLLM,
    TracedReranker,
    TracedSemanticCache,
    query_trace,
)
from backend.strategies.base import LLMStrategy


def _fake_client() -> tuple[MagicMock, MagicMock]:
    """Langfuse client whose observations are all the same MagicMock."""
    obs = MagicMock()
    client = MagicMock()
    client.start_as_current_observation.return_value.__enter__.return_value = obs
    client.start_as_current_observation.return_value.__exit__.return_value = False
    return client, obs


class _StubLLM(LLMStrategy):
    async def generate(self, context: ContextWindow) -> QueryResult:
        context.model, context.tokens_in, context.tokens_out = "gpt-4o", 900, 40
        context.cost_usd = Decimal("0.0026")
        return QueryResult(answer="full answer")

    async def generate_stream(self, context: ContextWindow) -> AsyncIterator[str]:
        context.model, context.tokens_in, context.tokens_out = "gpt-4o", 900, 2
        for token in ["Hello", " world"]:
            yield token


class TestQueryTrace:
    def test_noop_when_tracing_disabled(self):
        with query_trace(None, "q", "t1", uuid4()) as trace:
            trace.update(output="anything")  # must not raise

    def test_opens_root_observation(self):
        client, _ = _fake_client()
        with query_trace(client, "what is DI?", "t1", uuid4()):
            pass
        kwargs = client.start_as_current_observation.call_args.kwargs
        assert kwargs["name"] == "rag_query"
        assert kwargs["input"] == {"query": "what is DI?"}


class TestTracedLLM:
    async def test_generate_records_prompt_usage_and_cost(self):
        client, gen = _fake_client()
        ctx = ContextWindow(query="what is DI?")
        result = await TracedLLM(_StubLLM(), client).generate(ctx)

        assert result.answer == "full answer"
        prompt = client.start_as_current_observation.call_args.kwargs["input"]
        assert prompt[-1]["content"].endswith("Question: what is DI?")
        update = gen.update.call_args.kwargs
        assert update["model"] == "gpt-4o"
        assert update["usage_details"] == {"input": 900, "output": 40}
        assert update["cost_details"] == {"total": 0.0026}

    async def test_stream_passes_tokens_through_unchanged(self):
        client, gen = _fake_client()
        llm = TracedLLM(_StubLLM(), client)
        tokens = [t async for t in llm.generate_stream(ContextWindow(query="q"))]

        assert tokens == ["Hello", " world"]
        assert gen.update.call_args.kwargs["output"] == "Hello world"


class TestTracedReranker:
    async def test_scores_trace_with_top_rerank_score(self):
        client, obs = _fake_client()
        top = Chunk(content="relevant", rerank_score=0.91)
        inner = AsyncMock()
        inner.rerank.return_value = [top]

        result = await TracedReranker(inner, client).rerank("q", [top], top_n=1)

        assert result == [top]
        obs.score_trace.assert_called_once_with(name="top_rerank_score", value=0.91)


class TestTracedSemanticCache:
    async def test_records_hit_and_passes_result_through(self):
        client, obs = _fake_client()
        cached = QueryResult(answer="cached", cached=True)
        inner = AsyncMock()
        inner.get.return_value = cached

        result = await TracedSemanticCache(inner, client).get([0.1], "t1")

        assert result is cached
        obs.update.assert_called_once_with(output={"hit": True})
