from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock
from uuid import uuid4

from backend.core.query_pipeline import QueryPipeline
from backend.models import Chunk, ContextWindow, QueryResult
from backend.observers.base import QueryObserver
from backend.strategies.base import LLMStrategy

_SOURCES = [{"source_url": "https://fastapi.tiangolo.com", "source_type": "docs_site"}]


class _RecordingLLM(LLMStrategy):
    """Captures the ContextWindow the pipeline hands to the LLM."""

    def __init__(self) -> None:
        self.contexts: list[ContextWindow] = []

    async def generate(self, context: ContextWindow) -> QueryResult:
        self.contexts.append(context)
        return QueryResult(answer="DI is a pattern.", source_chunks=context.chunks)

    async def generate_stream(self, context: ContextWindow) -> AsyncIterator[str]:
        self.contexts.append(context)
        for token in ["DI ", "is ", "a pattern."]:
            yield token


class _RecordingObserver(QueryObserver):
    def __init__(self) -> None:
        self.calls: list[tuple[ContextWindow, QueryResult]] = []

    async def notify(self, context: ContextWindow, result: QueryResult) -> None:
        self.calls.append((context, result))


def _build(
    *,
    cached: QueryResult | None = None,
    rerank_score: float | None = 0.42,
) -> tuple[QueryPipeline, _RecordingLLM, _RecordingObserver, AsyncMock]:
    chunk = Chunk(tenant_id="t1", source_url="https://fastapi.tiangolo.com/di", content="DI docs")
    chunk.rerank_score = rerank_score

    embedder = AsyncMock()
    embedder.embed.return_value = [[0.1, 0.2, 0.3]]
    sparse = AsyncMock()
    sparse.encode.return_value = [([1, 2], [0.5, 0.5])]
    vector_db = AsyncMock()
    vector_db.hybrid_search.return_value = [chunk]
    reranker = AsyncMock()
    reranker.rerank.return_value = [chunk] if rerank_score is not None else []
    cache = AsyncMock()
    cache.get.return_value = cached
    conversations = AsyncMock()
    conversations.get_recent_turns.return_value = []
    sources = AsyncMock()
    sources.list_by_tenant.return_value = _SOURCES

    llm, observer = _RecordingLLM(), _RecordingObserver()
    pipeline = QueryPipeline(
        llm=llm,
        embedder=embedder,
        sparse_encoder=sparse,
        vector_db=vector_db,
        reranker=reranker,
        cache=cache,
        conversation_repo=conversations,
        observers=[observer],
        source_hash_repo=sources,
    )
    return pipeline, llm, observer, cache


async def _drain_background_tasks() -> None:
    for _ in range(3):
        await asyncio.sleep(0)


class TestHandle:
    async def test_llm_receives_reranked_chunks_and_tenant_sources(self):
        pipeline, llm, _, _ = _build()
        await pipeline.handle("What is DI?", "t1", uuid4())

        ctx = llm.contexts[0]
        assert [c.content for c in ctx.chunks] == ["DI docs"]
        assert ctx.tenant_sources == _SOURCES  # feeds the system prompt

    async def test_result_carries_top_rerank_score(self):
        pipeline, _, observer, cache = _build(rerank_score=0.42)
        result = await pipeline.handle("What is DI?", "t1", uuid4())
        await _drain_background_tasks()

        assert result.top_retrieval_score == 0.42
        cache.set.assert_awaited_once()
        assert observer.calls[0][1].top_retrieval_score == 0.42

    async def test_nothing_reranked_scores_zero(self):
        pipeline, _, _, _ = _build(rerank_score=None)
        result = await pipeline.handle("What is DI?", "t1", uuid4())
        assert result.top_retrieval_score == 0.0

    async def test_cache_hit_skips_retrieval_and_llm(self):
        cached = QueryResult(answer="cached answer", cached=True, top_retrieval_score=0.1)
        pipeline, llm, observer, _ = _build(cached=cached)

        result = await pipeline.handle("What is DI?", "t1", uuid4())
        await _drain_background_tasks()

        assert result is cached
        assert llm.contexts == []
        assert observer.calls[0][1].top_retrieval_score == 0.1


class TestHandleStream:
    async def test_streams_tokens_and_records_score(self):
        pipeline, _, observer, cache = _build(rerank_score=0.42)
        tokens = [t async for t in pipeline.handle_stream("What is DI?", "t1", uuid4())]
        await _drain_background_tasks()

        assert "".join(tokens) == "DI is a pattern."
        result = observer.calls[0][1]
        assert result.answer == "DI is a pattern."
        assert result.top_retrieval_score == 0.42
        cache.set.assert_awaited_once()

    async def test_cache_hit_yields_cached_answer(self):
        cached = QueryResult(answer="cached answer", cached=True)
        pipeline, llm, _, _ = _build(cached=cached)

        tokens = [t async for t in pipeline.handle_stream("What is DI?", "t1", uuid4())]

        assert tokens == ["cached answer"]
        assert llm.contexts == []
