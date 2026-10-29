"""Langfuse tracing for the query path.

Each strategy QueryPipeline depends on is wrapped in a thin decorator that
opens a Langfuse observation around the call. Observations nest through the
OpenTelemetry context, so one /query request becomes one trace:

    rag_query            (chain)       query in, answer out
    ├── embed            (embedding)
    ├── cache_lookup     (span)        hit / miss
    ├── hybrid_search    (retriever)   top-20 candidates
    ├── rerank           (span)        top-5 + trace score "top_rerank_score"
    └── generate         (generation)  full prompt, answer, model, tokens, cost,
                                       time to first token

Traces are grouped by conversation (session_id) and tenant (user_id).

QueryPipeline and the concrete strategies never import Langfuse: tracing is
applied when the query service wires its dependencies, and skipped entirely
when the Langfuse keys aren't configured. Spans are exported from a background
thread, so the request path only pays for in-process span bookkeeping.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from langfuse import Langfuse, propagate_attributes

from backend.config import settings
from backend.models import Chunk, ContextWindow, QueryResult
from backend.strategies.base import (
    EmbeddingStrategy,
    LLMStrategy,
    RerankerStrategy,
    SemanticCacheStrategy,
    VectorDBStrategy,
)
from backend.strategies.llm._prompts import build_messages

_PREVIEW_CHARS = 200


def build_langfuse_client() -> Langfuse | None:
    """Return a Langfuse client, or None when tracing isn't configured."""
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return None
    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        base_url=settings.langfuse_base_url,
        environment=settings.environment,
    )


@contextmanager
def query_trace(
    client: Langfuse | None,
    query: str,
    tenant_id: str,
    conversation_id: UUID,
) -> Iterator[Any]:
    """Open the root observation for one query; yields an object with update()."""
    if client is None:
        yield _NoopObservation()
        return
    with propagate_attributes(user_id=tenant_id, session_id=str(conversation_id)):
        with client.start_as_current_observation(
            name="rag_query", as_type="chain", input={"query": query}
        ) as root:
            yield root


class TracedEmbedding(EmbeddingStrategy):
    def __init__(self, inner: EmbeddingStrategy, client: Langfuse) -> None:
        self._inner = inner
        self._client = client

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        with self._client.start_as_current_observation(
            name="embed",
            as_type="embedding",
            model=settings.openai_embedding_model,
            input=texts,
        ) as obs:
            vectors = await self._inner.embed(texts)
            obs.update(output={"vectors": len(vectors)})
            return vectors


class TracedSemanticCache(SemanticCacheStrategy):
    def __init__(self, inner: SemanticCacheStrategy, client: Langfuse) -> None:
        self._inner = inner
        self._client = client

    async def get(self, embedding: list[float], tenant_id: str) -> QueryResult | None:
        with self._client.start_as_current_observation(name="cache_lookup") as obs:
            result = await self._inner.get(embedding, tenant_id)
            obs.update(output={"hit": result is not None})
            return result

    async def set(
        self,
        embedding: list[float],
        tenant_id: str,
        result: QueryResult,
        ttl_seconds: int,
    ) -> None:
        await self._inner.set(embedding, tenant_id, result, ttl_seconds)


class TracedVectorDB(VectorDBStrategy):
    def __init__(self, inner: VectorDBStrategy, client: Langfuse) -> None:
        self._inner = inner
        self._client = client

    async def hybrid_search(
        self,
        tenant_id: str,
        dense_vector: list[float],
        sparse_indices: list[int],
        sparse_values: list[float],
        top_k: int = 20,
    ) -> list[Chunk]:
        with self._client.start_as_current_observation(
            name="hybrid_search", as_type="retriever", input={"top_k": top_k}
        ) as obs:
            chunks = await self._inner.hybrid_search(
                tenant_id, dense_vector, sparse_indices, sparse_values, top_k
            )
            obs.update(output=[_summarize(c) for c in chunks])
            return chunks

    async def upsert(self, chunks: list[Chunk]) -> None:
        await self._inner.upsert(chunks)

    async def delete_by_filter(self, tenant_id: str, filter_dict: dict[str, str]) -> None:
        await self._inner.delete_by_filter(tenant_id, filter_dict)

    async def collection_exists(self, tenant_id: str) -> bool:
        return await self._inner.collection_exists(tenant_id)

    async def create_collection(self, tenant_id: str) -> None:
        await self._inner.create_collection(tenant_id)


class TracedReranker(RerankerStrategy):
    def __init__(self, inner: RerankerStrategy, client: Langfuse) -> None:
        self._inner = inner
        self._client = client

    async def rerank(self, query: str, chunks: list[Chunk], top_n: int = 5) -> list[Chunk]:
        with self._client.start_as_current_observation(
            name="rerank",
            input={"query": query, "candidates": len(chunks), "top_n": top_n},
        ) as obs:
            reranked = await self._inner.rerank(query, chunks, top_n)
            obs.update(output=[_summarize(c) for c in reranked])
            # Trace-level score: filter Langfuse for poorly-grounded answers
            if reranked and reranked[0].rerank_score is not None:
                obs.score_trace(name="top_rerank_score", value=reranked[0].rerank_score)
            return reranked


class TracedLLM(LLMStrategy):
    def __init__(self, inner: LLMStrategy, client: Langfuse) -> None:
        self._inner = inner
        self._client = client

    async def generate(self, context: ContextWindow) -> QueryResult:
        with self._client.start_as_current_observation(
            name="generate", as_type="generation", input=build_messages(context)
        ) as gen:
            result = await self._inner.generate(context)
            _record_generation(gen, context, result.answer)
            return result

    async def generate_stream(self, context: ContextWindow) -> AsyncIterator[str]:
        with self._client.start_as_current_observation(
            name="generate", as_type="generation", input=build_messages(context)
        ) as gen:
            parts: list[str] = []
            async for token in self._inner.generate_stream(context):
                if not parts:
                    gen.update(completion_start_time=datetime.now(UTC))
                parts.append(token)
                yield token
            _record_generation(gen, context, "".join(parts))


class _NoopObservation:
    def update(self, **kwargs: Any) -> None:
        pass


def _record_generation(gen: Any, context: ContextWindow, answer: str) -> None:
    """Attach model, usage and cost — set on the context by the LLM strategy."""
    details: dict[str, Any] = {"output": answer}
    if context.model:
        details["model"] = context.model
    if context.tokens_in or context.tokens_out:
        details["usage_details"] = {"input": context.tokens_in, "output": context.tokens_out}
        details["cost_details"] = {"total": float(context.cost_usd)}
    gen.update(**details)


def _summarize(chunk: Chunk) -> dict[str, Any]:
    return {
        "source_url": chunk.source_url,
        "score": chunk.rerank_score,
        "preview": chunk.content[:_PREVIEW_CHARS],
    }
