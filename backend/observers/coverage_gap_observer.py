from __future__ import annotations

from backend.models import ContextWindow, CoverageGap, QueryResult
from backend.observers.base import QueryObserver
from backend.repositories.base import CoverageGapRepository

# Greetings and chit-chat ("hi", "thanks!") always score low — they aren't gaps.
_MIN_QUERY_WORDS = 3


class CoverageGapObserver(QueryObserver):
    """Records questions the knowledge base couldn't answer well.

    A query is a gap when the top Cohere rerank score is below `threshold`
    (0.0 when nothing was retrieved). Cache hits count too — the score is
    stored with the cached result — so a gap asked 50 times shows up 50 times.

    Reads scores the pipeline already computed and runs after the response is
    sent, so it adds no latency and never touches the prompt.
    """

    def __init__(self, repo: CoverageGapRepository, threshold: float) -> None:
        self._repo = repo
        self._threshold = threshold

    async def notify(self, context: ContextWindow, result: QueryResult) -> None:
        score = result.top_retrieval_score
        if score is None or score >= self._threshold:
            return
        if len(context.query.split()) < _MIN_QUERY_WORDS or not context.query_embedding:
            return

        await self._repo.record(
            CoverageGap(
                tenant_id=context.tenant_id,
                query=context.query,
                top_score=score,
                query_embedding=context.query_embedding,
            )
        )
