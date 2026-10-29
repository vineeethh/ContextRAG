from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from backend.core.coverage_gaps import cluster_gaps
from backend.models import ContextWindow, CoverageGap, QueryResult
from backend.observers.coverage_gap_observer import CoverageGapObserver
from backend.strategies.cache.redis_semantic_cache import _deserialize, _serialize

_NOW = datetime(2026, 10, 1, tzinfo=UTC)


def _gap(query: str, embedding: list[float], minutes_ago: int = 0, score: float = 0.1):
    return CoverageGap(
        tenant_id="t1",
        query=query,
        top_score=score,
        query_embedding=embedding,
        created_at=_NOW - timedelta(minutes=minutes_ago),
    )


# ── cluster_gaps ───────────────────────────────────────────────────────────────

class TestClusterGaps:
    def test_empty_input_returns_no_topics(self):
        assert cluster_gaps([]) == []

    def test_similar_questions_share_a_topic(self):
        gaps = [
            _gap("how do I rotate keys?", [1.0, 0.05, 0.0], minutes_ago=0),
            _gap("rotating api keys", [0.98, 0.1, 0.0], minutes_ago=5),
            _gap("billing limits", [0.0, 0.0, 1.0], minutes_ago=10),
        ]
        topics = cluster_gaps(gaps)
        assert len(topics) == 2
        assert topics[0].count == 2
        assert topics[0].question == "how do I rotate keys?"  # newest phrasing leads
        assert topics[0].examples == ["rotating api keys"]

    def test_topics_sorted_by_count_then_recency(self):
        gaps = [
            _gap("a1", [1.0, 0.0], minutes_ago=0),
            _gap("b1", [0.0, 1.0], minutes_ago=1),
            _gap("b2", [0.0, 1.0], minutes_ago=2),
        ]
        topics = cluster_gaps(gaps)
        assert [t.question for t in topics] == ["b1", "a1"]

    def test_duplicate_phrasings_counted_but_not_repeated_in_examples(self):
        gaps = [
            _gap("What is X?", [1.0, 0.0], minutes_ago=0),
            _gap("what is x?", [1.0, 0.0], minutes_ago=1),
            _gap("What is X?", [1.0, 0.0], minutes_ago=2),
        ]
        topic = cluster_gaps(gaps)[0]
        assert topic.count == 3
        assert topic.examples == []

    def test_avg_score_and_last_asked(self):
        gaps = [
            _gap("q one", [1.0, 0.0], minutes_ago=0, score=0.2),
            _gap("q two", [1.0, 0.0], minutes_ago=3, score=0.0),
        ]
        topic = cluster_gaps(gaps)[0]
        assert topic.avg_score == pytest.approx(0.1)
        assert topic.last_asked == _NOW

    def test_zero_vector_does_not_crash(self):
        topics = cluster_gaps([_gap("empty embedding", [0.0, 0.0])])
        assert len(topics) == 1


# ── CoverageGapObserver ────────────────────────────────────────────────────────

def _ctx(query: str = "how do I configure rate limits?") -> ContextWindow:
    return ContextWindow(query=query, tenant_id="t1", query_embedding=[0.1, 0.2])


class TestCoverageGapObserver:
    async def test_records_low_score_query(self):
        repo = AsyncMock()
        observer = CoverageGapObserver(repo, threshold=0.3)
        await observer.notify(_ctx(), QueryResult(top_retrieval_score=0.12))
        repo.record.assert_awaited_once()
        gap = repo.record.call_args.args[0]
        assert gap.query == "how do I configure rate limits?"
        assert gap.top_score == 0.12

    async def test_ignores_well_covered_query(self):
        repo = AsyncMock()
        await CoverageGapObserver(repo, 0.3).notify(_ctx(), QueryResult(top_retrieval_score=0.8))
        repo.record.assert_not_awaited()

    async def test_nothing_retrieved_counts_as_gap(self):
        repo = AsyncMock()
        await CoverageGapObserver(repo, 0.3).notify(_ctx(), QueryResult(top_retrieval_score=0.0))
        repo.record.assert_awaited_once()

    async def test_cache_hit_with_low_score_counts(self):
        repo = AsyncMock()
        result = QueryResult(top_retrieval_score=0.05, cached=True)
        await CoverageGapObserver(repo, 0.3).notify(_ctx(), result)
        repo.record.assert_awaited_once()

    async def test_unknown_score_is_skipped(self):
        # Cache entries written before scores were stored have no score
        repo = AsyncMock()
        await CoverageGapObserver(repo, 0.3).notify(_ctx(), QueryResult(top_retrieval_score=None))
        repo.record.assert_not_awaited()

    async def test_short_chit_chat_is_skipped(self):
        repo = AsyncMock()
        await CoverageGapObserver(repo, 0.3).notify(
            _ctx("hi there"), QueryResult(top_retrieval_score=0.0)
        )
        repo.record.assert_not_awaited()


# ── score survives the semantic cache ──────────────────────────────────────────

class TestCachedScore:
    def test_round_trip_preserves_top_retrieval_score(self):
        restored = _deserialize(_serialize(QueryResult(answer="a", top_retrieval_score=0.42)))
        assert restored.top_retrieval_score == 0.42
