"""Groups coverage gaps into topics for the docs team.

Leader clustering on the query embeddings the pipeline already computed:
walk gaps newest-first; each one joins the first topic whose leader is at
least `similarity` cosine-similar, otherwise it starts a new topic. Gaps are
capped per request (see CoverageGapRepository.list_recent), so this is a
single small matrix-vector product per gap — no extra API calls.

0.80 is a starting point, looser than the semantic cache's 0.90 because a
topic should absorb paraphrases and related asks, not just near-duplicate
questions. Tune it on real traffic if topics merge or split too eagerly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from backend.models import CoverageGap

_TOPIC_SIMILARITY = 0.80
_MAX_EXAMPLES = 4


@dataclass
class GapTopic:
    question: str        # most recent phrasing — the topic's leader
    count: int           # times asked, cache hits included
    examples: list[str]  # other distinct phrasings
    avg_score: float     # mean top rerank score — lower means less covered
    last_asked: datetime


def cluster_gaps(
    gaps: list[CoverageGap],
    similarity: float = _TOPIC_SIMILARITY,
) -> list[GapTopic]:
    """Return topics sorted by how often they're asked, then by recency.

    `gaps` must be ordered newest first.
    """
    if not gaps:
        return []

    vectors = np.array([g.query_embedding for g in gaps], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.where(norms == 0, 1.0, norms)

    leaders: list[int] = []
    members: list[list[int]] = []
    for i in range(len(gaps)):
        if leaders:
            sims = vectors[leaders] @ vectors[i]
            best = int(np.argmax(sims))
            if sims[best] >= similarity:
                members[best].append(i)
                continue
        leaders.append(i)
        members.append([i])

    topics = [_to_topic([gaps[i] for i in idxs]) for idxs in members]
    topics.sort(key=lambda t: (t.count, t.last_asked), reverse=True)
    return topics


def _to_topic(rows: list[CoverageGap]) -> GapTopic:
    leader = rows[0]
    seen = {leader.query.strip().lower()}
    examples: list[str] = []
    for row in rows[1:]:
        key = row.query.strip().lower()
        if key not in seen and len(examples) < _MAX_EXAMPLES:
            seen.add(key)
            examples.append(row.query)

    return GapTopic(
        question=leader.query,
        count=len(rows),
        examples=examples,
        avg_score=round(sum(r.top_score for r in rows) / len(rows), 3),
        last_asked=leader.created_at,
    )
