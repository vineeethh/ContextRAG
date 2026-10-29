from __future__ import annotations

from uuid import UUID

import asyncpg

from backend.models import CoverageGap
from backend.repositories.base import CoverageGapRepository


class PostgresCoverageGapRepository(CoverageGapRepository):
    """Stores coverage gaps in the ``coverage_gaps`` table.

    The query embedding is kept alongside the text so the read path can group
    similar questions into topics without re-embedding anything.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def record(self, gap: CoverageGap) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO coverage_gaps (tenant_id, query, top_score, query_embedding) "
                "VALUES ($1, $2, $3, $4)",
                UUID(gap.tenant_id),
                gap.query,
                gap.top_score,
                gap.query_embedding,
            )

    async def list_recent(
        self, tenant_id: str, days: int = 30, limit: int = 1000
    ) -> list[CoverageGap]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT query, top_score, query_embedding, created_at "
                "FROM coverage_gaps "
                "WHERE tenant_id = $1 "
                "  AND created_at > NOW() - ($2 * INTERVAL '1 day') "
                "ORDER BY created_at DESC "
                "LIMIT $3",
                UUID(tenant_id),
                days,
                limit,
            )
        return [
            CoverageGap(
                tenant_id=tenant_id,
                query=r["query"],
                top_score=r["top_score"],
                query_embedding=list(r["query_embedding"]),
                created_at=r["created_at"],
            )
            for r in rows
        ]
