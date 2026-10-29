from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import UUID

from backend.models import CoverageGap, IngestionJob, IngestionStatus, Turn


class IngestionJobRepository(ABC):
    """Tracks ingestion job lifecycle.

    Status transitions (PENDING → PROCESSING → COMPLETED/FAILED) must be
    atomic — that's why this lives in PostgreSQL, not Redis.
    """

    @abstractmethod
    async def create(self, job: IngestionJob) -> IngestionJob:
        ...

    @abstractmethod
    async def get(self, job_id: UUID) -> IngestionJob | None:
        ...

    @abstractmethod
    async def update_status(
        self,
        job_id: UUID,
        status: IngestionStatus,
        error_message: str | None = None,
    ) -> None:
        ...

    @abstractmethod
    async def increment_processed(self, job_id: UUID, count: int = 1) -> None:
        ...


class ConversationRepository(ABC):
    """Persists conversation turns for multi-turn context."""

    @abstractmethod
    async def save_turn(self, turn: Turn) -> Turn:
        ...

    @abstractmethod
    async def get_recent_turns(
        self, conversation_id: UUID, limit: int = 3
    ) -> list[Turn]:
        """Return the last `limit` turns, oldest first."""
        ...

    @abstractmethod
    async def delete_conversation(self, conversation_id: UUID) -> None:
        ...


class SourceHashRepository(ABC):
    """Registry of ingested sources: last-seen content hash per (tenant_id, source_url).

    list_by_tenant() feeds the tenant-aware system prompt and the /sources
    listing; delete() keeps both in sync when a source is removed.
    """

    @abstractmethod
    async def get(self, tenant_id: str, source_url: str) -> str | None:
        """Return stored hash or None if this source has never been indexed."""
        ...

    @abstractmethod
    async def upsert(
        self, tenant_id: str, source_url: str, content_hash: str, source_type: str = "unknown"
    ) -> None:
        """Insert or update the hash for a (tenant_id, source_url) pair."""
        ...

    @abstractmethod
    async def delete(self, tenant_id: str, source_url: str) -> None:
        """Remove the hash record when a source is purged."""
        ...

    @abstractmethod
    async def list_by_tenant(self, tenant_id: str) -> list[dict]:
        """Return all source records for a tenant as dicts with source_url and source_type."""
        ...


class CoverageGapRepository(ABC):
    """Persists questions the knowledge base couldn't answer."""

    @abstractmethod
    async def record(self, gap: CoverageGap) -> None:
        ...

    @abstractmethod
    async def list_recent(
        self, tenant_id: str, days: int = 30, limit: int = 1000
    ) -> list[CoverageGap]:
        """Return the tenant's gaps from the last `days` days, newest first."""
        ...


class TenantRepository(ABC):
    """Source of truth for tenant identity and API key validation."""

    @abstractmethod
    async def get_tenant_id_by_api_key(self, api_key: str) -> str | None:
        """Return tenant_id or None if the key is invalid/revoked."""
        ...

    @abstractmethod
    async def tenant_exists(self, tenant_id: str) -> bool:
        ...









