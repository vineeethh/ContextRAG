from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from pathlib import Path

import pymupdf
import pymupdf4llm

from backend.connectors.base import ConnectorStrategy
from backend.connectors.chunkers.heading_aware_chunker import HeadingAwareChunker
from backend.connectors.chunkers.recursive_chunker import RecursiveChunker
from backend.exceptions import ParseError
from backend.models import Chunk, SourceType
from backend.strategies.base import ChunkerStrategy


class PDFConnector(ConnectorStrategy):
    """Handles tenant-uploaded files: PDF and Markdown.

    Source URL is the local path the upload endpoint saved the file to
    (see LocalFileStorage).

    Routing by extension:
      .pdf  → pymupdf4llm (PDF → Markdown) → RecursiveChunker
      .md   → raw Markdown content          → HeadingAwareChunker
    """

    def __init__(
        self,
        pdf_chunker: ChunkerStrategy | None = None,
        md_chunker: ChunkerStrategy | None = None,
    ) -> None:
        self._pdf_chunker = pdf_chunker or RecursiveChunker()
        self._md_chunker = md_chunker or HeadingAwareChunker()

    @property
    def source_type(self) -> SourceType:
        return SourceType.PDF

    async def fetch_chunks(
        self, source_url: str, tenant_id: str
    ) -> AsyncIterator[Chunk]:
        text, suffix = await self._get_text_and_suffix(source_url)
        chunker = self._md_chunker if suffix == ".md" else self._pdf_chunker
        metadata = {
            "tenant_id": tenant_id,
            "source_url": source_url,
            "source_type": SourceType.PDF.value,
        }
        for chunk in chunker.chunk(text, metadata):
            yield chunk

    async def compute_content_hash(self, source_url: str) -> str:
        return hashlib.sha256(Path(source_url).read_bytes()).hexdigest()

    # ── Private ───────────────────────────────────────────────────────────────

    async def _get_text_and_suffix(self, source_url: str) -> tuple[str, str]:
        suffix = Path(source_url).suffix.lower()
        if suffix == ".md":
            return Path(source_url).read_text(encoding="utf-8"), ".md"
        return _extract_text_from_path(source_url), ".pdf"


# ── PDF extraction helpers ─────────────────────────────────────────────────────

def _extract_text_from_path(path: str) -> str:
    try:
        doc = pymupdf.open(path)
        pages = list(range(len(doc)))
        doc.close()
        md = pymupdf4llm.to_markdown(path, pages=pages, show_progress=False)
        return md.replace("\n-----\n", "\n\n")
    except Exception as exc:
        raise ParseError(f"Failed to extract text from PDF {path}: {exc}") from exc
