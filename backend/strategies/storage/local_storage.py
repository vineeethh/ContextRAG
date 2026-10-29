from __future__ import annotations

import asyncio
import logging
import re
import uuid
from pathlib import Path

from backend.config import settings
from backend.exceptions import FileDeleteError, FileSaveError

logger = logging.getLogger(__name__)

ALLOWED_SUFFIXES = {".pdf", ".md"}
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


class LocalFileStorage:
    """Stores tenant-uploaded files in a local directory.

    In Docker the directory is a named volume mounted into both the ingestion
    service (writes uploads) and the Celery worker (reads them), so the saved
    path is all an ingestion job needs to find its file.

    Layout:  <upload_dir>/<tenant_id>/<stem>_<uuid><suffix>
    One directory per tenant keeps uploads isolated; owns() lets callers check
    that a client-supplied path belongs to the tenant before touching it.
    """

    def __init__(self, root: str | Path = settings.upload_dir) -> None:
        self._root = Path(root).resolve()

    async def save(self, data: bytes, tenant_id: str, filename: str) -> str:
        """Write bytes and return the absolute path, which becomes the source_url."""
        path = self._new_path(tenant_id, filename)
        try:
            await asyncio.to_thread(_write, path, data)
        except OSError as exc:
            raise FileSaveError(str(exc)) from exc
        logger.info("storage: saved %s → %s", filename, path)
        return str(path)

    def owns(self, tenant_id: str, path: str) -> bool:
        """True if path resolves to a file inside this tenant's upload directory."""
        return Path(path).resolve().is_relative_to(self._root / tenant_id)

    async def delete(self, path: str) -> None:
        """Delete a stored file. A missing file counts as already deleted."""
        try:
            await asyncio.to_thread(Path(path).unlink, missing_ok=True)
        except OSError as exc:
            raise FileDeleteError(str(exc)) from exc
        logger.info("storage: deleted %s", path)

    def _new_path(self, tenant_id: str, filename: str) -> Path:
        name = Path(filename).name
        stem = _UNSAFE_CHARS.sub("_", Path(name).stem)[:80] or "upload"
        suffix = Path(name).suffix.lower()
        return self._root / tenant_id / f"{stem}_{uuid.uuid4().hex}{suffix}"


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
