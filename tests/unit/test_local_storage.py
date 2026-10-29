from __future__ import annotations

from pathlib import Path

from backend.strategies.storage.local_storage import LocalFileStorage


class TestLocalFileStorage:
    async def test_save_writes_file_under_tenant_dir(self, tmp_path: Path):
        storage = LocalFileStorage(tmp_path)
        path = await storage.save(b"%PDF-1.4", "tenant-a", "My Report.pdf")

        saved = Path(path)
        assert saved.read_bytes() == b"%PDF-1.4"
        assert saved.parent == tmp_path / "tenant-a"
        assert saved.suffix == ".pdf"

    async def test_filename_is_sanitized(self, tmp_path: Path):
        storage = LocalFileStorage(tmp_path)
        path = await storage.save(b"x", "tenant-a", "../../etc/pass wd.md")

        saved = Path(path)
        assert saved.parent == tmp_path / "tenant-a"
        assert saved.name.startswith("pass_wd_")

    async def test_owns_only_own_tenant_files(self, tmp_path: Path):
        storage = LocalFileStorage(tmp_path)
        path = await storage.save(b"x", "tenant-a", "doc.md")

        assert storage.owns("tenant-a", path)
        assert not storage.owns("tenant-b", path)
        assert not storage.owns("tenant-a", str(tmp_path / "tenant-a" / ".." / "tenant-b" / "f"))
        assert not storage.owns("tenant-a", "https://docs.example.com/page")

    async def test_delete_removes_file_and_tolerates_missing(self, tmp_path: Path):
        storage = LocalFileStorage(tmp_path)
        path = await storage.save(b"x", "tenant-a", "doc.md")

        await storage.delete(path)
        assert not Path(path).exists()
        await storage.delete(path)  # second delete is a no-op
