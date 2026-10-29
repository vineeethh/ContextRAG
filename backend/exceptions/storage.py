from __future__ import annotations

from backend.exceptions.base import ErrorCode, KapaError


class StorageError(KapaError):
    """Any failure originating from the file storage layer."""
    component = "storage"


class FileSaveError(StorageError):
    """Failed to write an uploaded file to the upload directory.

    Not retryable — a full disk or a permissions problem won't fix itself.
    """
    error_code = ErrorCode.FILE_SAVE_FAILED
    retryable = False


class FileDeleteError(StorageError):
    """Failed to delete an uploaded file.

    Not retryable — a missing file is treated as already deleted;
    a permissions problem won't go away on retry.
    """
    error_code = ErrorCode.FILE_DELETE_FAILED
    retryable = False
