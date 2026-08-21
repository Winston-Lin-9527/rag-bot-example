"""Keyed blob storage.

Deliberately dumb: it moves bytes to and from a key. It knows nothing about
hashes, documents, or metadata — deciding *what* a key should be is a naming
policy that lives in services/documents.py, and metadata lives in the document
records. That keeps this protocol small enough that swapping LocalObjectStore
for a MinIO/S3 implementation is a one-class change: put(key, src) maps directly
onto fput_object(bucket, key, path).
"""

from __future__ import annotations

import os
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import BinaryIO, Protocol

from config.settings import DATA_DIR


class ObjectStore(Protocol):
    def put(self, key: str, src: Path) -> None:
        """Store the bytes at src under key, replacing anything already there."""

    def open(self, key: str) -> BinaryIO:
        """Open key for streaming reads."""

    def local_path(self, key: str) -> Path:
        """A real filesystem path for key.

        PyMuPDF and PaddleOCR both need a path rather than a file object, so
        every backend has to be able to produce one. A remote implementation
        would download to a cache directory here.
        """

    def exists(self, key: str) -> bool:
        ...

    def delete(self, key: str) -> None:
        """Remove key. A missing key is not an error."""


class LocalObjectStore:
    """Stores blobs as files under `root`."""

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

    def _resolve(self, key: str) -> Path:
        # Keys come from callers, so treat them as untrusted: reject anything
        # that escapes the root once normalised ("../../etc/passwd", absolute
        # paths, symlink games).
        if not key or key.startswith("/") or "\\" in key:
            raise ValueError(f"Invalid storage key: {key!r}")

        candidate = (self._root / key).resolve()
        if candidate != self._root and self._root not in candidate.parents:
            raise ValueError(f"Storage key escapes root: {key!r}")
        return candidate

    def put(self, key: str, src: Path) -> None:
        dest = self._resolve(key)
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Copy into the destination directory first, then os.replace onto the
        # final name. Same filesystem, so the rename is atomic and a crash
        # mid-write can never leave a truncated blob at a valid key.
        fd, staging = tempfile.mkstemp(dir=dest.parent, suffix=".part")
        os.close(fd)
        try:
            shutil.copyfile(src, staging)
            os.replace(staging, dest)
        except BaseException:
            Path(staging).unlink(missing_ok=True)
            raise

    def open(self, key: str) -> BinaryIO:
        return self._resolve(key).open("rb")

    def local_path(self, key: str) -> Path:
        return self._resolve(key)

    def exists(self, key: str) -> bool:
        try:
            return self._resolve(key).is_file()
        except ValueError:
            return False

    def delete(self, key: str) -> None:
        self._resolve(key).unlink(missing_ok=True)


# class MinioObjectStore: to be implemented if we want to use MinIO/S3 for blob storage. 
# The interface is the same as LocalObjectStore, 
# but the implementation would use the MinIO/S3 API to store and retrieve objects.

@lru_cache(maxsize=1)
def get_object_store() -> ObjectStore:
    """Process-wide store, mirroring services/chroma.get_chroma_client()."""
    return LocalObjectStore(Path(DATA_DIR) / "blobs")
