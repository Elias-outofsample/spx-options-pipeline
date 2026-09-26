from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.feather as feather

from .constants import DEFAULT_CACHE_SIZE_MB
from .params import QueryParams

logger = logging.getLogger(__name__)

_CACHE_META_FILE = "_cache_meta.json"


class FeatherCache:
    """
    Disk-based cache using uncompressed Feather (Arrow IPC) format.

    Uncompressed is required for memory_map=True to work as true zero-copy.
    ZSTD Feather requires decompression into RAM, defeating the purpose.

    Cache invalidation:
    1. params.cache_key() doesn't match any existing entry
    2. Manual invalidate(dataset) or invalidate()
    3. LRU eviction when total size exceeds max_cache_size_mb
    """

    def __init__(self, cache_dir: Path | str, max_cache_size_mb: int = DEFAULT_CACHE_SIZE_MB):
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._max_size_mb = max_cache_size_mb
        self._meta = self._load_meta()

    def get(self, params: QueryParams) -> pa.Table | None:
        """Return cached Arrow table if available, else None."""
        key = params.cache_key()
        if key not in self._meta:
            return None

        entry = self._meta[key]
        cache_path = self._cache_dir / entry["file"]

        if not cache_path.exists():
            del self._meta[key]
            self._save_meta()
            return None

        logger.debug("Cache HIT for %s [%s]", params.dataset, key)
        entry["accessed_at"] = time.time()
        self._save_meta()
        return feather.read_table(str(cache_path), memory_map=True)

    def put(self, params: QueryParams, table: pa.Table) -> Path:
        """Write Arrow table to cache. Returns cache file path."""
        key = params.cache_key()
        rel_path = f"{params.dataset}/{key}.feather"
        cache_path = self._cache_dir / rel_path
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        feather.write_feather(table, str(cache_path), compression="uncompressed")

        self._meta[key] = {
            "file": rel_path,
            "created_at": time.time(),
            "dataset": params.dataset,
            "start_date": str(params.start_date),
            "end_date": str(params.end_date),
            "size_bytes": cache_path.stat().st_size,
        }
        self._save_meta()
        self._enforce_size_limit()

        return cache_path

    def invalidate(self, dataset: str | None = None) -> int:
        """
        Clear cache for a dataset, or all caches if dataset is None.
        Returns the number of entries removed.
        """
        to_remove = []
        for key, entry in self._meta.items():
            if dataset is None or entry.get("dataset") == dataset:
                to_remove.append(key)
                path = self._cache_dir / entry["file"]
                if path.exists():
                    path.unlink()

        for key in to_remove:
            del self._meta[key]
        self._save_meta()
        return len(to_remove)

    def _enforce_size_limit(self) -> None:
        total = sum(e.get("size_bytes", 0) for e in self._meta.values())
        limit = self._max_size_mb * 1024 * 1024

        if total <= limit:
            return

        entries = sorted(
            self._meta.items(),
            key=lambda x: x[1].get("accessed_at", x[1].get("created_at", 0)),
        )
        while total > limit and entries:
            key, entry = entries.pop(0)
            path = self._cache_dir / entry["file"]
            if path.exists():
                total -= entry.get("size_bytes", 0)
                path.unlink()
            del self._meta[key]

        self._save_meta()

    def _load_meta(self) -> dict:
        meta_path = self._cache_dir / _CACHE_META_FILE
        if meta_path.exists():
            with open(meta_path) as f:
                return json.load(f)
        return {}

    def _save_meta(self) -> None:
        meta_path = self._cache_dir / _CACHE_META_FILE
        with open(meta_path, "w") as f:
            json.dump(self._meta, f, indent=2)
