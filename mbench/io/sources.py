# Copyright 2026 MBench community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Zhang Zhang / Tsinghua University] in [2026].

"""Source locators for local directories and HuggingFace dataset URLs.

- ``SourceLocator`` is the abstract boundary used by adapters.
- ``LocalSourceLocator`` reads from a regular filesystem directory.
- ``HfSourceLocator`` reads small text files from a HuggingFace dataset repo
  via ``huggingface_hub``; large binaries are returned as ``hf://`` URLs and
  only downloaded on demand by ``materialize``.
- ``locator_for(source)`` picks the right concrete locator from a string.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


HF_PREFIX = "hf://datasets/"


class SourceLocator(ABC):
    """Abstract data-source root.

    All paths are *relative* to the dataset root the locator was opened on.
    """

    @abstractmethod
    def read_text(self, relpath: str) -> str:
        """Return the UTF-8 contents of a small text file."""

    @abstractmethod
    def list_dir(self, relpath: str) -> list[str]:
        """List immediate children (file or directory names, no path prefix)."""

    @abstractmethod
    def exists(self, relpath: str) -> bool:
        """Return True if the relative path exists in the source."""

    @abstractmethod
    def resolve(self, relpath: str) -> str:
        """Return a stable string path that an EvaluationItem can hold.

        For local sources this is an absolute filesystem path.
        For HF sources this is an ``hf://datasets/{repo}/{relpath}`` URL.
        Either form can be passed back to ``materialize`` later.
        """

    @abstractmethod
    def materialize(self, relpath: str) -> str:
        """Ensure the file is locally readable and return its absolute path.

        Local sources just return the absolute path. HF sources call
        ``hf_hub_download`` (cached). Adapters should NOT call this — it is for
        metric plugins that need to actually open large media or artifacts.
        """


class LocalSourceLocator(SourceLocator):
    def __init__(self, root: str | Path) -> None:
        root_path = Path(root).expanduser().resolve()
        if not root_path.is_dir():
            raise FileNotFoundError(f"Local dataset root not found: {root_path}")
        self.root = root_path

    def _abs(self, relpath: str) -> Path:
        relpath = relpath.lstrip("/")
        return (self.root / relpath).resolve()

    def read_text(self, relpath: str) -> str:
        return self._abs(relpath).read_text(encoding="utf-8")

    def list_dir(self, relpath: str) -> list[str]:
        target = self._abs(relpath) if relpath else self.root
        if not target.is_dir():
            return []
        return sorted(p.name for p in target.iterdir())

    def exists(self, relpath: str) -> bool:
        return self._abs(relpath).exists()

    def resolve(self, relpath: str) -> str:
        return str(self._abs(relpath))

    def materialize(self, relpath: str) -> str:
        path = self._abs(relpath)
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist in local source: {path}")
        return str(path)

    def __repr__(self) -> str:  # pragma: no cover - debug only
        return f"LocalSourceLocator(root={self.root!r})"


class HfSourceLocator(SourceLocator):
    """Reads a HuggingFace dataset repo, scoped to a sub-prefix.

    A locator opened on ``hf://datasets/foo/bar/MBench-A-New`` answers
    relative queries inside ``MBench-A-New/``.
    """

    def __init__(self, repo_id: str, subpath: str = "", revision: str | None = None) -> None:
        try:
            from huggingface_hub import HfApi  # noqa: F401  # used in helpers
        except ImportError as e:
            raise ImportError(
                "Reading hf:// sources requires the 'huggingface_hub' package. "
                "Install it via: pip install -e .[hf]"
            ) from e

        self.repo_id = repo_id
        self.subpath = subpath.strip("/")
        self.revision = revision
        self._listing_cache: dict[str, list[str]] = {}
        self._all_files: list[str] | None = None
        self._snapshot_root: str | None = None
        self._prefetched: bool = False

    # --- internal helpers ---

    def _join(self, relpath: str) -> str:
        relpath = relpath.lstrip("/")
        if self.subpath and relpath:
            return f"{self.subpath}/{relpath}"
        return self.subpath or relpath

    def _ensure_file_listing(self) -> list[str]:
        if self._all_files is not None:
            return self._all_files
        from huggingface_hub import HfApi

        api = HfApi()
        files = api.list_repo_files(
            repo_id=self.repo_id,
            repo_type="dataset",
            revision=self.revision,
        )
        self._all_files = list(files)
        return self._all_files

    # --- batch metadata prefetch ---

    # Patterns considered "metadata": small text files the adapter needs to
    # join EvaluationItems together. Large media (videos, npz) are excluded.
    _METADATA_PATTERNS: tuple[str, ...] = (
        "dataset.yaml",
        "samples/**/sample.json",
        "models/*/model.yaml",
        "models/*/samples.jsonl",
    )

    def prefetch_metadata(self, patterns: list[str] | None = None) -> str:
        """Download all small adapter-relevant files in one shot.

        Turns hundreds of serial ``read_text`` round-trips into one
        ``snapshot_download`` call (which uses internal parallelism).
        Subsequent ``read_text`` calls are served from the local snapshot.
        """
        if self._prefetched:
            return self._snapshot_root or ""

        from huggingface_hub import snapshot_download

        # Patterns are scoped to the locator's subpath so we don't download
        # sibling subprojects (e.g. when reading MBench-A-New we don't want
        # MBench-V-new files).
        scope = self.subpath
        scoped_patterns: list[str] = []
        for p in (patterns or self._METADATA_PATTERNS):
            scoped_patterns.append(f"{scope}/{p}" if scope else p)

        local_root = snapshot_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            revision=self.revision,
            allow_patterns=scoped_patterns,
        )
        self._snapshot_root = local_root
        self._prefetched = True
        return local_root

    def _try_prefetched_path(self, path_in_repo: str) -> str | None:
        if not self._snapshot_root:
            return None
        candidate = Path(self._snapshot_root) / path_in_repo
        if candidate.is_file():
            return str(candidate)
        return None

    # --- SourceLocator interface ---

    def read_text(self, relpath: str) -> str:
        path_in_repo = self._join(relpath)
        # Prefer prefetched local copy if available.
        local = self._try_prefetched_path(path_in_repo)
        if local is not None:
            return Path(local).read_text(encoding="utf-8")

        from huggingface_hub import hf_hub_download

        local = hf_hub_download(
            repo_id=self.repo_id,
            filename=path_in_repo,
            repo_type="dataset",
            revision=self.revision,
        )
        return Path(local).read_text(encoding="utf-8")

    def list_dir(self, relpath: str) -> list[str]:
        prefix = self._join(relpath).rstrip("/")
        prefix_with_sep = f"{prefix}/" if prefix else ""
        if prefix in self._listing_cache:
            return self._listing_cache[prefix]

        all_files = self._ensure_file_listing()
        children: set[str] = set()
        for f in all_files:
            if prefix_with_sep and not f.startswith(prefix_with_sep):
                continue
            tail = f[len(prefix_with_sep):]
            if not tail:
                continue
            head = tail.split("/", 1)[0]
            if head:
                children.add(head)
        result = sorted(children)
        self._listing_cache[prefix] = result
        return result

    def exists(self, relpath: str) -> bool:
        path_in_repo = self._join(relpath)
        all_files = self._ensure_file_listing()
        # File match
        if path_in_repo in all_files:
            return True
        # Directory match
        prefix = path_in_repo.rstrip("/") + "/"
        return any(f.startswith(prefix) for f in all_files)

    def resolve(self, relpath: str) -> str:
        path_in_repo = self._join(relpath)
        return f"{HF_PREFIX}{self.repo_id}/{path_in_repo}"

    def materialize(self, relpath: str) -> str:
        from huggingface_hub import hf_hub_download

        path_in_repo = self._join(relpath)
        return hf_hub_download(
            repo_id=self.repo_id,
            filename=path_in_repo,
            repo_type="dataset",
            revision=self.revision,
        )

    def __repr__(self) -> str:  # pragma: no cover - debug only
        return (
            f"HfSourceLocator(repo_id={self.repo_id!r}, subpath={self.subpath!r}, "
            f"revision={self.revision!r})"
        )


def locator_for(source: str, **kwargs: Any) -> SourceLocator:
    """Pick the right locator for a string source.

    - ``hf://datasets/{repo}/{subpath}`` -> ``HfSourceLocator``
    - any other string -> ``LocalSourceLocator``
    """
    if source.startswith(HF_PREFIX):
        repo_id, subpath = _split_hf_url(source)
        return HfSourceLocator(repo_id=repo_id, subpath=subpath, **kwargs)
    return LocalSourceLocator(root=source, **kwargs)


def _split_hf_url(url: str) -> tuple[str, str]:
    """Split ``hf://datasets/{owner}/{name}/{subpath...}`` into (repo_id, subpath)."""
    if not url.startswith(HF_PREFIX):
        raise ValueError(f"Not an hf:// URL: {url}")
    rest = url[len(HF_PREFIX):]
    parts = rest.split("/")
    if len(parts) < 2:
        raise ValueError(
            f"hf:// URL must include both owner and dataset name, got: {url}"
        )
    repo_id = f"{parts[0]}/{parts[1]}"
    subpath = "/".join(parts[2:])
    return repo_id, subpath


def materialize(path: str) -> str:
    """Ensure a stored path string is locally readable.

    Adapter outputs (``EvaluationItem.media.videos[].path``,
    ``EvaluationItem.artifacts[*].path``) are stored as either local absolute
    paths or ``hf://datasets/...`` URLs. Metric plugins call this helper
    immediately before opening the file.
    """
    if path.startswith(HF_PREFIX):
        repo_id, sub = _split_hf_url(path)
        # subpath holds full path-in-repo when called via materialize; open locator
        # with empty subpath and resolve the full path-in-repo.
        loc = HfSourceLocator(repo_id=repo_id, subpath="")
        return loc.materialize(sub)
    return str(Path(path).expanduser().resolve())


# Silence "unused" noise for the imports re-exported in __init__.
_ = urlparse
