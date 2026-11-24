"""Portable helpers for manifest image paths.

The frozen manifests in this repository were produced on Windows, so some
historical records contain paths such as ``E:\\ocr\\...\\IMG_0001.jpg``.  Those
strings are useful provenance but are not valid paths on Linux/macOS and
``os.path.basename`` does not split a Windows path on POSIX.  Keep path handling
in one place so every script compares pages by the same stable identifier and can
find the local copy of an image after a checkout moves.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional, Union

PathLike = Union[str, os.PathLike]


def image_id(path: PathLike) -> str:
    """Return the final path component for both POSIX and Windows separators."""
    text = str(path).replace("\\", "/").rstrip("/")
    return text.rsplit("/", 1)[-1]


def resolve_image_path(
    path: PathLike,
    manifest: Optional[PathLike] = None,
    search_dirs: Optional[Iterable[PathLike]] = None,
) -> str:
    """Resolve a manifest image path to an existing local file when possible.

    Resolution order:
      1. the path exactly as written (relative paths are relative to cwd);
      2. paths relative to the manifest directory, its ``images/`` child and its
         parent's ``images/`` child (covers both ``splits/foo.jsonl`` and
         ``splits/images/...`` layouts);
      3. the repository's conventional dataset image directory;
      4. explicit ``search_dirs``.

    If nothing exists, the original string is returned unchanged so the caller
    raises the usual, transparent ``FileNotFoundError``.
    """
    raw = str(path)
    direct = Path(raw)
    if direct.is_file():
        return str(direct)

    name = image_id(raw)
    candidates = []

    if manifest is not None:
        mdir = Path(manifest).resolve().parent
        candidates.extend([
            mdir / raw,
            mdir / name,
            mdir / "images" / name,
            mdir.parent / "images" / name,
            mdir.parent.parent / "images" / name,
        ])

    candidates.extend([
        Path.cwd() / raw,
        Path.cwd() / "datasets" / "handwriting_notes" / "images" / name,
        Path.cwd() / "images" / name,
    ])
    for d in search_dirs or ():
        candidates.append(Path(d) / name)

    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return str(candidate)
    return raw
