# Copyright 2023-2026 Andreas Schachner
#
# This file is part of StringJAX Tools.
#
# StringJAX Tools is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# StringJAX Tools is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with StringJAX Tools. If not, see <https://www.gnu.org/licenses/>.

"""Record which code produced a result: package versions, source trees and git state."""

from __future__ import annotations

import datetime
import importlib.metadata as metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

__all__ = ["git_state", "package_info", "provenance"]


def _git(path: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        # --no-optional-locks: ``git status`` must not take the index lock of the repository
        ["git", "--no-optional-locks", "-C", str(path), *args],
        capture_output=True,
        text=True,
        timeout=10,
    )


def git_state(path: str | os.PathLike[str], tracked_file: str | os.PathLike[str] | None = None):
    r"""
    Return the git commit of the repository containing ``path`` and whether its tree is dirty.

    The record holds ``commit`` (the full commit hash), ``dirty`` and ``root`` (the
    repository's top-level directory); each is ``None`` when the state is unknown (no repository,
    no git, or ``tracked_file`` untracked). ``dirty`` covers the whole repository, not only
    ``path``, and counts changes to tracked files only (untracked files are ignored). If
    ``tracked_file`` is given and git does not track it, the repository is not the file's source
    (for example an environment's ``site-packages`` inside some unrelated repository), and the
    state is reported as unknown.

    Args:
        path: A directory inside the repository.
        tracked_file: A file that must be tracked by that repository for its state to count.

    Returns:
        dict: The keys ``commit``, ``dirty`` and ``root``.
    """
    unknown: dict[str, Any] = {"commit": None, "dirty": None, "root": None}
    path = Path(path)
    try:
        root = _git(path, "rev-parse", "--show-toplevel")
        head = _git(path, "rev-parse", "HEAD")
        if root.returncode != 0 or head.returncode != 0:
            return unknown
        if tracked_file is not None:
            listed = _git(path, "ls-files", "--error-unmatch", "--", str(Path(tracked_file)))
            if listed.returncode != 0:
                return unknown
        status = _git(path, "status", "--porcelain", "--untracked-files=no")
        if status.returncode != 0:
            return unknown
    except (OSError, subprocess.SubprocessError):
        return unknown
    return {
        "commit": head.stdout.strip(),
        "dirty": bool(status.stdout.strip()),
        "root": root.stdout.strip(),
    }


def _distribution(name: str, mapping: Mapping[str, list[str]] | None = None) -> str | None:
    """Name of the installed distribution that provides the top-level module ``name``."""
    if mapping is None:
        mapping = metadata.packages_distributions()
    candidates = mapping.get(name, [])
    for candidate in [*candidates, name, name.replace("_", "-")]:
        try:
            metadata.version(candidate)
        except metadata.PackageNotFoundError:
            continue
        return candidate
    return None


def _editable(distribution: str) -> bool | None:
    """Whether ``distribution`` is an editable install (PEP 610), ``None`` if not recorded."""
    try:
        text = metadata.distribution(distribution).read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return None
    if not text:
        return None
    try:
        return bool(json.loads(text).get("dir_info", {}).get("editable", False))
    except (ValueError, AttributeError):
        return None


def package_info(
    name: str, *, _distributions: Mapping[str, list[str]] | None = None
) -> dict[str, Any]:
    r"""
    Describe the installed package with the top-level import name ``name``, without importing it.

    An editable install's distribution metadata records the version at install time, which can
    lag the source tree. The record therefore also holds the module's own ``__version__`` (only
    if the module is already imported), its source directory and that directory's git state, and
    flags a disagreement between the two versions. The record's keys:

    - ``distribution``: the installed distribution's name, or ``None``;
    - ``distribution_version``: its version, or ``None``;
    - ``editable``: whether that distribution is an editable install (from its PEP 610
      ``direct_url.json``), ``None`` if this is not recorded;
    - ``imported``: whether the module is already imported;
    - ``module_version``: the module's ``__version__``, ``None`` unless imported;
    - ``version_mismatch``: whether the two version strings differ as written (no
      normalisation), ``None`` if either is unknown;
    - ``path``: the source directory, ``None`` if the module is not found;
    - ``git``: :func:`git_state` of ``path``, requiring the module's file to be tracked.

    Args:
        name: A top-level import name, e.g. ``"stringjax_tools"``.

    Returns:
        dict: The JSON-safe record described above.

    Raises:
        ValueError: If ``name`` is not a top-level name (finding a submodule would import its
            parent package).
    """
    if not name or "." in name:
        raise ValueError(f"expected a top-level import name, got {name!r}")
    distribution = _distribution(name, _distributions)
    distribution_version = metadata.version(distribution) if distribution else None
    module = sys.modules.get(name)  # never imported here: only what the caller has loaded
    module_version = getattr(module, "__version__", None) if module is not None else None
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        spec = None
    origin = spec.origin if spec is not None and spec.has_location and spec.origin else None
    source = Path(origin).resolve() if origin else None
    where = source.parent if source else None
    mismatch = None
    if distribution_version is not None and module_version is not None:
        mismatch = str(module_version) != distribution_version
    return {
        "distribution": distribution,
        "distribution_version": distribution_version,
        "editable": _editable(distribution) if distribution else None,
        "imported": module is not None,
        "module_version": None if module_version is None else str(module_version),
        "version_mismatch": mismatch,
        "path": str(where) if where else None,
        "git": git_state(source.parent, source)
        if source
        else {"commit": None, "dirty": None, "root": None},
    }


def _jax_state() -> dict[str, Any] | None:
    """JAX's version and 64-bit setting, read without initialising a backend; ``None`` if absent."""
    try:
        import jax
    except ImportError:
        return None
    enable_x64 = bool(jax.config.jax_enable_x64)  # type: ignore[attr-defined]
    return {"version": jax.__version__, "enable_x64": enable_x64}


def provenance(packages: Iterable[str] | str, **extra: Any) -> dict[str, Any]:
    r"""
    Return a JSON-safe record of the code and environment a result was produced with.

    Store it next to the result. Call it after the packages a run uses are imported, so that
    ``module_version`` and the 64-bit setting are the ones the run saw. The record's keys:
    ``time`` (UTC, ISO 8601), ``python``, ``platform``, ``packages`` (name ->
    :func:`package_info`), ``jax`` (``version`` and ``enable_x64``, or ``None`` if JAX is not
    installed; no backend is initialised) and the ``extra`` entries.

    Args:
        packages: Top-level import names to describe (:func:`package_info` each); a single
            name may be given on its own.
        **extra: Further entries, e.g. the settings of the computation.

    Returns:
        dict: The JSON-safe record described above.

    Raises:
        ValueError: If an ``extra`` entry has the name of a standard key.

    Example:
        .. code-block:: python

            from stringjax_tools import provenance

            record = provenance(["jaxvacua", "jaxpolylog"], seed=0, tolerance=1e-10)
    """
    names = [packages] if isinstance(packages, str) else list(packages)
    distributions = metadata.packages_distributions()  # once: it scans every installed package
    record: dict[str, Any] = {
        "time": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: package_info(name, _distributions=distributions) for name in names},
        "jax": _jax_state(),
    }
    clash = set(record) & set(extra)
    if clash:
        raise ValueError(f"extra entries would overwrite {sorted(clash)}")
    record.update(extra)
    return record
