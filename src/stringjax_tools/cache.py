# Copyright 2022-2026 Andreas Schachner
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

"""Opt-in helpers for JAX compilation-cache configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax

__all__ = ["configure_compilation_cache"]


def configure_compilation_cache(
    cache_dir: str | Path | None = None,
    *,
    max_size_bytes: int | None = None,
    min_compile_time_secs: float | None = None,
    min_entry_size_bytes: int | None = None,
    enable_xla_caches: str | None = None,
    explain_cache_misses: bool | None = None,
) -> dict[str, Any]:
    r"""
    Configure JAX's persistent compilation cache explicitly.

    This helper has no import-time side effects.  Call it near the top of a
    script or notebook before any JAX compilation happens.

    Args:
        cache_dir: Directory for the persistent compilation cache.  If supplied,
            the directory is created if needed.  If ``None``, the cache
            directory is not changed.
        max_size_bytes: Maximum persistent cache size in bytes.  JAX interprets
            ``0`` as disabling the persistent cache and ``-1`` as no size limit.
        min_compile_time_secs: Minimum compilation time required before an entry
            is written to the persistent cache.
        min_entry_size_bytes: Minimum entry size required before an entry is
            written to the persistent cache.
        enable_xla_caches: Value forwarded to
            ``jax_persistent_cache_enable_xla_caches``.
        explain_cache_misses: If ``True``, ask JAX to explain persistent-cache
            misses.

    Returns:
        Configuration names and values that were applied.
    """
    updates: dict[str, Any] = {}

    if cache_dir is not None:
        cache_path = Path(cache_dir).expanduser()
        cache_path.mkdir(parents=True, exist_ok=True)
        updates["jax_compilation_cache_dir"] = str(cache_path)
    if max_size_bytes is not None:
        updates["jax_compilation_cache_max_size"] = int(max_size_bytes)
    if min_compile_time_secs is not None:
        updates["jax_persistent_cache_min_compile_time_secs"] = float(
            min_compile_time_secs
        )
    if min_entry_size_bytes is not None:
        updates["jax_persistent_cache_min_entry_size_bytes"] = int(min_entry_size_bytes)
    if enable_xla_caches is not None:
        updates["jax_persistent_cache_enable_xla_caches"] = enable_xla_caches
    if explain_cache_misses is not None:
        updates["jax_explain_cache_misses"] = bool(explain_cache_misses)

    for name, value in updates.items():
        jax.config.update(name, value)

    return updates

