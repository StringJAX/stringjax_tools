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

"""Manual cached ``vmap``/``jit`` factories."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable

import jax

from ._utils import cache_key_value, lru_cache_get, lru_cache_set, snapshot_static_value

__all__ = [
    "clear_vmap_caches",
    "vmapping_func",
    "vmapping_func_cached",
]

_VMAP_JIT_CACHE_MAXSIZE = 256
_VMAP_JIT_CACHE: OrderedDict[tuple[Any, ...], Callable[..., Any]] = OrderedDict()


def vmapping_func(
    func: Callable[..., Any],
    in_axes: int | tuple[Any, ...] | None = None,
    **kwargs: Any,
) -> Callable[..., Any]:
    r"""
    Return a fresh ``jax.jit(jax.vmap(...))`` wrapper around ``func``.

    ``kwargs`` are bound inside the mapped closure.  Each call constructs a new
    wrapper object, so repeated use should normally prefer
    :func:`vmapping_func_cached`.
    """

    def mapped_call(*args: Any) -> Any:
        return func(*args, **kwargs)

    return jax.jit(jax.vmap(mapped_call, in_axes=in_axes))


def _build_vmap_jit(
    func: Callable[..., Any],
    in_axes: int | tuple[Any, ...] | None,
    frozen_kwargs: tuple[tuple[str, Any], ...],
) -> Callable[..., Any]:
    r"""
    Build one ``jax.jit(jax.vmap(...))`` wrapper.

    This function is intentionally small and private-ish: cache lookup and
    cache-key normalisation happen in :func:`vmapping_func_cached`.
    """
    kwargs = dict(frozen_kwargs)

    def mapped_call(*args: Any) -> Any:
        return func(*args, **kwargs)

    return jax.jit(jax.vmap(mapped_call, in_axes=in_axes))


def vmapping_func_cached(
    func: Callable[..., Any],
    in_axes: int | tuple[Any, ...] | None = None,
    **kwargs: Any,
) -> Callable[..., Any]:
    r"""
    Return a cached ``jax.jit(jax.vmap(...))`` wrapper around ``func``.

    The cache key includes the callable, ``in_axes``, and a robust key for
    bound keyword configuration.  Keyword values may be unhashable; they are
    used by the closure, while their cache-key representation is produced by
    :func:`stringjax_tools._utils.cache_key_value`.
    """
    kwargs_items = tuple(
        (key, snapshot_static_value(value))
        for key, value in sorted(kwargs.items(), key=lambda item: item[0])
    )
    cache_key = (
        func,
        cache_key_value(in_axes),
        tuple((key, cache_key_value(value)) for key, value in kwargs_items),
    )

    cached = lru_cache_get(_VMAP_JIT_CACHE, cache_key)
    if cached is None:
        cached = _build_vmap_jit(func, in_axes, kwargs_items)
        lru_cache_set(
            _VMAP_JIT_CACHE,
            cache_key,
            cached,
            maxsize=_VMAP_JIT_CACHE_MAXSIZE,
        )
    return cached


def clear_vmap_caches() -> None:
    r"""Clear wrappers cached by :func:`vmapping_func_cached`."""
    _VMAP_JIT_CACHE.clear()
