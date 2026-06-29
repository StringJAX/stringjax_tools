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

"""Internal utility helpers shared by StringJAX Tools modules."""

from __future__ import annotations

import inspect
from collections import OrderedDict
from collections.abc import Mapping, MutableMapping
from typing import Any, Callable, TypeVar

import numpy as np


T = TypeVar("T")


def cache_key_value(value: Any, seen: set[int] | None = None) -> Any:
    r"""
    Return a hashable cache-key component for Python configuration values.

    Hashable values are keyed by value.  Containers are recursively converted
    to stable tuple representations.  Ordinary objects are keyed by identity
    and shallow ``__dict__`` state so that simple model-attribute changes can
    create fresh wrapper entries.
    """
    if seen is None:
        seen = set()

    object_id = id(value)
    if isinstance(value, np.ndarray):
        return (
            "ndarray",
            value.shape,
            value.dtype.str,
            value.tobytes(),
        )
    if isinstance(value, Mapping):
        if object_id in seen:
            return ("cycle", type(value).__module__, type(value).__qualname__, object_id)
        seen.add(object_id)
        try:
            return (
                "mapping",
                tuple(
                    sorted(
                        (
                            (cache_key_value(key, seen), cache_key_value(val, seen))
                            for key, val in value.items()
                        ),
                        key=repr,
                    )
                ),
            )
        finally:
            seen.remove(object_id)
    if isinstance(value, tuple):
        if object_id in seen:
            return ("cycle", type(value).__module__, type(value).__qualname__, object_id)
        seen.add(object_id)
        try:
            return ("tuple", tuple(cache_key_value(item, seen) for item in value))
        finally:
            seen.remove(object_id)
    if isinstance(value, list):
        if object_id in seen:
            return ("cycle", type(value).__module__, type(value).__qualname__, object_id)
        seen.add(object_id)
        try:
            return ("list", tuple(cache_key_value(item, seen) for item in value))
        finally:
            seen.remove(object_id)
    if isinstance(value, set):
        if object_id in seen:
            return ("cycle", type(value).__module__, type(value).__qualname__, object_id)
        seen.add(object_id)
        try:
            return (
                "set",
                tuple(sorted((cache_key_value(item, seen) for item in value), key=repr)),
            )
        finally:
            seen.remove(object_id)
    if hasattr(value, "__dict__") and not inspect.ismodule(value):
        if object_id in seen:
            return ("cycle", type(value).__module__, type(value).__qualname__, object_id)

        seen.add(object_id)
        try:
            state_key = cache_key_value(vars(value), seen)
        finally:
            seen.remove(object_id)
        return (
            "object",
            type(value).__module__,
            type(value).__qualname__,
            object_id,
            state_key,
        )

    try:
        hash(value)
    except TypeError:
        return ("id", id(value))
    return ("value", value)


def snapshot_static_value(value: Any, seen: dict[int, Any] | None = None) -> Any:
    r"""
    Snapshot standard mutable containers captured by cached JAX wrappers.

    Wrapper caches are keyed by the value of Python-side static configuration.
    If a cached closure kept a reference to a mutable list or dict, later
    mutation of that object could invalidate the relation between the cache key
    and the closure's behavior.  This helper recursively copies built-in
    containers while leaving ordinary objects by reference, which preserves the
    expected bound-method / model-instance behavior.
    """
    if seen is None:
        seen = {}

    object_id = id(value)
    if object_id in seen:
        return seen[object_id]

    if isinstance(value, np.ndarray):
        snapshot_array = value.copy()
        seen[object_id] = snapshot_array
        return snapshot_array
    if isinstance(value, dict):
        snapshot: dict[Any, Any] = {}
        seen[object_id] = snapshot
        snapshot.update(
            (
                snapshot_static_value(key, seen),
                snapshot_static_value(item, seen),
            )
            for key, item in value.items()
        )
        return snapshot
    if isinstance(value, list):
        snapshot_list: list[Any] = []
        seen[object_id] = snapshot_list
        snapshot_list.extend(snapshot_static_value(item, seen) for item in value)
        return snapshot_list
    if isinstance(value, tuple):
        snapshot_tuple = tuple(snapshot_static_value(item, seen) for item in value)
        seen[object_id] = snapshot_tuple
        return snapshot_tuple
    if isinstance(value, set):
        snapshot_set: set[Any] = set()
        seen[object_id] = snapshot_set
        snapshot_set.update(snapshot_static_value(item, seen) for item in value)
        return snapshot_set

    return value


def lru_cache_get(cache: MutableMapping[Any, T], key: Any) -> T | None:
    r"""Return a cached value and refresh its LRU position."""
    try:
        value = cache.pop(key)
    except KeyError:
        return None
    cache[key] = value
    return value


def lru_cache_set(
    cache: OrderedDict[Any, T],
    key: Any,
    value: T,
    *,
    maxsize: int,
) -> None:
    r"""Insert ``value`` into a bounded local LRU cache."""
    if key in cache:
        cache.pop(key)
    elif len(cache) >= maxsize:
        cache.popitem(last=False)
    cache[key] = value


def clear_ordered_cache(cache: OrderedDict[Any, Callable[..., Any]]) -> None:
    r"""Clear a local OrderedDict-backed wrapper cache."""
    cache.clear()
