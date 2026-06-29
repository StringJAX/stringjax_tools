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

"""Configurable pytree registration helpers for stateful model classes.

JAX reconstructs custom pytree nodes with the registered unflatten function,
not by calling ``__init__``.  Stateful classes therefore need a deliberate state
policy: semantic configuration should be preserved as static auxiliary data,
numerical leaves should be traced, and ignored attributes should be limited to
recomputable eager caches or scratch state.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Callable

import jax
import numpy as np

__all__ = [
    "PytreePolicy",
    "flatten_func",
    "make_pytree_flatteners",
    "unflatten_func_class",
]

AuxData = tuple[tuple[str, ...], tuple[tuple[str, Any], ...]]


def _as_tuple(values: Iterable[Any] | None) -> tuple[Any, ...]:
    r"""Normalise optional iterables to tuples."""
    if values is None:
        return ()
    if isinstance(values, str):
        return (values,)
    return tuple(values)


def _validate_static_value(key: str, value: Any) -> None:
    r"""Raise if ``value`` is unsuitable for JAX pytree auxiliary data."""
    try:
        hash(value)
    except TypeError as exc:
        raise ValueError(
            f"Static pytree attribute {key!r} is not hashable. "
            "Use validate_static=False only if you know JAX can safely carry "
            "this value as auxiliary data, or remove it from static_keys."
        ) from exc

    try:
        equality = value == value
    except Exception as exc:
        raise ValueError(
            f"Static pytree attribute {key!r} has unstable equality semantics."
        ) from exc

    if not isinstance(equality, (bool, np.bool_)):
        raise ValueError(
            f"Static pytree attribute {key!r} does not compare to a scalar bool. "
            "This usually means an array-like object was marked static."
        )


def flatten_func(
    obj: Any,
    *,
    static_keys: Iterable[str] = (),
    ignore_keys: Iterable[str] = (),
    static_types: tuple[type, ...] = (str, bool),
    validate_static: bool = True,
) -> tuple[tuple[Any, ...], AuxData]:
    r"""
    Flatten ``obj`` for the JAX pytree protocol.

    Args:
        obj: Instance to flatten.
        static_keys: Attribute names stored as static auxiliary data.
        ignore_keys: Attribute names excluded from the pytree entirely.  These
            attributes are absent after reconstruction unless the consuming
            package restores them in a custom unflatten wrapper.  Do not put
            semantic configuration or user input here.
        static_types: Python value types that should always be static.
        validate_static: If ``True``, reject unhashable or array-like static
            values before they enter JAX auxiliary data.

    Returns:
        ``(children, aux_data)`` in the form required by
        ``jax.tree_util.register_pytree_node``.  ``aux_data`` is structured as
        ``(child_keys, static_items)``.
    """
    static_key_set = set(_as_tuple(static_keys))
    ignore_key_set = set(_as_tuple(ignore_keys))

    children: list[Any] = []
    child_keys: list[str] = []
    static_items: list[tuple[str, Any]] = []

    for key, value in vars(obj).items():
        if key in ignore_key_set:
            continue

        if key in static_key_set or isinstance(value, static_types):
            if validate_static:
                _validate_static_value(key, value)
            static_items.append((key, value))
        else:
            children.append(value)
            child_keys.append(key)

    return tuple(children), (tuple(child_keys), tuple(static_items))


def unflatten_func_class(
    aux_data: AuxData,
    children: tuple[Any, ...],
    myclass: type,
) -> Any:
    r"""
    Reconstruct ``myclass`` from pytree auxiliary data and children.

    ``__init__`` is intentionally bypassed.  This mirrors the standard pattern
    for JAX pytree registration of stateful objects whose constructors may
    perform validation, IO, or other eager side effects.
    """
    child_keys, static_items = aux_data
    if len(child_keys) != len(children):
        raise ValueError(
            "Pytree aux_data and children are inconsistent: "
            f"{len(child_keys)} child key(s) for {len(children)} child value(s)."
        )

    obj = object.__new__(myclass)
    for key, value in zip(child_keys, children, strict=True):
        object.__setattr__(obj, key, value)
    for key, value in static_items:
        object.__setattr__(obj, key, value)
    return obj


@dataclass(frozen=True)
class PytreePolicy:
    r"""
    Shared pytree flattening policy for one package or model family.

    A StringJAX package should define its domain-specific ``static_keys`` and
    ``ignore_keys`` locally, then reuse one policy to register all classes that
    share the same state conventions.

    Warning:
        ``ignore_keys`` are dropped during flattening and are not restored by
        this base policy.  They are suitable for recomputable caches, scratch
        arrays and eager-only helpers, but not for semantic state such as
        user-supplied bounds, physical parameters or configuration values.  Put
        such state in ``static_keys`` when it is hashable and immutable, or keep
        it as a traced child when it is array-like.  If an ignored cache must be
        present on a reconstructed object, use a class-specific unflatten
        wrapper that restores a safe default such as ``None`` or a fresh
        ``dict``.
    """

    static_keys: Iterable[str] = ()
    ignore_keys: Iterable[str] = ()
    static_types: tuple[type, ...] = (str, bool)
    validate_static: bool = True

    def __post_init__(self) -> None:
        r"""Freeze iterable configuration so the policy is stable."""
        object.__setattr__(self, "static_keys", _as_tuple(self.static_keys))
        object.__setattr__(self, "ignore_keys", _as_tuple(self.ignore_keys))
        object.__setattr__(self, "static_types", tuple(self.static_types))

    def flatten(self, obj: Any) -> tuple[tuple[Any, ...], AuxData]:
        r"""Flatten ``obj`` using this policy."""
        return flatten_func(
            obj,
            static_keys=self.static_keys,
            ignore_keys=self.ignore_keys,
            static_types=self.static_types,
            validate_static=self.validate_static,
        )

    def unflatten(
        self,
        aux_data: AuxData,
        children: tuple[Any, ...],
        myclass: type,
    ) -> Any:
        r"""Reconstruct ``myclass`` from pytree data."""
        return unflatten_func_class(aux_data, children, myclass)

    def make_flatteners(
        self,
        myclass: type,
    ) -> tuple[Callable[[Any], tuple[tuple[Any, ...], AuxData]], Callable[[AuxData, tuple[Any, ...]], Any]]:
        r"""Return class-specific ``(flatten, unflatten)`` callables."""

        def unflatten(aux_data: AuxData, children: tuple[Any, ...]) -> Any:
            return self.unflatten(aux_data, children, myclass)

        return self.flatten, unflatten

    def register(self, myclass: type) -> type:
        r"""
        Register ``myclass`` as a JAX pytree node and return the class.

        Returning the class lets this method be used either directly or as a
        decorator.
        """
        flatten, unflatten = self.make_flatteners(myclass)
        jax.tree_util.register_pytree_node(myclass, flatten, unflatten)
        return myclass


def make_pytree_flatteners(
    myclass: type,
    *,
    static_keys: Iterable[str] = (),
    ignore_keys: Iterable[str] = (),
    static_types: tuple[type, ...] = (str, bool),
    validate_static: bool = True,
) -> tuple[Callable[[Any], tuple[tuple[Any, ...], AuxData]], Callable[[AuxData, tuple[Any, ...]], Any]]:
    r"""
    Build class-specific pytree flatten/unflatten callables.

    This convenience wrapper is useful when a project wants explicit
    registration calls without first naming a shared :class:`PytreePolicy`.
    """
    policy = PytreePolicy(
        static_keys=static_keys,
        ignore_keys=ignore_keys,
        static_types=static_types,
        validate_static=validate_static,
    )
    return policy.make_flatteners(myclass)
