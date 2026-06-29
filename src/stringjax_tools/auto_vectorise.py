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

"""Rank-checked automatic vectorisation helpers.

The central helper is :func:`auto_vmap`, a conservative decorator for functions
whose single-sample input ranks are known.  At call time, selected arguments are
checked before JAX tracing.  Rank ``r`` is treated as one sample, rank ``r + 1``
as a paired leading-axis batch, and checked batched arguments must share the
same leading size.
"""

from __future__ import annotations

import inspect
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from numbers import Integral
from typing import Any, Callable, Mapping, Sequence, TypeAlias

import jax
import jax.numpy as jnp

from ._utils import cache_key_value, lru_cache_get, lru_cache_set, snapshot_static_value

Shape = tuple[int, ...]
ShapeLike = int | Sequence[int] | None
ShapeSpec = Any

__all__ = [
    "ArgSpec",
    "auto_vmap",
    "auto_vmap_default_shapes",
    "auto_vmap_defaults",
    "clear_auto_vmap_caches",
    "get_auto_vmap_default_shapes",
    "get_auto_vmap_defaults",
    "reset_auto_vmap_default_shapes",
    "reset_auto_vmap_defaults",
    "set_auto_vmap_default_shapes",
    "set_auto_vmap_defaults",
]

_AUTO_VMAP_CACHE_MAXSIZE = 256
_AUTO_VMAP_CACHE: OrderedDict[tuple[Any, ...], Callable[..., Any]] = OrderedDict()
_AUTO_VMAP_DEFAULT_RANKS: dict[str, int] = {}
_AUTO_VMAP_DEFAULT_SHAPES: dict[str, ShapeSpec] = {}


def _is_nonnegative_int(value: Any) -> bool:
    r"""Return whether ``value`` is a valid non-negative integer dimension."""
    return isinstance(value, Integral) and not isinstance(value, bool) and int(value) >= 0


def _normalise_rank(name: str, rank: Any) -> int:
    r"""Normalise and validate a sample-rank declaration."""
    if not _is_nonnegative_int(rank):
        raise ValueError(
            f"Sample rank for argument {name!r} must be a non-negative integer, "
            f"got {rank!r}."
        )
    return int(rank)


def _normalise_rank_mapping(ranks: Mapping[str, Any] | None) -> dict[str, int]:
    r"""Normalise a mapping of argument names to sample ranks."""
    if ranks is None:
        return {}
    return {name: _normalise_rank(name, rank) for name, rank in dict(ranks).items()}


def _normalise_concrete_shape(shape: ShapeLike) -> Shape:
    r"""Return a concrete integer shape as a tuple."""
    if shape is None:
        return ()
    if _is_nonnegative_int(shape):
        return (int(shape),)
    if isinstance(shape, str):
        raise ValueError(f"Concrete shapes cannot contain unresolved string {shape!r}.")

    try:
        dims = tuple(shape)
    except TypeError as exc:
        raise ValueError(f"Could not interpret shape {shape!r}.") from exc

    if not all(_is_nonnegative_int(dim) for dim in dims):
        raise ValueError(f"Shape entries must be non-negative integers, got {dims}.")
    return tuple(int(dim) for dim in dims)


@dataclass(frozen=True)
class ArgSpec:
    r"""
    Optional exact trailing sample-shape specification.

    ``ArgSpec`` is not needed for the common rank-only API.  It is retained as
    a readable wrapper for entries in ``sample_shapes`` or global shape
    defaults.

    Args:
        shape: Expected trailing sample shape.  Use ``None`` or ``()`` for a
            scalar, an ``int`` for a fixed one-dimensional sample, a tuple/list
            for higher-rank samples, a string for attribute lookup on the first
            bound object exposing that attribute, or a callable receiving the
            bound argument mapping and returning a concrete shape.
    """

    shape: ShapeSpec = ()


ShapeSpecLike: TypeAlias = ArgSpec | ShapeSpec


def _argument_shape(value: Any) -> Shape:
    r"""Return the runtime shape of an argument without entering a JAX trace."""
    try:
        return tuple(jnp.shape(value))
    except Exception as exc:
        raise ValueError(f"Could not determine shape of argument {value!r}.") from exc


def _find_attribute_owner(
    attr_name: str,
    bound_arguments: Mapping[str, Any],
) -> Any:
    r"""Find the first bound object exposing ``attr_name``."""
    for preferred in ("self", "cls"):
        owner = bound_arguments.get(preferred)
        if owner is not None and hasattr(owner, attr_name):
            return owner

    for owner in bound_arguments.values():
        if hasattr(owner, attr_name):
            return owner

    raise ValueError(
        f"Could not resolve shape attribute {attr_name!r}; no bound argument "
        "exposes this attribute."
    )


def _resolve_shape_dim(dim: Any, bound_arguments: Mapping[str, Any]) -> int:
    r"""Resolve one dimension in a shape specification."""
    if _is_nonnegative_int(dim):
        return int(dim)
    if isinstance(dim, str):
        value = getattr(_find_attribute_owner(dim, bound_arguments), dim)
        if not _is_nonnegative_int(value):
            raise ValueError(
                f"Shape attribute {dim!r} resolved to {value!r}, but dimensions "
                "must be non-negative integers."
            )
        return int(value)
    raise ValueError(f"Unsupported shape dimension specification {dim!r}.")


def _resolve_shape_spec(
    spec: ShapeSpecLike,
    bound_arguments: Mapping[str, Any],
) -> Shape:
    r"""Resolve an optional exact sample-shape specification."""
    if isinstance(spec, ArgSpec):
        spec = spec.shape

    if callable(spec):
        return _normalise_concrete_shape(spec(bound_arguments))
    if spec is None:
        return ()
    if _is_nonnegative_int(spec):
        return (int(spec),)
    if isinstance(spec, str):
        return (_resolve_shape_dim(spec, bound_arguments),)

    try:
        dims = tuple(spec)
    except TypeError as exc:
        raise ValueError(f"Could not interpret shape specification {spec!r}.") from exc

    return tuple(_resolve_shape_dim(dim, bound_arguments) for dim in dims)


def _infer_vmap_axis(
    name: str,
    value: Any,
    sample_rank: int,
) -> tuple[int | None, int | None, Shape]:
    r"""
    Infer whether ``value`` is unbatched or carries one leading batch axis.

    Returns ``(axis, batch_size, sample_shape)``.
    """
    shape = _argument_shape(value)
    ndim = len(shape)

    if ndim == sample_rank:
        return None, None, shape
    if ndim == sample_rank + 1:
        return 0, shape[0], shape[1:]

    raise ValueError(
        f"Argument {name!r} has shape {shape}; expected sample rank "
        f"{sample_rank} or a leading-axis batch with rank {sample_rank + 1}."
    )


def _call_with_bound_arguments(
    func: Callable[..., Any],
    parameters: Mapping[str, inspect.Parameter],
    arguments: Mapping[str, Any],
) -> Any:
    r"""Call ``func`` from an ordered mapping of bound argument values."""
    positional: list[Any] = []
    keywords: dict[str, Any] = {}

    for name, parameter in parameters.items():
        if name not in arguments:
            continue

        value = arguments[name]
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            positional.append(value)
        elif parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
            positional.append(value)
        elif parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            positional.extend(value)
        elif parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            keywords[name] = value
        elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
            keywords.update(value)

    return func(*positional, **keywords)


def set_auto_vmap_defaults(**sample_ranks: Any) -> dict[str, int]:
    r"""Update global default sample ranks used by :func:`auto_vmap`."""
    _AUTO_VMAP_DEFAULT_RANKS.update(_normalise_rank_mapping(sample_ranks))
    return get_auto_vmap_defaults()


def get_auto_vmap_defaults() -> dict[str, int]:
    r"""Return a copy of the active global sample-rank defaults."""
    return dict(_AUTO_VMAP_DEFAULT_RANKS)


def reset_auto_vmap_defaults() -> None:
    r"""Clear all global sample-rank defaults."""
    _AUTO_VMAP_DEFAULT_RANKS.clear()


@contextmanager
def auto_vmap_defaults(**sample_ranks: Any) -> Iterator[dict[str, int]]:
    r"""Temporarily update global sample-rank defaults inside a ``with`` block."""
    previous = get_auto_vmap_defaults()
    set_auto_vmap_defaults(**sample_ranks)
    try:
        yield get_auto_vmap_defaults()
    finally:
        _AUTO_VMAP_DEFAULT_RANKS.clear()
        _AUTO_VMAP_DEFAULT_RANKS.update(previous)


def set_auto_vmap_default_shapes(**sample_shapes: ShapeSpecLike) -> dict[str, ShapeSpec]:
    r"""
    Update global default exact sample-shape specifications.

    These defaults are used only when :func:`auto_vmap` is called with
    ``validate_shapes=True``.
    """
    _AUTO_VMAP_DEFAULT_SHAPES.update(sample_shapes)
    return get_auto_vmap_default_shapes()


def get_auto_vmap_default_shapes() -> dict[str, ShapeSpec]:
    r"""Return a copy of the active global sample-shape defaults."""
    return dict(_AUTO_VMAP_DEFAULT_SHAPES)


def reset_auto_vmap_default_shapes() -> None:
    r"""Clear all global sample-shape defaults."""
    _AUTO_VMAP_DEFAULT_SHAPES.clear()


@contextmanager
def auto_vmap_default_shapes(**sample_shapes: ShapeSpecLike) -> Iterator[dict[str, ShapeSpec]]:
    r"""Temporarily update global sample-shape defaults inside a ``with`` block."""
    previous = get_auto_vmap_default_shapes()
    set_auto_vmap_default_shapes(**sample_shapes)
    try:
        yield get_auto_vmap_default_shapes()
    finally:
        _AUTO_VMAP_DEFAULT_SHAPES.clear()
        _AUTO_VMAP_DEFAULT_SHAPES.update(previous)


def _validate_local_names(
    kind: str,
    names: Sequence[str],
    parameters: Mapping[str, inspect.Parameter],
    func_name: str,
) -> None:
    r"""Validate local decorator declarations against a function signature."""
    missing = sorted(name for name in names if name not in parameters)
    if missing:
        raise ValueError(
            f"auto_vmap {kind} refer to unknown argument(s) {missing} "
            f"for function {func_name!r}."
        )

    unsupported = sorted(
        name
        for name in names
        if parameters[name].kind
        in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    )
    if unsupported:
        raise ValueError(
            "auto_vmap cannot infer ranks or shapes for variadic argument(s) "
            f"{unsupported} in function {func_name!r}."
        )


def _active_ranks(
    parameters: Mapping[str, inspect.Parameter],
    local_ranks: Mapping[str, int],
) -> dict[str, int]:
    r"""Merge applicable global ranks with local rank overrides."""
    ranks = {
        name: rank
        for name, rank in _AUTO_VMAP_DEFAULT_RANKS.items()
        if name in parameters
    }
    ranks.update(local_ranks)
    return ranks


def _active_shapes(
    parameters: Mapping[str, inspect.Parameter],
    local_shapes: Mapping[str, ShapeSpecLike],
) -> dict[str, ShapeSpecLike]:
    r"""Merge applicable global shapes with local shape overrides."""
    shapes = {
        name: spec
        for name, spec in _AUTO_VMAP_DEFAULT_SHAPES.items()
        if name in parameters
    }
    shapes.update(local_shapes)
    return shapes


def auto_vmap(
    sample_ranks: Mapping[str, int] | None = None,
    *,
    sample_shapes: Mapping[str, ShapeSpecLike] | None = None,
    validate_shapes: bool = False,
    jit: bool = True,
    **named_sample_ranks: int,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    r"""
    Decorate a single-sample function with conservative automatic batching.

    Sample ranks decide batching.  For an argument declared with sample rank
    ``r``, rank ``r`` is treated as one unbatched sample and rank ``r + 1`` is
    treated as a leading-axis batch.  If several selected arguments are
    batched, their leading batch sizes must agree.  Exact trailing-shape
    validation is optional and separate from batching detection.
    """
    local_ranks = _normalise_rank_mapping(sample_ranks)
    local_ranks.update(_normalise_rank_mapping(named_sample_ranks))
    local_shapes = dict(sample_shapes or {})

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        signature = inspect.signature(func)
        parameters = signature.parameters

        _validate_local_names("sample ranks", tuple(local_ranks), parameters, func.__name__)
        _validate_local_names("sample shapes", tuple(local_shapes), parameters, func.__name__)

        @wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()

            ranks = _active_ranks(parameters, local_ranks)
            shape_specs = _active_shapes(parameters, local_shapes)
            if not ranks:
                if validate_shapes and shape_specs:
                    raise ValueError(
                        "auto_vmap shape specifications require matching "
                        f"sample-rank declarations; no ranks are active for "
                        f"{sorted(shape_specs)} in function {func.__name__!r}."
                    )
                raise ValueError(
                    f"auto_vmap has no sample-rank declarations for function "
                    f"{func.__name__!r}. Pass ranks to auto_vmap or configure "
                    "global defaults with set_auto_vmap_defaults(...)."
                )

            _validate_local_names("active sample ranks", tuple(ranks), parameters, func.__name__)
            if validate_shapes:
                unchecked_shape_specs = sorted(set(shape_specs) - set(ranks))
                if unchecked_shape_specs:
                    raise ValueError(
                        "auto_vmap shape specifications require matching "
                        f"sample-rank declarations; no ranks are active for "
                        f"{unchecked_shape_specs} in function {func.__name__!r}."
                    )

            batch_size: int | None = None
            dynamic_values: list[Any] = []
            dynamic_names: list[str] = []
            in_axes: list[int | None] = []
            resolved_shapes: dict[str, Shape] = {}
            resolved_rank_items: list[tuple[str, int]] = []

            for name, sample_rank in ranks.items():
                axis, size, sample_shape = _infer_vmap_axis(
                    name,
                    bound.arguments[name],
                    sample_rank,
                )
                dynamic_names.append(name)
                dynamic_values.append(bound.arguments[name])
                in_axes.append(axis)
                resolved_rank_items.append((name, sample_rank))

                if validate_shapes and name in shape_specs:
                    expected_shape = _resolve_shape_spec(shape_specs[name], bound.arguments)
                    resolved_shapes[name] = expected_shape
                    if sample_shape != expected_shape:
                        raise ValueError(
                            f"Argument {name!r} has trailing sample shape {sample_shape}; "
                            f"expected {expected_shape}."
                        )

                if size is None:
                    continue
                if batch_size is None:
                    batch_size = size
                elif batch_size != size:
                    raise ValueError(
                        f"Incompatible batch sizes in auto_vmap call: got {batch_size} "
                        f"and {size} while checking argument {name!r}."
                    )

            if batch_size is None:
                return func(*args, **kwargs)

            static_arguments = {
                name: snapshot_static_value(value)
                for name, value in bound.arguments.items()
                if name not in ranks
            }
            static_key = tuple(
                (name, cache_key_value(value))
                for name, value in static_arguments.items()
            )
            cache_key = (
                func,
                tuple(dynamic_names),
                tuple(in_axes),
                bool(jit),
                tuple(resolved_rank_items),
                tuple(
                    (name, resolved_shapes[name])
                    for name in dynamic_names
                    if name in resolved_shapes
                ),
                static_key,
            )

            cached = lru_cache_get(_AUTO_VMAP_CACHE, cache_key)
            if cached is None:

                def mapped_call(*dynamic_args: Any) -> Any:
                    call_arguments = dict(static_arguments)
                    call_arguments.update(zip(dynamic_names, dynamic_args, strict=True))
                    return _call_with_bound_arguments(func, parameters, call_arguments)

                vmapped = jax.vmap(mapped_call, in_axes=tuple(in_axes))
                cached = jax.jit(vmapped) if jit else vmapped
                lru_cache_set(
                    _AUTO_VMAP_CACHE,
                    cache_key,
                    cached,
                    maxsize=_AUTO_VMAP_CACHE_MAXSIZE,
                )

            return cached(*dynamic_values)

        wrapped.auto_vmap_local_ranks = local_ranks
        wrapped.auto_vmap_local_shapes = local_shapes
        wrapped.auto_vmap_axes = lambda *args, **kwargs: _axes_by_call(
            signature, parameters, local_ranks, args, kwargs
        )
        return wrapped

    return decorator


def _axes_by_call(
    signature: inspect.Signature,
    parameters: Mapping[str, inspect.Parameter],
    local_ranks: Mapping[str, int],
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> dict[str, int | None]:
    r"""Return inferred axes for a call, primarily for debugging and tests."""
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    ranks = _active_ranks(parameters, local_ranks)
    return {
        name: _infer_vmap_axis(name, bound.arguments[name], rank)[0]
        for name, rank in ranks.items()
    }


def clear_auto_vmap_caches(clear_jax: bool = True) -> None:
    r"""
    Clear cached wrappers created by :func:`auto_vmap`.

    Args:
        clear_jax: If ``True``, also call ``jax.clear_caches()`` when available.
            This clears JAX's in-process compilation caches but does not remove
            entries from a persistent on-disk compilation cache.
    """
    _AUTO_VMAP_CACHE.clear()
    if clear_jax and hasattr(jax, "clear_caches"):
        jax.clear_caches()


def _smoke_test() -> None:
    r"""Run a small self-contained smoke test for ``python -m stringjax_tools``."""
    import numpy as np

    @auto_vmap(moduli=1, tau=0, fluxes=1)
    def f(moduli: Any, tau: Any, fluxes: Any, scale: float = 1.0) -> Any:
        return scale * (jnp.sum(moduli) + tau + jnp.sum(fluxes))

    single = f(jnp.array([1.0, 2.0]), jnp.array(3.0), jnp.arange(6.0), scale=2.0)
    paired = f(jnp.ones((4, 2)), jnp.arange(4.0), jnp.ones((4, 6)), scale=2.0)
    broadcast = f(jnp.ones((4, 2)), jnp.array(3.0), jnp.ones(6), scale=2.0)

    assert np.asarray(single).shape == ()
    np.testing.assert_allclose(np.asarray(single), 42.0)
    assert np.asarray(paired).shape == (4,)
    np.testing.assert_allclose(np.asarray(paired), np.array([16.0, 18.0, 20.0, 22.0]))
    assert np.asarray(broadcast).shape == (4,)
    np.testing.assert_allclose(np.asarray(broadcast), np.full(4, 22.0))

    try:
        f(jnp.ones((4, 2)), jnp.ones((5,)), jnp.ones((4, 6)))
    except ValueError:
        pass
    else:
        raise AssertionError("incompatible batch sizes did not raise ValueError")

    try:
        f(jnp.ones((4, 2, 1)), jnp.ones((4,)), jnp.ones((4, 6)))
    except ValueError:
        pass
    else:
        raise AssertionError("malformed moduli rank did not raise ValueError")

    print("stringjax_tools smoke test passed")
