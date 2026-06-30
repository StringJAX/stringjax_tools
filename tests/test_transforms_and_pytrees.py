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

"""Tests for transform, cache, and pytree helpers."""

from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from stringjax_tools.cache import configure_compilation_cache
from stringjax_tools.jit import (
    is_static,
    jit_with_dynamic_static_args,
    jit_with_static_args,
)
from stringjax_tools.pytrees import (
    PytreePolicy,
    flatten_func,
    make_pytree_flatteners,
    unflatten_func_class,
)
from stringjax_tools._utils import cache_key_value
from stringjax_tools.vmap import (
    _VMAP_JIT_CACHE,
    clear_vmap_caches,
    vmapping_func,
    vmapping_func_cached,
)


def setup_function():
    r"""Keep module-level caches isolated between tests."""
    clear_vmap_caches()


def test_manual_vmap_and_cached_vmap_reuse_wrapper():
    r"""The cached manual vmap helper should reuse wrapper objects."""

    def f(x, y, scale=1.0):
        return scale * (x + y)

    fresh = vmapping_func(f, in_axes=(0, None), scale=2.0)
    np.testing.assert_allclose(np.asarray(fresh(jnp.arange(3.0), 10.0)), [20, 22, 24])

    cached_1 = vmapping_func_cached(f, in_axes=(0, None), scale=2.0)
    cached_2 = vmapping_func_cached(f, in_axes=(0, None), scale=2.0)

    assert cached_1 is cached_2
    assert len(_VMAP_JIT_CACHE) == 1
    np.testing.assert_allclose(np.asarray(cached_1(jnp.arange(3.0), 10.0)), [20, 22, 24])


def test_cached_vmap_accepts_unhashable_keyword_configuration():
    r"""Keyword values may be unhashable while still producing stable cache keys."""

    def f(x, shifts):
        return x + shifts[0]

    cached_1 = vmapping_func_cached(f, in_axes=0, shifts=[3.0])
    cached_2 = vmapping_func_cached(f, in_axes=0, shifts=[3.0])

    assert cached_1 is cached_2
    assert len(_VMAP_JIT_CACHE) == 1
    np.testing.assert_allclose(np.asarray(cached_1(jnp.arange(3.0))), [3, 4, 5])


def test_cached_vmap_snapshots_mutable_keyword_configuration():
    r"""Cached closures should not observe later mutations of keyword config."""

    def f(x, shifts):
        return x + shifts[0]

    shifts = [3.0]
    cached = vmapping_func_cached(f, in_axes=0, shifts=shifts)
    shifts[0] = 100.0

    np.testing.assert_allclose(np.asarray(cached(jnp.arange(3.0))), [3, 4, 5])


def test_cached_vmap_snapshots_numpy_keyword_configuration():
    r"""Cached closures should copy mutable NumPy keyword arrays."""

    def f(x, shifts):
        return x + shifts[0]

    shifts = np.array([3.0])
    cached = vmapping_func_cached(f, in_axes=0, shifts=shifts)
    shifts[0] = 100.0

    np.testing.assert_allclose(np.asarray(cached(jnp.arange(3.0))), [3, 4, 5])


def test_cache_key_handles_recursive_containers():
    r"""Recursive container config should not crash cache-key construction."""
    values = []
    values.append(values)

    key = cache_key_value(values)

    assert key[0] == "list"
    assert key[1][0][0] == "cycle"


def test_jit_helpers_handle_static_arguments():
    r"""Static-argument helpers should preserve the historical util behavior."""

    def f(x, scale):
        return scale * x

    wrapped = jit_with_static_args(f, static_argnums=(1,))
    np.testing.assert_allclose(np.asarray(wrapped(jnp.arange(3.0), 2.0)), [0, 2, 4])

    dynamic = jit_with_dynamic_static_args(f)
    np.testing.assert_allclose(np.asarray(dynamic(jnp.arange(3.0), 3.0)), [0, 3, 6])

    assert is_static(1.0)
    assert not is_static(jnp.ones(2))
    assert not is_static(np.ones(2))


def test_configure_compilation_cache_records_expected_jax_updates(tmp_path, monkeypatch):
    r"""The cache helper applies only requested JAX config keys."""
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        jax.config,
        "update",
        lambda name, value: calls.append((name, value)),
    )

    cache_dir = tmp_path / "jax-cache"
    updates = configure_compilation_cache(
        cache_dir=cache_dir,
        max_size_bytes=1024,
        min_compile_time_secs=0.5,
        min_entry_size_bytes=0,
        enable_xla_caches="none",
        explain_cache_misses=True,
    )

    assert cache_dir.is_dir()
    assert updates == {
        "jax_compilation_cache_dir": str(cache_dir),
        "jax_compilation_cache_max_size": 1024,
        "jax_persistent_cache_min_compile_time_secs": 0.5,
        "jax_persistent_cache_min_entry_size_bytes": 0,
        "jax_persistent_cache_enable_xla_caches": "none",
        "jax_explain_cache_misses": True,
    }
    assert calls == list(updates.items())


def test_direct_pytree_flatten_and_unflatten_helpers():
    r"""Generic flatten/unflatten helpers should not encode package policy."""

    class Model:
        def __init__(self):
            self.h12 = 2
            self.name = "toy"
            self.array = jnp.array([1.0, 2.0])
            self._cache = {"ignored": True}

    model = Model()
    children, aux = flatten_func(
        model,
        static_keys=("h12",),
        ignore_keys=("_cache",),
    )
    rebuilt = unflatten_func_class(aux, children, Model)

    assert len(children) == 1
    np.testing.assert_allclose(np.asarray(rebuilt.array), [1.0, 2.0])
    assert rebuilt.h12 == 2
    assert rebuilt.name == "toy"
    assert not hasattr(rebuilt, "_cache")


def test_pytree_key_inputs_accept_single_strings():
    r"""A single key passed as a string should not be split into characters."""

    class Model:
        def __init__(self):
            self.h12 = 2
            self.value = jnp.array([1.0])

    children, aux = flatten_func(Model(), static_keys="h12")
    rebuilt = unflatten_func_class(aux, children, Model)

    assert len(children) == 1
    assert rebuilt.h12 == 2
    np.testing.assert_allclose(np.asarray(rebuilt.value), [1.0])


def test_pytree_policy_registration_maps_dynamic_children():
    r"""A shared policy can register a class without per-class policy globals."""

    class RegisteredModel:
        def __init__(self, scale, values):
            self.scale = scale
            self.values = values
            self._cache = "drop"

    policy = PytreePolicy(static_keys=("scale",), ignore_keys=("_cache",))
    policy.register(RegisteredModel)

    model = RegisteredModel(2, jnp.array([1.0, 2.0]))
    updated = jax.tree_util.tree_map(lambda x: x + 1, model)

    assert updated.scale == 2
    assert not hasattr(updated, "_cache")
    np.testing.assert_allclose(np.asarray(updated.values), [2.0, 3.0])


def test_pytree_policy_restores_ignored_defaults():
    r"""Ignored cache fields can be restored to safe eager defaults."""

    class CachedModel:
        def __init__(self, values):
            self.values = values
            self._sampler = "eager-cache"
            self._cache = {"before": True}

    policy = PytreePolicy(ignore_defaults={"_sampler": None, "_cache": dict})
    policy.register(CachedModel)

    model = CachedModel(jnp.array([1.0]))
    rebuilt = jax.tree_util.tree_map(lambda x: x, model)

    assert rebuilt._sampler is None
    assert rebuilt._cache == {}
    assert rebuilt._cache is not model._cache
    np.testing.assert_allclose(np.asarray(rebuilt.values), [1.0])


def test_pytree_policy_registration_can_be_used_as_decorator():
    r"""Policy registration returns the class for decorator-style use."""
    policy = PytreePolicy(static_keys="scale", ignore_keys="_cache")

    @policy.register
    class DecoratedModel:
        def __init__(self, scale, values):
            self.scale = scale
            self.values = values
            self._cache = "drop"

    model = DecoratedModel(5, jnp.array([1.0]))
    updated = jax.tree_util.tree_map(lambda x: x + 1, model)

    assert isinstance(updated, DecoratedModel)
    assert updated.scale == 5
    assert not hasattr(updated, "_cache")
    np.testing.assert_allclose(np.asarray(updated.values), [2.0])


def test_make_pytree_flatteners_explicit_style():
    r"""The explicit factory should work when a project does not name a policy."""

    class ExplicitModel:
        def __init__(self, values):
            self.kind = "explicit"
            self.values = values

    flatten, unflatten = make_pytree_flatteners(ExplicitModel)
    children, aux = flatten(ExplicitModel(jnp.array([1.0])))
    rebuilt = unflatten(aux, children)

    assert rebuilt.kind == "explicit"
    np.testing.assert_allclose(np.asarray(rebuilt.values), [1.0])


def test_make_pytree_flatteners_restores_ignored_defaults():
    r"""The explicit factory exposes the same ignored-default mechanism."""

    class ExplicitCachedModel:
        def __init__(self, values):
            self.values = values
            self._cache = {"drop": True}

    flatten, unflatten = make_pytree_flatteners(
        ExplicitCachedModel,
        ignore_defaults={"_cache": dict},
    )
    children, aux = flatten(ExplicitCachedModel(jnp.array([1.0])))
    rebuilt_1 = unflatten(aux, children)
    rebuilt_2 = unflatten(aux, children)

    assert rebuilt_1._cache == {}
    assert rebuilt_2._cache == {}
    assert rebuilt_1._cache is not rebuilt_2._cache


def test_pytree_static_validation_rejects_unhashable_static_values():
    r"""Static aux data should be hashable by default."""

    class BadStatic:
        def __init__(self):
            self.meta = {"not": "hashable"}
            self.values = jnp.ones(2)

    with pytest.raises(ValueError, match="not hashable"):
        flatten_func(BadStatic(), static_keys=("meta",))

    children, aux = flatten_func(
        BadStatic(),
        static_keys=("meta",),
        validate_static=False,
    )
    rebuilt = unflatten_func_class(aux, children, BadStatic)
    assert rebuilt.meta == {"not": "hashable"}


def test_pytree_static_validation_rejects_non_self_equal_values():
    r"""NaN-like static auxiliary data has unstable equality semantics."""

    class BadStatic:
        def __init__(self):
            self.meta = float("nan")
            self.values = jnp.ones(2)

    with pytest.raises(ValueError, match="not self-equal"):
        flatten_func(BadStatic(), static_keys=("meta",))


def test_pytree_policy_rejects_ambiguous_or_invalid_keys():
    r"""A policy should not silently ignore static semantic state."""

    with pytest.raises(ValueError, match="both static and ignored"):
        PytreePolicy(static_keys=("scale",), ignore_keys=("scale",))

    with pytest.raises(ValueError, match="both static and ignored"):
        PytreePolicy(static_keys=("_cache",), ignore_defaults={"_cache": None})

    with pytest.raises(ValueError, match="string attribute names"):
        PytreePolicy(static_keys=(1,))

    with pytest.raises(ValueError, match="string attribute names"):
        PytreePolicy(ignore_defaults={1: None})


def test_pytree_ignore_defaults_are_not_externally_mutable():
    r"""A frozen policy should not expose mutable default configuration."""

    defaults = {"_cache": dict}
    policy = PytreePolicy(ignore_defaults=defaults)
    defaults["_sampler"] = None

    assert tuple(policy.ignore_defaults) == ("_cache",)
    with pytest.raises(TypeError):
        policy.ignore_defaults["_other"] = None


def test_pytree_static_validation_accepts_numpy_scalars():
    r"""NumPy scalar metadata should be valid static auxiliary data."""

    class NumpyStatic:
        def __init__(self):
            self.degree = np.int64(3)
            self.values = jnp.ones(2)

    children, aux = flatten_func(NumpyStatic(), static_keys=("degree",))
    rebuilt = unflatten_func_class(aux, children, NumpyStatic)

    assert rebuilt.degree == np.int64(3)
    np.testing.assert_allclose(np.asarray(rebuilt.values), [1.0, 1.0])
