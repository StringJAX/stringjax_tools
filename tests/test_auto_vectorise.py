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

"""Tests for rank-checked automatic vectorisation helpers."""

from __future__ import annotations

import numpy as np
import pytest

import jax.numpy as jnp

import stringjax_tools
import stringjax_tools.auto_vectorise as av


def setup_function():
    r"""Keep module-level defaults isolated between tests."""
    av.reset_auto_vmap_defaults()
    av.reset_auto_vmap_default_shapes()
    av.clear_auto_vmap_caches(clear_jax=False)


def _toy_observable(moduli, tau, fluxes, scale=1.0):
    r"""Small single-point function with the common vector/scalar pattern."""
    return scale * (jnp.sum(moduli) + tau + jnp.sum(fluxes))


def test_public_package_exports_expected_api():
    r"""The package root should expose the documented public helpers."""
    expected = {
        "ArgSpec",
        "PytreePolicy",
        "auto_vmap",
        "auto_vmap_default_shapes",
        "auto_vmap_defaults",
        "clear_auto_vmap_caches",
        "clear_vmap_caches",
        "configure_compilation_cache",
        "flatten_func",
        "get_auto_vmap_default_shapes",
        "get_auto_vmap_defaults",
        "is_static",
        "jit_with_dynamic_static_args",
        "jit_with_static_args",
        "make_pytree_flatteners",
        "reset_auto_vmap_default_shapes",
        "reset_auto_vmap_defaults",
        "set_auto_vmap_default_shapes",
        "set_auto_vmap_defaults",
        "unflatten_func_class",
        "vmapping_func",
        "vmapping_func_cached",
    }

    assert set(stringjax_tools.__all__) == expected
    for name in expected:
        assert hasattr(stringjax_tools, name)


def test_rank_only_auto_vmap_scalar_paired_and_broadcast_calls():
    r"""The decorator preserves scalar calls and maps only leading batches."""
    wrapped = av.auto_vmap(moduli=1, tau=0, fluxes=1)(_toy_observable)

    single = wrapped(
        jnp.array([1.0, 2.0]),
        jnp.array(3.0),
        jnp.arange(6.0),
        scale=2.0,
    )
    paired = wrapped(
        jnp.ones((4, 2)),
        jnp.arange(4.0),
        jnp.ones((4, 6)),
        scale=2.0,
    )
    broadcast = wrapped(
        jnp.ones((4, 2)),
        jnp.array(3.0),
        jnp.ones(6),
        scale=2.0,
    )

    np.testing.assert_allclose(np.asarray(single), 42.0)
    np.testing.assert_allclose(np.asarray(paired), [16.0, 18.0, 20.0, 22.0])
    np.testing.assert_allclose(np.asarray(broadcast), np.full(4, 22.0))

    assert wrapped.auto_vmap_axes(
        jnp.ones((4, 2)), jnp.array(3.0), jnp.ones(6)
    ) == {"moduli": 0, "tau": None, "fluxes": None}


def test_square_vector_batch_is_treated_as_batched():
    r"""A shape like ``(h, h)`` is batched when the declared sample rank is one."""
    wrapped = av.auto_vmap(moduli=1)(_toy_observable)
    result = wrapped(jnp.ones((3, 3)), jnp.array(0.0), jnp.zeros(6))
    np.testing.assert_allclose(np.asarray(result), np.full(3, 3.0))


def test_auto_vmap_rejects_malformed_and_mismatched_batches():
    r"""Invalid ranks fail before JAX traces the function."""
    wrapped = av.auto_vmap(moduli=1, tau=0, fluxes=1)(_toy_observable)

    with pytest.raises(ValueError, match="Incompatible batch sizes"):
        wrapped(jnp.ones((4, 2)), jnp.ones((5,)), jnp.ones((4, 6)))

    with pytest.raises(ValueError, match="expected sample rank 1"):
        wrapped(jnp.ones((4, 2, 1)), jnp.ones((4,)), jnp.ones((4, 6)))


def test_auto_vmap_rejects_unknown_variadic_and_bad_rank_specs():
    r"""Decorator construction catches ambiguous local declarations."""
    with pytest.raises(ValueError, match="unknown argument"):
        av.auto_vmap(missing=1)(_toy_observable)

    with pytest.raises(ValueError, match="non-negative integer"):
        av.auto_vmap(moduli=(2,))(_toy_observable)

    def with_args(*values):
        return values[0]

    with pytest.raises(ValueError, match="variadic"):
        av.auto_vmap(values=1)(with_args)

    def with_kwargs(**values):
        return values["x"]

    with pytest.raises(ValueError, match="variadic"):
        av.auto_vmap(values=1)(with_kwargs)


def test_global_rank_defaults_and_local_override():
    r"""Global defaults can supply ranks, while local declarations override them."""
    av.set_auto_vmap_defaults(moduli=1, tau=0, fluxes=1)

    wrapped = av.auto_vmap()(_toy_observable)
    result = wrapped(jnp.ones((2, 3)), jnp.arange(2.0), jnp.ones((2, 4)))
    np.testing.assert_allclose(np.asarray(result), [7.0, 8.0])

    local = av.auto_vmap(moduli=2, tau=0, fluxes=1)(_toy_observable)
    assert local.auto_vmap_axes(
        jnp.ones((5, 2, 3)), jnp.ones((5,)), jnp.ones((5, 4))
    ) == {"moduli": 0, "tau": 0, "fluxes": 0}


def test_global_rank_context_manager_restores_on_exit():
    r"""The rank-default context manager restores the previous state."""
    av.set_auto_vmap_defaults(x=1)

    with av.auto_vmap_defaults(y=0) as active:
        assert active == {"x": 1, "y": 0}
        assert av.get_auto_vmap_defaults() == {"x": 1, "y": 0}

    assert av.get_auto_vmap_defaults() == {"x": 1}


def test_auto_vmap_without_applicable_rank_declarations_raises_on_call():
    r"""A wrapper with no applicable ranks gives a clear error."""
    wrapped = av.auto_vmap()(_toy_observable)

    with pytest.raises(ValueError, match="no sample-rank declarations"):
        wrapped(jnp.ones(2), jnp.array(0.0), jnp.ones(6))


def test_shape_validation_resolves_string_attributes_and_callables():
    r"""Shape specs can be resolved from bound objects without hard-coded names."""

    class ToyModel:
        def __init__(self, dim, width):
            self.dim = dim
            self.width = width

        @av.auto_vmap(
            sample_ranks={"x": 1, "y": 1},
            sample_shapes={
                "x": "dim",
                "y": lambda bound: (2 * bound["self"].width,),
            },
            validate_shapes=True,
            jit=False,
        )
        def evaluate(self, x, y):
            return jnp.sum(x) + jnp.sum(y)

    model = ToyModel(dim=3, width=2)
    result = model.evaluate(jnp.ones((5, 3)), jnp.ones((5, 4)))
    np.testing.assert_allclose(np.asarray(result), np.full(5, 7.0))

    with pytest.raises(ValueError, match="trailing sample shape"):
        model.evaluate(jnp.ones((5, 2)), jnp.ones((5, 4)))


def test_shape_validation_requires_matching_rank_declarations():
    r"""Shape declarations without rank declarations should fail loudly."""
    wrapped = av.auto_vmap(
        sample_shapes={"x": 3},
        validate_shapes=True,
    )(lambda x: jnp.sum(x))

    with pytest.raises(ValueError, match="require matching sample-rank"):
        wrapped(jnp.ones((2, 3)))


def test_global_shape_defaults_and_context_manager():
    r"""Global shape defaults are optional and restored by their context manager."""
    av.set_auto_vmap_defaults(x=1)
    av.set_auto_vmap_default_shapes(x=2)

    wrapped = av.auto_vmap(validate_shapes=True)(lambda x: jnp.sum(x))
    np.testing.assert_allclose(np.asarray(wrapped(jnp.ones((3, 2)))), np.full(3, 2.0))

    with av.auto_vmap_default_shapes(x=3):
        wrapped_3 = av.auto_vmap(validate_shapes=True)(lambda x: jnp.sum(x))
        np.testing.assert_allclose(
            np.asarray(wrapped_3(jnp.ones((4, 3)))),
            np.full(4, 3.0),
        )

    with pytest.raises(ValueError, match="trailing sample shape"):
        wrapped(jnp.ones((3, 3)))


def test_auto_vmap_caches_wrappers_and_clear_works(monkeypatch):
    r"""Repeated calls reuse the local wrapper cache, which can be cleared."""
    wrapped = av.auto_vmap(moduli=1, tau=0, fluxes=1)(_toy_observable)

    wrapped(jnp.ones((4, 2)), jnp.arange(4.0), jnp.ones((4, 6)), scale=1.0)
    assert len(av._AUTO_VMAP_CACHE) == 1

    wrapped(jnp.ones((4, 2)), jnp.arange(4.0), jnp.ones((4, 6)), scale=1.0)
    assert len(av._AUTO_VMAP_CACHE) == 1

    calls: list[str] = []
    monkeypatch.setattr(av.jax, "clear_caches", lambda: calls.append("jax"))
    av.clear_auto_vmap_caches(clear_jax=True)

    assert len(av._AUTO_VMAP_CACHE) == 0
    assert calls == ["jax"]


def test_auto_vmap_snapshots_mutable_static_arguments():
    r"""Cached wrappers should not observe later mutations of static config."""

    @av.auto_vmap(x=1, jit=False)
    def shifted_sum(x, shifts):
        return jnp.sum(x) + shifts[0]

    shifts = [3.0]
    np.testing.assert_allclose(np.asarray(shifted_sum(jnp.ones((2, 2)), shifts)), [5.0, 5.0])

    shifts[0] = 100.0
    np.testing.assert_allclose(
        np.asarray(shifted_sum(jnp.ones((2, 2)), [3.0])),
        [5.0, 5.0],
    )
