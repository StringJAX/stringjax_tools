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

"""Small JIT helpers for functions with Python-side static arguments."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

import jax
import numpy as np

__all__ = [
    "is_static",
    "jit_with_dynamic_static_args",
    "jit_with_static_args",
]


def jit_with_static_args(
    func: Callable[..., Any],
    static_argnums: tuple[int, ...] = (),
) -> Callable[..., Any]:
    r"""
    Wrap ``func`` with ``jax.jit`` and explicit positional static arguments.

    Args:
        func: Function to JIT-compile.
        static_argnums: Positional argument indices to mark static.

    Returns:
        JIT-compiled wrapper around ``func``.
    """

    @wraps(func)
    def call_func(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return jax.jit(call_func, static_argnums=static_argnums)


def is_static(arg: Any) -> bool:
    r"""
    Heuristic test for whether ``arg`` should be treated as JAX-static.

    Python scalars, strings, booleans and ordinary configuration objects are
    treated as static.  JAX arrays and NumPy arrays are treated as dynamic.
    """
    return not isinstance(arg, (jax.Array, np.ndarray))


def jit_with_dynamic_static_args(func: Callable[..., Any]) -> Callable[..., Any]:
    r"""
    JIT ``func`` while deciding static positional arguments at call time.

    This is convenient for diagnostics and prototypes, but production code
    should prefer :func:`jit_with_static_args` with explicit ``static_argnums``
    because changing static positions can cause repeated compilation.
    """

    @wraps(func)
    def wrapped_func(*args: Any, **kwargs: Any) -> Any:
        static_argnums = tuple(i for i, arg in enumerate(args) if is_static(arg))
        jit_func = jax.jit(func, static_argnums=static_argnums)
        return jit_func(*args, **kwargs)

    return wrapped_func
