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

"""Reusable JAX transform helpers for StringJAX packages."""

from .auto_vectorise import (
    ArgSpec,
    auto_vmap,
    auto_vmap_default_shapes,
    auto_vmap_defaults,
    clear_auto_vmap_caches,
    get_auto_vmap_default_shapes,
    get_auto_vmap_defaults,
    reset_auto_vmap_default_shapes,
    reset_auto_vmap_defaults,
    set_auto_vmap_default_shapes,
    set_auto_vmap_defaults,
)
from .cache import configure_compilation_cache
from .jit import is_static, jit_with_dynamic_static_args, jit_with_static_args
from .pytrees import (
    PytreePolicy,
    flatten_func,
    make_pytree_flatteners,
    unflatten_func_class,
)
from .vmap import clear_vmap_caches, vmapping_func, vmapping_func_cached

__version__ = "0.1.0.dev0"

__all__ = [
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
]

