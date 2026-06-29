# Usage Guide

`stringjax_tools` provides generic JAX mechanics for the wider StringJAX
ecosystem.  The package deliberately avoids application-specific names and
conventions: each consuming package should define its own argument-shape
defaults, pytree static keys, ignored cache attributes, and model policies.

## Automatic vectorisation

Use `auto_vmap` when a function is naturally written for one sample, but users
may pass either one sample or a paired leading-axis batch:

```python
import jax.numpy as jnp
from stringjax_tools.auto_vectorise import auto_vmap


@auto_vmap(moduli=1, tau=0, fluxes=1)
def observable(moduli, tau, fluxes):
    return jnp.sum(moduli) + tau + jnp.sum(fluxes)
```

Here `moduli` and `fluxes` are vectors, while `tau` is scalar.  Inputs with one
extra leading axis are mapped in lockstep.  Inputs without the extra axis are
broadcast with `in_axes=None`.

Exact sample shapes are optional and should be reserved for public API
boundaries, tests, and diagnostics:

```python
@auto_vmap(
    sample_ranks={"moduli": 1, "fluxes": 1},
    sample_shapes={
        "moduli": "h12",
        "fluxes": lambda bound: (2 * bound["self"].n_fluxes,),
    },
    validate_shapes=True,
)
def diagnostic(self, moduli, fluxes):
    ...
```

String dimensions are resolved from a bound object, usually `self`.  Derived
dimensions should be expressed as callables.

## Manual transform helpers

Use `vmapping_func_cached` when the batching structure is already explicit and
you want to reuse a `jax.jit(jax.vmap(...))` wrapper:

```python
from stringjax_tools.vmap import vmapping_func_cached

batched = vmapping_func_cached(single_sample_function, in_axes=(0, None, 0))
```

Static keyword configuration is snapshotted when the cached wrapper is built.
This keeps wrapper cache keys consistent even if a list or dictionary supplied
by the caller is mutated later.  Prefer immutable configuration objects in
performance-sensitive code, since mutable Python state and JAX compilation
caches are an awkward mix.

Use `jit_with_static_args` when positional static arguments are known:

```python
from stringjax_tools.jit import jit_with_static_args

jit_f = jit_with_static_args(f, static_argnums=(2,))
```

The dynamic static-argument helper is mainly for prototypes, since changing the
static argument pattern can trigger repeated compilation.

## Pytree policies

Each package should define its own static and ignored attribute policy:

```python
from stringjax_tools.pytrees import PytreePolicy

PACKAGE_PYTREE_POLICY = PytreePolicy(
    static_keys=("h11", "h12", "model_ID", "_user_Q"),
    ignore_keys=("_sampler", "_cache"),
)

PACKAGE_PYTREE_POLICY.register(MyModelClass)
```

Semantic model state belongs in `static_keys` when it is hashable and immutable,
or in traced children when it is array-like.  Ignored attributes should be
limited to recomputable caches and eager-only scratch state.  Since JAX
reconstruction bypasses `__init__`, ignored caches that may be read after a
round-trip need an explicit restore step in a class-specific unflatten wrapper,
for example restoring `_sampler` to `None` or `_cache` with a fresh `dict`.
Do not ignore user-supplied configuration such as physical bounds or tadpole
values; otherwise the reconstructed object can be missing or silently corrupt
that state.

By default, static pytree attributes are validated before they enter JAX
auxiliary data.  If a consuming package deliberately stores non-hashable
metadata as static data, set `validate_static=False` in that package's local
policy and document why this is safe.

## Compilation cache

Persistent compilation-cache setup is explicit and opt-in:

```python
from stringjax_tools.cache import configure_compilation_cache

configure_compilation_cache(cache_dir=".jax-cache", max_size_bytes=10_000_000_000)
```

Call this before the first JAX compilation in the current Python process.
