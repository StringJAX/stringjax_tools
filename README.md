# StringJAX Tools

`stringjax_tools` is a small utility package for reusable JAX transform
patterns used across StringJAX projects.  It currently focuses on:

- conservative rank-checked automatic vectorisation;
- manual cached `jax.jit(jax.vmap(...))` wrappers;
- small helpers for explicit static-argument JIT compilation;
- explicit JAX persistent-compilation-cache setup;
- configurable pytree registration for stateful model classes.

The package was extracted from JAXVacua so that this machinery can be reused by
other StringJAX packages without importing flux-vacuum-specific code.

## Installation

From this directory:

```bash
pip install -e ".[dev,docs]"
```

The intended published package name is:

```bash
pip install stringjax-tools
```

The Python import name is:

```python
import stringjax_tools
```

## Documentation

The Sphinx documentation lives in `documentation/`, following the same basic
layout as JAXVacua:

```bash
make -C documentation html
```

For strict local validation:

```bash
make -C documentation html SPHINXOPTS="-W -q"
```

The feature notebooks are included under `documentation/source/notebooks/` and
are linked from the Sphinx tutorials page.

## Automatic Vectorisation

Keep the single-sample function explicit, then wrap it at the boundary where
callers may pass either one sample or a paired leading-axis batch.

```python
import jax.numpy as jnp
from stringjax_tools.auto_vectorise import auto_vmap


def observable_single(moduli, tau, fluxes, scale=1.0):
    return scale * (jnp.sum(jnp.abs(moduli) ** 2) + tau + jnp.sum(fluxes**2))


observable = auto_vmap(moduli=1, tau=0, fluxes=1)(observable_single)
```

With these declarations:

- `moduli` has sample rank `1`, so `(h,)` is one sample and `(N, h)` is a batch.
- `tau` has sample rank `0`, so `()` is one sample and `(N,)` is a batch.
- `fluxes` has sample rank `1`, so `(n,)` is one sample and `(N, n)` is a batch.

If several checked arguments are batched, all leading batch sizes must agree.
Checked arguments without a batch axis are broadcast with `in_axes=None`.

## Exact Shape Validation

Ranks decide batching.  Exact trailing dimensions are optional:

```python
class Model:
    def __init__(self, h12, n_fluxes):
        self.h12 = h12
        self.n_fluxes = n_fluxes

    @auto_vmap(
        sample_ranks={"moduli": 1, "fluxes": 1},
        sample_shapes={
            "moduli": "h12",
            "fluxes": lambda bound: (2 * bound["self"].n_fluxes,),
        },
        validate_shapes=True,
    )
    def diagnostic(self, moduli, fluxes):
        return jnp.sum(jnp.abs(moduli) ** 2) + 0.01 * jnp.sum(fluxes**2)
```

String dimensions are resolved from a bound object, usually `self`.  Derived
dimensions should be callables rather than expression strings.

## Manual Transform Helpers

When a function already has a deliberate `in_axes` structure, use:

```python
from stringjax_tools.vmap import vmapping_func_cached

batched = vmapping_func_cached(single_sample_function, in_axes=(0, None, 0))
```

Static keyword configuration is snapshotted when the cached wrapper is built.
Prefer immutable configuration objects, such as tuples, for values that should
participate in wrapper caching.

For explicit static positional arguments:

```python
from stringjax_tools.jit import jit_with_static_args

jit_f = jit_with_static_args(f, static_argnums=(2,))
```

## Pytree Policies

Each StringJAX package should define its own static and ignored attribute policy
locally.  The tool package supplies only generic registration machinery.

```python
from stringjax_tools.pytrees import PytreePolicy

JAXVACUA_PYTREE_POLICY = PytreePolicy(
    static_keys=("h11", "h12", "model_ID", "_user_Q"),
    ignore_keys=("_sampler", "_cache"),
)

JAXVACUA_PYTREE_POLICY.register(MyModelClass)
```

Semantic model state belongs in `static_keys` when it is hashable and immutable,
or in traced children when it is array-like.  Ignored attributes should be
limited to recomputable caches, scratch state, and eager-only helpers.  Since
JAX reconstruction bypasses `__init__`, ignored caches that may be read after a
round trip need an explicit restore step in a class-specific unflatten wrapper,
for example restoring `_sampler` to `None` or `_cache` with a fresh `dict`.
Do not ignore user-supplied configuration such as physical bounds or tadpole
values.
Leave `validate_static=True` unless the consuming package deliberately stores
non-hashable metadata in JAX pytree auxiliary data.

## Compilation Cache

`configure_compilation_cache(...)` is opt-in and has no import-time side
effects.  Call it before the first JAX compilation in a process:

```python
from stringjax_tools.cache import configure_compilation_cache

configure_compilation_cache(
    cache_dir=".jax-cache",
    max_size_bytes=10_000_000_000,
)
```

## License

StringJAX Tools is distributed under the GNU General Public License v3.0 or
later.
