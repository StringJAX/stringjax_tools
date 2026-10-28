.. currentmodule:: stringjax_tools


stringjax_tools
===============

Package overview
----------------

``stringjax_tools`` is organised around small, composable JAX transform helpers,
plus general-purpose infrastructure: running a call in a child process with an
enforced timeout, and recording the provenance of a result.
The modules are intentionally independent from any StringJAX application
package.

For an executable first pass through the public surface, see
:doc:`notebooks/00_crucial_features`.  The notebook demonstrates the small set
of features that downstream packages most often combine: rank-checked
``auto_vmap``, explicit cached ``vmap`` wrappers, static-argument JIT helpers,
``PytreePolicy``, and explicit persistent-compilation-cache setup.

Common imports
--------------

The most common helpers are re-exported from the package root:

.. code-block:: python

    from stringjax_tools import (
        PytreePolicy,
        auto_vmap,
        configure_compilation_cache,
        jit_with_static_args,
        vmapping_func_cached,
    )

Submodule imports remain available when a consuming package wants a narrower
namespace.

.. toctree::
    :maxdepth: 1

    stringjax_tools.auto_vectorise
    stringjax_tools.vmap
    stringjax_tools.jit
    stringjax_tools.cache
    stringjax_tools.pytrees
    stringjax_tools.process
    stringjax_tools.reproducibility
