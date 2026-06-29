stringjax_tools.pytrees
=======================

.. currentmodule:: stringjax_tools.pytrees

.. automodule:: stringjax_tools.pytrees

The package provides the pytree mechanics only.  Consuming packages should
define their own static and ignored attribute names locally, usually through a
shared :class:`PytreePolicy`.

State classification
--------------------

JAX rebuilds custom pytree objects through the registered unflatten function;
it does not call ``__init__``.  For stateful scientific model classes, classify
each attribute before registration:

.. list-table::
   :header-rows: 1

   * - Attribute kind
     - Typical treatment
     - Examples
   * - Static configuration
     - Preserve with ``static_keys`` if hashable and immutable.
     - model IDs, dimensions, user-supplied scalar bounds
   * - Traced numerical data
     - Leave as dynamic pytree children.
     - JAX arrays, differentiable state, model tensors
   * - Recomputable eager cache
     - Ignore during flattening and restore to a safe default in a custom
       unflatten wrapper if eager access after reconstruction is expected.
     - lazy samplers, memoised eigensystems, scratch dictionaries

Do not put semantic state in ``ignore_keys``.  Ignored attributes are absent on
objects reconstructed by the base :class:`PytreePolicy`; this is safe only when
the reconstructed object never reads that attribute or a class-specific
unflatten wrapper restores an appropriate default.  A common robust pattern is
to keep user configuration as static auxiliary data and ignore only the cache
built from that configuration.

The preferred future policy shape is an explicit restored-cache mechanism such
as ``ignore_defaults={"_sampler": None, "_cache": dict}``, where callable
defaults are factories for fresh mutable objects.  Until that is part of the
public API, implement the restore step in the class-specific unflatten function
when needed.

Pytree policies
---------------

.. autosummary::
    :toctree: _autosummary
    :template: custom-class-template.rst

    PytreePolicy

.. autosummary::
    :toctree: _autosummary

    flatten_func
    unflatten_func_class
    make_pytree_flatteners
