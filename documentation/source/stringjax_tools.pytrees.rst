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
     - Ignore during flattening and restore with ``ignore_defaults`` if eager
       access after reconstruction is expected.
     - lazy samplers, memoised eigensystems, scratch dictionaries

Do not put semantic state in ``ignore_keys`` or ``ignore_defaults``.  A common
robust pattern is to keep user configuration as static auxiliary data and
ignore only the cache built from that configuration.  Use
``ignore_defaults={"_sampler": None, "_cache": dict}`` when ignored caches must
exist after reconstruction; callable defaults are factories for fresh mutable
objects.

Static and ignored attribute names must be disjoint.  Static values are
validated by default: they must be hashable and compare to a scalar, self-equal
boolean.  This rejects array-like metadata and ``NaN``-like values before they
enter JAX auxiliary data.

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
