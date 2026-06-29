StringJAX Tools
===============

**StringJAX Tools** is a small support package for reusable JAX transform
patterns used across StringJAX projects.  It collects conservative
auto-vectorisation, cached manual ``vmap``/``jit`` wrappers, static-argument
JIT helpers, explicit compilation-cache setup, and configurable pytree
registration utilities.

The package is intentionally application-agnostic.  It should not know about
flux vacua, Calabi-Yau moduli, charge conventions, or any other package-level
domain language.  StringJAX packages provide their own naming and state
policies; this package provides the JAX mechanics.

Recommended first path
----------------------

1. Read :doc:`usage` for the intended design and migration style.
2. Run through :doc:`tutorials` for executable feature examples.
3. Use :doc:`stringjax_tools` for the API reference.
4. Add package-local policies, such as pytree static and ignored keys, in the
   consuming package rather than in ``stringjax_tools``.

Reference lookup
----------------

* :ref:`genindex`
* :ref:`modindex`

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Start here

   usage
   tutorials
   stringjax_tools
