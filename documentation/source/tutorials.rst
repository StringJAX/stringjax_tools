Tutorial Notebooks
==================

The tutorial notebooks are short executable examples for the main helper
families in ``stringjax_tools``.  They are intended as API orientation rather
than performance benchmarks.

Recommended path
----------------

1. Start with automatic vectorisation if you want a function to accept either
   one sample or a paired leading-axis batch.
2. Use the manual vmap helpers when the batching axes are already explicit.
3. Read the JIT and cache notebooks when tuning compilation behavior.
4. Use the pytree notebook when registering stateful model classes.

.. toctree::
   :maxdepth: 1

   notebooks/01_auto_vectorise
   notebooks/02_manual_vmap_helpers
   notebooks/03_jit_helpers
   notebooks/04_compilation_cache
   notebooks/05_pytree_policies

