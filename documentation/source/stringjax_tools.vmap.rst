stringjax_tools.vmap
====================

.. currentmodule:: stringjax_tools.vmap

.. automodule:: stringjax_tools.vmap

Cached wrappers snapshot static keyword configuration at construction time.
For predictable cache reuse, prefer immutable configuration values.
The feature tour in :doc:`notebooks/00_crucial_features` shows the intended
relationship between explicit cached ``vmap`` wrappers and ``auto_vmap``.

.. autosummary::
    :toctree: _autosummary

    vmapping_func
    vmapping_func_cached
    clear_vmap_caches
