stringjax_tools.cache
=====================

.. currentmodule:: stringjax_tools.cache

.. automodule:: stringjax_tools.cache

Compilation-cache configuration is deliberately opt-in and should be called
before the first JAX compilation in a Python process.  The feature tour in
:doc:`notebooks/00_crucial_features` keeps this setup disabled by default and
shows where it belongs in a real script.

.. autosummary::
    :toctree: _autosummary

    configure_compilation_cache
