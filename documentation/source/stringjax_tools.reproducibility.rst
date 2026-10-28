stringjax_tools.reproducibility
===============================

.. currentmodule:: stringjax_tools.reproducibility

.. automodule:: stringjax_tools.reproducibility

Store :func:`provenance` next to a result.  For an editable install the
distribution version can lag the source tree, so each package's record also
holds its source directory and that directory's git commit and dirty state.

.. autosummary::
    :toctree: _autosummary

    provenance
    package_info
    git_state
