stringjax_tools.process
=======================

.. currentmodule:: stringjax_tools.process

.. automodule:: stringjax_tools.process

Use :func:`run_in_subprocess` when a call must be stopped after a deadline, in
particular one that runs compiled code or starts its own worker processes.  A
timer that interrupts the main thread cannot stop such a call, and does nothing
at all once a library has installed its own SIGINT handler.

.. autosummary::
    :toctree: _autosummary

    run_in_subprocess
    ChildProcessFailed
