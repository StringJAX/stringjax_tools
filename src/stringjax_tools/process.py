# Copyright 2023-2026 Andreas Schachner
#
# This file is part of StringJAX Tools.
#
# StringJAX Tools is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# StringJAX Tools is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with StringJAX Tools. If not, see <https://www.gnu.org/licenses/>.

"""Run a function in a separate Python process with an enforced timeout."""

from __future__ import annotations

import builtins
import inspect
import os
import pickle
import signal
import subprocess
import sys
import tempfile
import threading
import warnings
from collections.abc import Callable, Iterable, Mapping
from typing import Any

__all__ = ["ChildProcessFailed", "run_in_subprocess"]

_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_worker.py")

# Write ends of the lifeline pipes of calls in progress. A process forked from the parent
# (``os.fork`` without ``exec``, e.g. ``multiprocessing`` with the ``fork`` start method)
# would otherwise hold a copy, and the child would not notice that the parent has died.
_open_lifelines: set[int] = set()
_lifelines_lock = threading.Lock()


def _close_lifelines_in_forked_child() -> None:
    for fd in list(_open_lifelines):
        try:
            os.close(fd)
        except OSError:
            pass
    _open_lifelines.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_close_lifelines_in_forked_child)


class ChildProcessFailed(RuntimeError):
    r"""
    Raised by :func:`run_in_subprocess` when the call did not produce a result.

    Attributes:
        target: The ``"module:function"`` that was run.
        exc_type: Name of the exception raised in the child; for an unreadable result, of the
            exception raised while reading it in the parent; ``None`` if the child exited
            without a result.
        child_traceback: The child's formatted traceback, or ``None``.
        returncode: The child's exit code if it exited without a result, otherwise ``None``.
    """

    # Every argument after ``message`` has a default: unpickling and copying rebuild the
    # exception from ``args`` alone and then restore the attributes, e.g. when it comes back
    # from a ``ProcessPoolExecutor`` worker.
    def __init__(
        self,
        message: str,
        *,
        target: str = "",
        exc_type: str | None = None,
        child_traceback: str | None = None,
        returncode: int | None = None,
    ) -> None:
        super().__init__(message)
        self.target = target
        self.exc_type = exc_type
        self.child_traceback = child_traceback
        self.returncode = returncode


def _target_name(target: str | Callable[..., Any]) -> str:
    """Return ``"module:qualname"`` for ``target``, checking that a child can import it."""
    if isinstance(target, str):
        module_name, sep, function_name = target.partition(":")
        if not (sep and module_name and function_name):
            raise ValueError(f"target must be 'module:function', got {target!r}")
    elif callable(target):
        if inspect.ismethod(target) and not isinstance(target.__self__, type):
            raise ValueError(
                f"{target!r} is bound to an instance, which the child would not receive; pass "
                "the instance as an argument to a module-level function"
            )
        module_name = getattr(target, "__module__", None) or ""
        function_name = getattr(target, "__qualname__", None) or ""
        if not (module_name and function_name):
            raise ValueError(f"cannot locate {target!r} by module and name")
        if "<" in function_name:
            raise ValueError(
                f"{module_name}.{function_name} is a lambda or a local function, which a child "
                "process cannot import; use a module-level function"
            )
    else:
        raise TypeError(f"target must be a string or a callable, got {type(target).__name__}")
    if module_name == "__main__":
        raise ValueError(
            "a function defined in __main__ cannot be imported by the child process; move it "
            "into an importable module"
        )
    return f"{module_name}:{function_name}"


def _kill_session(proc: subprocess.Popen) -> None:
    """Kill the child's whole session (it leads its own process group) and reap the child."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()  # not (or no longer) a group leader: at least the child itself
    proc.wait()


def _warning_category(mro_names: list[str]) -> type[Warning]:
    """The first built-in warning class in a child warning's class hierarchy."""
    for name in mro_names:
        category = getattr(builtins, name, None)
        if isinstance(category, type) and issubclass(category, Warning):
            return category
    return UserWarning


def _reraise_child_warnings(records: list[tuple[str, list[str]]], target: str) -> None:
    """Re-emit warnings recorded in the child, so that the caller's warning filters apply."""
    for message, mro_names in records:
        warnings.warn(
            f"{message} (raised in the child process running {target})",
            _warning_category(mro_names),
            stacklevel=3,
        )


def run_in_subprocess(
    target: str | Callable[..., Any],
    args: tuple = (),
    kwargs: Mapping[str, Any] | None = None,
    *,
    timeout: float | None = None,
    extra_paths: Iterable[str | os.PathLike[str]] | str | os.PathLike[str] = (),
    env: Mapping[str, str] | None = None,
) -> Any:
    r"""
    Run ``target(*args, **kwargs)`` in a fresh Python process and return its value.

    Unlike a timer that interrupts the main thread, the timeout is enforced: when it expires,
    the child and every process in its process group are killed. A simulated interrupt
    (``_thread.interrupt_main``) acts only under Python's default SIGINT handler, so it does
    nothing once a library has installed its own (CYTools, through cysignals, does), and it
    cannot stop work running in compiled code or in worker processes.

    The child runs this module's worker script in a new session (``start_new_session=True``), so
    it and every process it starts form one process group, which the terminal's Ctrl-C does not
    reach; a process that starts a session of its own leaves the group and is not covered. The
    worker does not import ``stringjax_tools`` or JAX; it imports only the module of ``target``
    (and what unpickling the arguments needs). Arguments and result travel as pickles through
    temporary files, so they must be picklable. The child gets the parent's ``sys.path`` (with
    ``extra_paths`` in front, set before the arguments are unpickled), ``stdin`` from
    ``/dev/null``, the parent's ``stdout`` and ``stderr``, and a private temporary directory
    (``TMPDIR``).

    - When ``target`` returns or raises, the worker writes the outcome and then kills its process
      group, so no process it started outlives the call.
    - On ``timeout`` (seconds, counted from the start of the child) the whole group is killed and
      ``TimeoutError`` is raised; a result the child had finished writing by then is returned
      instead.
    - On ``KeyboardInterrupt`` in the parent, or any other exception while waiting, the group is
      killed and the exception re-raised.
    - If the parent dies without an exception (SIGKILL, SIGTERM, a kernel restart), the child
      notices that its end of a pipe from the parent has closed, removes its temporary directory
      and kills its own group, so no worker is left running.
    - An exception in the child, including one while unpickling the arguments or pickling the
      result, raises :class:`ChildProcessFailed` with the child's exception type, message and
      traceback.
    - Warnings raised in the child are re-emitted in the parent when a result is returned, each
      with the nearest built-in category in its class hierarchy (``UserWarning`` if none); if
      the caller turns warnings into errors, the first one raises and the result is lost.

    POSIX only (process groups, ``os.killpg``). Starting the child costs a Python interpreter
    start-up plus the imports of ``target``'s module, so the helper suits calls that take seconds
    or longer. For example:

    .. code-block:: python

        from stringjax_tools import run_in_subprocess

        value = run_in_subprocess("math:factorial", (20,), timeout=10)

    Args:
        target: ``"module:function"`` or a module-level function (a ``"module:Class.method"``
            qualified name also works, for static and class methods). Lambdas, local functions,
            methods bound to an instance and functions defined in ``__main__`` are rejected,
            since the child cannot import them.
        args: Positional arguments (a tuple or list).
        kwargs: Keyword arguments.
        timeout: Seconds before the child is killed; ``None`` waits indefinitely.
        extra_paths: Directories prepended to the child's ``sys.path`` (a single path may be
            given on its own).
        env: Variables set in the child's environment, overriding the parent's (except
            ``TMPDIR``, which is always the call's private directory).

    Returns:
        Any: The value returned by ``target``.

    Raises:
        TimeoutError: If the child did not finish within ``timeout`` seconds.
        ChildProcessFailed: If the child raised, exited without a result, or wrote a result
            the parent cannot read (a subclass of ``RuntimeError``).
        ValueError: If ``target`` is not of a form a child can import, or ``timeout`` is not
            positive.
        TypeError: If ``target`` is neither a string nor a callable, or ``args`` is a string.
        NotImplementedError: On a platform without process groups.
        RuntimeError: If no Python interpreter is known (``sys.executable`` is empty).

    Pickling errors of the arguments are raised in the parent, before the child starts.
    """
    if not hasattr(os, "killpg"):
        raise NotImplementedError("run_in_subprocess needs POSIX process groups (os.killpg)")
    if not sys.executable:
        raise RuntimeError("sys.executable is empty: no Python interpreter to start the child")
    name = _target_name(target)
    if isinstance(args, (str, bytes)):
        raise TypeError("args must be a tuple or list of arguments, not a string")
    if timeout is not None and not timeout > 0:
        raise ValueError(f"timeout must be positive or None, got {timeout!r}")
    if isinstance(extra_paths, (str, os.PathLike)):
        extra_paths = (extra_paths,)
    paths = [os.fspath(p) for p in extra_paths] + list(sys.path)

    with tempfile.TemporaryDirectory(prefix="stringjax_tools_child_") as tmp:
        task_path = os.path.join(tmp, "task.pkl")
        result_path = os.path.join(tmp, "result.pkl")
        with open(task_path, "wb") as f:
            # the import path first: the worker sets it before it unpickles the arguments
            pickle.dump((name, paths), f)
            pickle.dump((tuple(args), dict(kwargs or {})), f)
        child_env = dict(os.environ)
        child_env.update(env or {})
        child_env["TMPDIR"] = tmp  # files left by a killed worker go with tmp
        # the lifeline: the child kills its group when the parent's end closes (parent gone)
        with _lifelines_lock:
            lifeline_read, lifeline_write = os.pipe()
            _open_lifelines.add(lifeline_write)
        try:
            try:
                proc = subprocess.Popen(
                    [sys.executable, _WORKER, task_path, result_path, str(lifeline_read)],
                    start_new_session=True,
                    env=child_env,
                    stdin=subprocess.DEVNULL,
                    pass_fds=(lifeline_read,),
                )
            finally:
                os.close(lifeline_read)
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                _kill_session(proc)
                if not os.path.exists(result_path):  # the result is written atomically
                    raise TimeoutError(
                        f"{name} exceeded {timeout} s; the child process and its workers were "
                        "stopped"
                    ) from None
            except BaseException:
                _kill_session(proc)
                raise
        finally:
            with _lifelines_lock:
                _open_lifelines.discard(lifeline_write)
                os.close(lifeline_write)
        if not os.path.exists(result_path):
            raise ChildProcessFailed(
                f"{name}: the child process exited with code {proc.returncode} without a result",
                target=name,
                returncode=proc.returncode,
            )
        try:
            with open(result_path, "rb") as f:
                outcome = pickle.load(f)
        except Exception as e:
            raise ChildProcessFailed(
                f"{name}: the child's result could not be read ({type(e).__name__}: {e})",
                target=name,
                exc_type=type(e).__name__,
            ) from e

    if outcome[0] == "ok":
        _reraise_child_warnings(outcome[2], name)
        return outcome[1]
    _, exc_type, message, child_traceback = outcome
    raise ChildProcessFailed(
        f"{name} failed in the child process: {exc_type}: {message}\n{child_traceback}",
        target=name,
        exc_type=exc_type,
        child_traceback=child_traceback,
    )
