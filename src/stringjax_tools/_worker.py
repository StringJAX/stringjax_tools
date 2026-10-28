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

"""Child process of :func:`stringjax_tools.process.run_in_subprocess`.

Run as a script, ``python _worker.py <task> <result> <lifeline fd>``, so that it does not import
the ``stringjax_tools`` package (nor JAX), only the module of the target. The task file holds two
pickles: ``(target, sys.path)``, where ``target`` is ``"module:qualname"``, and then
``(args, kwargs)``, unpickled after ``sys.path`` is set; the file is removed once read. The
outcome is written atomically to ``<result>``: ``("ok", value, warnings)`` or ``("error",
exception type name, message, traceback)``; then the worker kills its process group, so nothing
the target started outlives the call. A daemon thread reads the lifeline pipe from the parent; at
end of file (the parent is gone) it removes the temporary directory and kills the process group.
The ``__main__`` guard is required: a target that starts workers with ``multiprocessing`` under
``spawn`` (the macOS default) re-imports this file.
"""

import sys

# Python puts this script's directory (inside the package) first on the path; drop it before
# anything else is imported, so that no module of the package can shadow a top-level one.
del sys.path[0]

import importlib  # noqa: E402
import os  # noqa: E402
import pickle  # noqa: E402
import shutil  # noqa: E402
import signal  # noqa: E402
import threading  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402


def _watch_lifeline(fd, tmp_dir):
    """Once the parent's end of the pipe closes: remove ``tmp_dir``, kill this process group."""
    try:
        while os.read(fd, 1):
            pass
    except OSError:
        pass
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.killpg(0, signal.SIGKILL)


def _resolve(target):
    module_name, function_name = target.split(":", 1)
    obj = importlib.import_module(module_name)
    for part in function_name.split("."):
        obj = getattr(obj, part)
    return obj


def _describe(exc):
    try:
        return str(exc)
    except Exception:  # an exception whose __str__ fails
        return f"<{type(exc).__name__}: str() failed>"


def _flush():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass


def main(task_path, result_path, lifeline_fd):
    threading.Thread(
        target=_watch_lifeline, args=(lifeline_fd, os.path.dirname(task_path)), daemon=True
    ).start()
    try:
        with open(task_path, "rb") as f:
            target, path = pickle.load(f)
            sys.path[:] = path
            args, kwargs = pickle.load(f)
        os.remove(task_path)
        function = _resolve(target)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = function(*args, **kwargs)
        records = [(_describe(w.message), [c.__name__ for c in w.category.__mro__]) for w in caught]
        payload = pickle.dumps(("ok", value, records))
    except BaseException as e:  # report everything to the parent, including an interrupt here
        payload = pickle.dumps(("error", type(e).__name__, _describe(e), traceback.format_exc()))
    with open(result_path + ".tmp", "wb") as f:
        f.write(payload)
    os.replace(result_path + ".tmp", result_path)
    _flush()
    os.killpg(0, signal.SIGKILL)  # this process and anything the target left running


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
