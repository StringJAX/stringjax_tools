"""Tests for :func:`stringjax_tools.run_in_subprocess`."""

from __future__ import annotations

import copy
import os
import pickle
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from stringjax_tools import ChildProcessFailed, run_in_subprocess
from stringjax_tools.process import _target_name

pytestmark = pytest.mark.skipif(not hasattr(os, "killpg"), reason="POSIX only")

HELPERS = textwrap.dedent(
    """
    import os, subprocess, sys, time, warnings

    def add(a, b=0):
        return a + b

    def fail():
        raise ValueError("bad input")

    class Unprintable(Exception):
        def __str__(self):
            raise RuntimeError("no str")

    def fail_unprintably():
        raise Unprintable()

    def sleep_then(seconds, value):
        time.sleep(seconds)
        return value

    def write_pid_and_sleep(pid_file, seconds):
        with open(pid_file, "w") as f:
            f.write(str(os.getpid()))
        time.sleep(seconds)

    def _spawn(pid_file, seconds):
        grandchild = subprocess.Popen(
            [sys.executable, "-c", f"import time; time.sleep({seconds})"])
        with open(pid_file, "w") as f:
            f.write(str(grandchild.pid))

    def spawn_grandchild_and_sleep(pid_file, seconds):
        _spawn(pid_file, seconds)
        time.sleep(seconds)

    def spawn_grandchild_and_return(pid_file, seconds):
        _spawn(pid_file, seconds)
        return "returned"

    class SubDeprecation(DeprecationWarning):
        pass

    def warn_and_return(value):
        warnings.warn("careful", RuntimeWarning)
        warnings.warn("old", SubDeprecation)
        return value

    def unpicklable():
        return lambda: None

    def loaded_modules():
        return sorted(m for m in sys.modules
                      if m.split(".")[0] in ("jax", "jaxlib", "stringjax_tools", "numpy"))

    def environment(name):
        return os.environ.get(name)

    def hard_exit():
        os._exit(3)

    class Holder:
        @staticmethod
        def double(x):
            return 2 * x

        @classmethod
        def name(cls):
            return cls.__name__

        def method(self):
            return self
    """
)


@pytest.fixture()
def helpers(tmp_path: Path) -> Path:
    (tmp_path / "sjt_helpers.py").write_text(HELPERS)
    return tmp_path


@pytest.fixture()
def reaper():
    """Collect PIDs a test starts; kill whatever is still alive at the end, pass or fail."""
    pids: list[int] = []
    yield pids
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # a zombie still answers kill(0); count it as dead
    state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    return bool(state.stdout.strip()) and not state.stdout.strip().startswith("Z")


def _wait_dead(pid: int, seconds: float = 10.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return False


def _read_pid(path: Path, seconds: float = 20.0) -> int:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if path.exists() and path.read_text().strip():
            return int(path.read_text())
        time.sleep(0.05)
    raise AssertionError(f"{path} was never written")


def test_returns_value(helpers):
    assert run_in_subprocess("sjt_helpers:add", (2,), {"b": 3}, extra_paths=helpers) == 5


def test_stdlib_target_and_qualified_names(helpers):
    assert run_in_subprocess("math:factorial", (5,), timeout=30) == 120
    assert run_in_subprocess("sjt_helpers:Holder.double", (4,), extra_paths=[helpers]) == 8
    assert run_in_subprocess("sjt_helpers:Holder.name", extra_paths=[helpers]) == "Holder"


def test_child_imports_neither_jax_nor_this_package(helpers):
    assert run_in_subprocess("sjt_helpers:loaded_modules", extra_paths=helpers) == []


def test_exception_in_child(helpers):
    with pytest.raises(ChildProcessFailed) as info:
        run_in_subprocess("sjt_helpers:fail", extra_paths=helpers)
    err = info.value
    assert isinstance(err, RuntimeError)
    assert err.exc_type == "ValueError"
    assert "bad input" in str(err)
    assert "raise ValueError" in err.child_traceback
    with pytest.raises(ChildProcessFailed) as info:
        run_in_subprocess("sjt_helpers:fail_unprintably", extra_paths=helpers)
    assert info.value.exc_type == "Unprintable"


def test_child_process_failed_survives_pickling_and_copying():
    err = ChildProcessFailed(
        "boom", target="m:f", exc_type="ValueError", child_traceback="tb", returncode=None
    )
    for clone in (pickle.loads(pickle.dumps(err)), copy.copy(err)):
        assert type(clone) is ChildProcessFailed and str(clone) == "boom"
        assert (clone.target, clone.exc_type, clone.child_traceback) == ("m:f", "ValueError", "tb")


def test_unpicklable_result_and_missing_target(helpers):
    with pytest.raises(ChildProcessFailed) as info:
        run_in_subprocess("sjt_helpers:unpicklable", extra_paths=helpers)
    assert info.value.exc_type in ("PicklingError", "AttributeError")
    with pytest.raises(ChildProcessFailed) as info:
        run_in_subprocess("sjt_helpers:no_such_function", extra_paths=helpers)
    assert info.value.exc_type == "AttributeError"


def test_child_exit_without_result(helpers):
    with pytest.raises(ChildProcessFailed) as info:
        run_in_subprocess("sjt_helpers:hard_exit", extra_paths=helpers)
    assert info.value.returncode == 3


def test_timeout_kills_child_and_grandchild(helpers, tmp_path, reaper):
    pid_file = tmp_path / "grandchild.pid"
    start = time.monotonic()
    with pytest.raises(TimeoutError, match="exceeded"):
        run_in_subprocess(
            "sjt_helpers:spawn_grandchild_and_sleep",
            (str(pid_file), 60),
            timeout=5,
            extra_paths=helpers,
        )
    assert time.monotonic() - start < 30
    grandchild = _read_pid(pid_file)
    reaper.append(grandchild)
    assert _wait_dead(grandchild), "the grandchild survived the timeout"


def test_normal_return_leaves_no_grandchild(helpers, tmp_path, reaper):
    pid_file = tmp_path / "grandchild.pid"
    assert (
        run_in_subprocess(
            "sjt_helpers:spawn_grandchild_and_return", (str(pid_file), 60), extra_paths=helpers
        )
        == "returned"
    )
    grandchild = _read_pid(pid_file)
    reaper.append(grandchild)
    assert _wait_dead(grandchild, 5), "a process the target started outlived the call"


def test_fast_call_within_timeout(helpers):
    assert (
        run_in_subprocess("sjt_helpers:sleep_then", (0.1, "done"), timeout=30, extra_paths=helpers)
        == "done"
    )


def test_warnings_are_reemitted_with_builtin_categories(helpers):
    with pytest.warns(Warning) as record:
        assert run_in_subprocess("sjt_helpers:warn_and_return", (7,), extra_paths=helpers) == 7
    categories = {str(w.message).split(" (")[0]: w.category for w in record}
    assert categories == {"careful": RuntimeWarning, "old": DeprecationWarning}


def test_env_overrides(helpers, monkeypatch):
    monkeypatch.setenv("SJT_PROBE", "parent")
    assert (
        run_in_subprocess("sjt_helpers:environment", ("SJT_PROBE",), extra_paths=helpers)
        == "parent"
    )
    assert (
        run_in_subprocess(
            "sjt_helpers:environment",
            ("SJT_PROBE",),
            env={"SJT_PROBE": "child"},
            extra_paths=helpers,
        )
        == "child"
    )


def _module_level(x):
    return x


def test_target_names():
    import math

    assert _target_name(math.factorial) == "math:factorial"
    assert _target_name(_module_level) == f"{__name__}:_module_level"
    with pytest.raises(ValueError, match="lambda or a local function"):
        _target_name(lambda: None)

    def local():
        pass

    with pytest.raises(ValueError, match="lambda or a local function"):
        _target_name(local)

    class Bound:
        def method(self):
            pass

        @classmethod
        def factory(cls):
            pass

    with pytest.raises(ValueError, match="bound to an instance"):
        _target_name(Bound().method)
    with pytest.raises(ValueError, match="lambda or a local function"):
        _target_name(Bound.factory)  # a classmethod passes the bound check; Bound is local
    for bad in ("math", ":f", "math:"):
        with pytest.raises(ValueError, match="module:function"):
            _target_name(bad)
    with pytest.raises(ValueError, match="__main__"):
        _target_name("__main__:f")
    with pytest.raises(TypeError):
        _target_name(3)


@pytest.mark.parametrize("timeout", [0, -1.0])
def test_rejects_non_positive_timeout(timeout):
    with pytest.raises(ValueError, match="timeout"):
        run_in_subprocess("math:factorial", (3,), timeout=timeout)


def test_rejects_string_args():
    with pytest.raises(TypeError, match="args"):
        run_in_subprocess("math:factorial", "3")


def _parent_script(helpers: Path, pid_file: Path, *, fork_file: Path | None = None) -> str:
    src = Path(__file__).resolve().parents[1] / "src"
    fork = ""
    if fork_file is not None:  # fork (no exec) while the call runs, as multiprocessing may
        fork = textwrap.dedent(
            f"""
            import os, threading, time

            def fork_later():
                while not os.path.exists({str(pid_file)!r}):
                    time.sleep(0.05)
                pid = os.fork()
                if pid == 0:
                    time.sleep(60)
                    os._exit(0)
                with open({str(fork_file)!r}, "w") as f:
                    f.write(str(pid))

            threading.Thread(target=fork_later, daemon=True).start()
            """
        )
    return (
        textwrap.dedent(
            f"""
        import os, signal, sys
        # a process started in the background of a non-interactive shell inherits SIGINT as
        # ignored; the test is about the parent under Python's default handler
        signal.signal(signal.SIGINT, signal.default_int_handler)
        sys.path[:0] = [{str(src)!r}]
        from stringjax_tools.process import run_in_subprocess
        """
        )
        + fork
        + textwrap.dedent(
            f"""
        try:
            run_in_subprocess("sjt_helpers:write_pid_and_sleep", ({str(pid_file)!r}, 120),
                              extra_paths=[{str(helpers)!r}])
        except KeyboardInterrupt:
            # the child must already be killed and reaped when the interrupt arrives here
            child = int(open({str(pid_file)!r}).read())
            try:
                os.kill(child, 0)
            except ProcessLookupError:
                sys.exit(42)
            sys.exit(43)
        """
        )
    )


def _child_dirs(tmpdir: Path) -> list[Path]:
    return sorted(tmpdir.glob("stringjax_tools_child_*"))


def test_parent_killed_hard_leaves_no_child_and_no_files(helpers, tmp_path, reaper):
    pid_file = tmp_path / "child.pid"
    tmpdir = tmp_path / "tmpdir"
    tmpdir.mkdir()
    parent = subprocess.Popen(
        [sys.executable, "-c", _parent_script(helpers, pid_file)],
        env={**os.environ, "TMPDIR": str(tmpdir)},
    )
    reaper.append(parent.pid)
    child = _read_pid(pid_file)
    reaper.append(child)
    assert _child_dirs(tmpdir), "the call's temporary directory was not where expected"
    parent.kill()  # SIGKILL: no exception handler runs in the parent
    parent.wait()
    assert _wait_dead(child), "the child outlived its SIGKILLed parent"
    deadline = time.monotonic() + 5
    while _child_dirs(tmpdir) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _child_dirs(tmpdir), "the temporary directory outlived the parent"


def test_fork_in_parent_does_not_keep_the_child_alive(helpers, tmp_path, reaper):
    pid_file, fork_file = tmp_path / "child.pid", tmp_path / "fork.pid"
    parent = subprocess.Popen(
        [sys.executable, "-c", _parent_script(helpers, pid_file, fork_file=fork_file)]
    )
    reaper.append(parent.pid)
    child = _read_pid(pid_file)
    reaper.append(child)
    forked = _read_pid(fork_file)
    reaper.append(forked)
    parent.kill()
    parent.wait()
    assert _alive(forked), "the forked copy of the parent should still be running"
    assert _wait_dead(child), "a forked copy of the parent kept the child alive"


def test_keyboard_interrupt_in_parent_kills_child(helpers, tmp_path, reaper):
    pid_file = tmp_path / "child.pid"
    parent = subprocess.Popen([sys.executable, "-c", _parent_script(helpers, pid_file)])
    reaper.append(parent.pid)
    child = _read_pid(pid_file)
    reaper.append(child)
    parent.send_signal(signal.SIGINT)
    # 42: the interrupt propagated and the child was already gone; 43: it was still alive
    assert parent.wait(timeout=30) == 42
