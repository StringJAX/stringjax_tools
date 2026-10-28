"""Tests for :mod:`stringjax_tools.reproducibility`."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import stringjax_tools
from stringjax_tools import git_state, package_info, provenance

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.org", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture()
def repo_package(tmp_path: Path, monkeypatch):
    """A package ``sjt_probe_pkg`` in a fresh git repository, importable via sys.path."""
    repo = tmp_path / "repo"
    pkg = repo / "sjt_probe_pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text('__version__ = "9.9.9"\n')
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")
    monkeypatch.syspath_prepend(str(repo))
    monkeypatch.delitem(sys.modules, "sjt_probe_pkg", raising=False)
    yield repo, _git(repo, "rev-parse", "HEAD")
    sys.modules.pop("sjt_probe_pkg", None)


@needs_git
def test_git_state_clean_dirty_and_untracked(repo_package):
    repo, commit = repo_package
    init = repo / "sjt_probe_pkg" / "__init__.py"
    state = git_state(repo / "sjt_probe_pkg", init)
    assert state == {"commit": commit, "dirty": False, "root": str(repo.resolve())}
    (repo / "untracked.txt").write_text("x")  # untracked files do not count
    assert git_state(repo)["dirty"] is False
    init.write_text('__version__ = "9.9.10"\n')
    assert git_state(repo)["dirty"] is True
    stray = repo / "stray.py"
    stray.write_text("")
    assert git_state(repo, stray) == {"commit": None, "dirty": None, "root": None}


def test_git_state_outside_a_repository(tmp_path):
    assert git_state(tmp_path) == {"commit": None, "dirty": None, "root": None}


@needs_git
def test_package_info_not_imported_then_imported(repo_package):
    repo, commit = repo_package
    info = package_info("sjt_probe_pkg")
    assert info["imported"] is False and info["module_version"] is None
    assert info["distribution"] is None and info["version_mismatch"] is None
    assert info["editable"] is None
    assert len(info["git"]["commit"]) == 40  # the full hash
    assert Path(info["path"]) == (repo / "sjt_probe_pkg").resolve()
    assert info["git"]["commit"] == commit
    assert "sjt_probe_pkg" not in sys.modules  # package_info does not import

    import sjt_probe_pkg  # noqa: F401

    assert package_info("sjt_probe_pkg")["module_version"] == "9.9.9"


def test_package_info_missing_and_bad_names():
    info = package_info("sjt_no_such_package_xyz")
    assert info["path"] is None and info["distribution"] is None
    assert info["git"] == {"commit": None, "dirty": None, "root": None}
    for bad in ("", "os.path"):
        with pytest.raises(ValueError):
            package_info(bad)


def test_package_info_of_this_package():
    info = package_info("stringjax_tools")
    assert info["imported"] is True
    assert info["module_version"] == stringjax_tools.__version__
    assert Path(info["path"]) == Path(stringjax_tools.__file__).resolve().parent
    if info["distribution_version"] is not None:
        assert info["distribution"] == "stringjax-tools"
        assert info["editable"] in (True, False, None)
        assert info["version_mismatch"] is (
            info["distribution_version"] != stringjax_tools.__version__
        )


def test_version_mismatch_is_flagged(monkeypatch):
    fake = type(sys)("sjt_fake_mod")
    fake.__version__ = "0.0.1"
    monkeypatch.setitem(sys.modules, "sjt_fake_mod", fake)
    monkeypatch.setattr(
        "stringjax_tools.reproducibility._distribution",
        lambda name, mapping=None: "stringjax-tools",
    )
    monkeypatch.setattr("stringjax_tools.reproducibility.metadata.version", lambda name: "0.0.2")
    info = package_info("sjt_fake_mod")
    assert info["version_mismatch"] is True


def test_provenance_record_is_json_safe():
    import jax

    record = provenance(["stringjax_tools", "sjt_no_such_package_xyz"], seed=3)
    json.dumps(record)
    assert set(record) == {"time", "python", "platform", "packages", "jax", "seed"}
    assert record["seed"] == 3
    assert record["jax"] == {
        "version": jax.__version__,
        "enable_x64": bool(jax.config.jax_enable_x64),
    }
    assert list(record["packages"]) == ["stringjax_tools", "sjt_no_such_package_xyz"]
    assert list(provenance("stringjax_tools")["packages"]) == ["stringjax_tools"]
    with pytest.raises(ValueError, match="overwrite"):
        provenance([], time="now")


def test_provenance_reads_the_current_x64_setting():
    """The x64 entry is JAX's setting in this process, read in a fresh interpreter."""
    src = Path(stringjax_tools.__file__).resolve().parents[1]
    script = textwrap.dedent(
        f"""
        import sys; sys.path.insert(0, {str(src)!r})
        import jax
        from stringjax_tools import provenance
        a = provenance([])["jax"]["enable_x64"]
        jax.config.update("jax_enable_x64", True)
        b = provenance([])["jax"]["enable_x64"]
        print(a, b)
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        env={"PATH": "/usr/bin:/bin"},
    ).stdout.split()
    assert out == ["False", "True"]


def test_version_is_the_same_in_every_place():
    """``__version__``, ``pyproject.toml`` and ``CITATION.cff`` agree (the release tag is checked
    against the built wheel, i.e. against ``pyproject.toml``)."""
    import re

    root = Path(__file__).resolve().parents[1]
    pyproject = re.search(r'^version\s*=\s*"([^"]+)"', (root / "pyproject.toml").read_text(), re.M)
    citation = re.search(r"^version:\s*(\S+)\s*$", (root / "CITATION.cff").read_text(), re.M)
    assert pyproject and citation
    assert pyproject.group(1) == citation.group(1) == stringjax_tools.__version__
