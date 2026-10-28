# Changelog

Notable changes to StringJAX Tools, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.2.0] - 2026-10-28

### Added
- `run_in_subprocess` and `ChildProcessFailed` (`stringjax_tools.process`): run a module-level
  function in a fresh Python process with an enforced timeout. On expiry the child and every
  process in its process group are killed (`TimeoutError`); when the function returns or raises,
  the group is killed too, so nothing it started outlives the call. An exception in the child
  raises `ChildProcessFailed` (a `RuntimeError`, picklable) with its type and traceback; warnings
  are re-emitted in the parent with the nearest built-in category; if the parent dies, the child
  removes its temporary files and stops, also when the parent has forked; the function may itself
  start `multiprocessing` workers (also under `spawn`). Unlike a timer that
  interrupts the main thread, it works when a library has installed its own SIGINT handler and
  when the call runs compiled code or worker processes. POSIX only.
- `provenance`, `package_info` and `git_state` (`stringjax_tools.reproducibility`): a JSON-safe
  record of the code a result came from (distribution and module versions, a flag when they
  disagree, whether the install is editable, source directory, full git commit and dirty state,
  JAX's version and 64-bit setting), without importing the packages described.
- A release workflow (`.github/workflows/release.yml`): pushing a tag `vX.Y.Z` builds the sdist
  and wheel, runs `twine check --strict`, refuses a tag that differs from the built version, and
  publishes to PyPI through Trusted Publishing; a manual run only builds and checks.
- A feature-tour notebook (`documentation/source/notebooks/00_crucial_features.ipynb`) and its
  documentation pages (added after 0.1.0, first released here).
- This changelog and `CITATION.cff`.

### Changed
- Requires Python 3.12 or later (was 3.10), as the other StringJAX packages do; CI tests 3.12 and 3.13.
  Current JAX releases need Python 3.11 (JAX 0.7) or 3.12 (JAX 0.12); on Python 3.10 or 3.11, pip keeps
  stringjax-tools 0.1.0.
- Read the Docs builds with `fail_on_warning: true`, as CI already did (`-W`).

### Fixed
- The project URLs pointed at `github.com/StringJAX/stringjax-tools`, which does not exist; they
  now point at `github.com/StringJAX/stringjax_tools` (also in the documentation theme).

## [0.1.0] - 2026-06-30

First release: `auto_vmap` and its defaults, `vmapping_func` and `vmapping_func_cached`,
`jit_with_static_args`, `is_static`, `jit_with_dynamic_static_args`,
`configure_compilation_cache`, `PytreePolicy` and the pytree flattening helpers.
