"""Make the optional extras optional at test time too.

Every backend is an extra -- ``uv sync --extra jax`` is a documented install --
but the test files import theirs at module level, so before this file a partial
install did not skip anything: collection aborted on the first missing import
and *nothing* ran, including the checks that need no backend at all.  On a
numpy-only tree that was 5 collection errors and 0 tests.

Three layers, so that no test file has to declare anything:

1. :data:`collect_ignore` drops the modules whose *module-level* imports are
   missing, because a collection error is fatal to the whole run in a way a
   skipped test is not.  :func:`pytest_report_header` prints which and why, so
   dropping them is not silent.
2. :func:`pytest_collection_modifyitems` skips one half of a ``[warp]`` /
   ``[jax]`` parametrization when that backend is absent, leaving the other
   half to run.
3. :func:`pytest_runtest_makereport` turns a ``ModuleNotFoundError`` for a
   known extra into a skip, which covers the tests that reach a backend only at
   runtime -- through the registry's lazy imports -- and so cannot be spotted
   by looking at imports.

With the full ``--extra all`` install none of this fires and the whole suite
runs, which is the configuration the numbers in the README come from.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

# Import names (not distribution names) of every optional dependency a test can
# reach, mapped to the extra that provides it -- the message says which one to
# install.  Only these are ever converted to a skip; anything else stays a
# failure, so a genuine missing import in the package cannot hide here.
EXTRA_FOR_MODULE = {
    "warp": "warp",
    "warp_nn": "warp",
    "warp_rl": "warp",
    "jax": "jax",
    "jaxlib": "jax",
    "flax": "jax",
    "optax": "jax",
    "jax_rl": "jax",
    "stable_baselines3": "sb3",
    "torch": "sb3",
    "gymnasium": "gym",
    "Box2D": "box2d",
    "pygame": "render",
    "imageio": "record",
}

# What each test module imports at module level, and so needs merely to be
# collected.  Files absent from this mapping need nothing beyond numpy.
MODULE_LEVEL_REQUIREMENTS = {
    "test_cartpole_env.py": ("gymnasium",),
    "test_mountain_car.py": ("gymnasium",),
    "test_sb3_backend.py": ("gymnasium",),
    "test_jax_ppo.py": ("jax", "optax", "jax_rl"),
    "test_warp_ppo.py": ("warp", "warp_rl"),
}


def _missing(module: str) -> bool:
    """True if ``module`` is not installed.

    Ask about the third-party module, never about ``warp_rl`` / ``jax_rl``:
    those are directories in this repo, so ``find_spec`` finds them and returns
    a spec happily -- it locates a module without executing it, and it is
    executing ``warp_rl/__init__.py`` that would raise.  ``find_spec`` can also
    raise instead of returning None when a parent package is absent, hence the
    guard.
    """
    try:
        return importlib.util.find_spec(module) is None
    except (ImportError, ValueError):
        return True


def _install_hint(modules) -> str:
    extras = sorted({EXTRA_FOR_MODULE.get(m, m) for m in modules})
    return f"needs {', '.join(sorted(modules))} -- uv sync --extra {','.join(extras)}"


# -- layer 1: modules that cannot even be imported --------------------------

_here = pathlib.Path(__file__).parent
_unknown = [name for name in MODULE_LEVEL_REQUIREMENTS if not (_here / name).exists()]
if _unknown:  # a renamed test file must not silently stop being guarded
    raise RuntimeError(f"conftest lists test files that do not exist: {', '.join(_unknown)}")

_dropped = {name: tuple(m for m in modules if _missing(m)) for name, modules in MODULE_LEVEL_REQUIREMENTS.items()}
_dropped = {name: missing for name, missing in _dropped.items() if missing}
collect_ignore = sorted(_dropped)


def pytest_report_header() -> list[str]:
    if not _dropped:
        return []
    return ["not collected (module-level imports unavailable):"] + [
        f"  {name}: {_install_hint(missing)}" for name, missing in sorted(_dropped.items())
    ]


# -- layer 2: one half of a [warp] / [jax] parametrization ------------------


def pytest_collection_modifyitems(items) -> None:
    # the backend name is also its module name, which is the one to ask about
    absent = {backend for backend in ("warp", "jax") if _missing(backend)}
    if not absent:
        return
    for item in items:
        callspec = getattr(item, "callspec", None)
        backend = callspec.params.get("backend") if callspec else None
        if backend in absent:
            item.add_marker(pytest.mark.skip(reason=_install_hint((backend,))))


# -- layer 3: backends reached only at runtime, via the lazy registry -------


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if call.excinfo is None or report.outcome != "failed":
        return
    error = call.excinfo.value
    if not isinstance(error, ModuleNotFoundError):
        return
    name = (error.name or "").split(".")[0]
    # only a module that is genuinely absent, and only one we know is optional
    if name in EXTRA_FOR_MODULE and _missing(name):
        # (path, line, reason) is what pytest renders for a skip; item.location
        # points at the test rather than at the top of the file
        path, lineno, _ = item.location
        report.outcome = "skipped"
        report.longrepr = (path, (lineno or 0) + 1, _install_hint((name,)))
