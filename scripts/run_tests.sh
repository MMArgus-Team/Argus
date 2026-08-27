#!/usr/bin/env bash
# Canonical test runner for argus-agent. Run this instead of calling
# `pytest` directly to guarantee your local run matches CI behavior.
#
# What this script enforces:
#   * Per-file isolation via scripts/run_tests_parallel.py — each test
#     file runs in its own freshly-spawned `python -m pytest <file>`
#     subprocess. No xdist, no shared workers, no module-level leakage
#     between files.
#   * TZ=UTC, LANG=C.UTF-8, PYTHONHASHSEED=0 (deterministic)
#   * Env vars blanked (conftest.py also does this, but this
#     is belt-and-suspenders for anyone running pytest outside our
#     conftest path — e.g. on a single file)
#   * Proper venv activation (probes .venv, venv, then ~/.argus/...;
#     accepts BOTH POSIX `bin/activate` and Windows `Scripts/activate`
#     layouts, so it works from Git Bash / MSYS on Windows too).
#
# Usage:
#   scripts/run_tests.sh                            # full suite
#   scripts/run_tests.sh -j 4                       # cap parallelism
#   scripts/run_tests.sh tests/agent/               # discover only here
#   scripts/run_tests.sh tests/agent/ tests/acp/    # multiple roots
#   scripts/run_tests.sh tests/foo.py               # single file
#   scripts/run_tests.sh tests/foo.py -- --tb=long  # path + pytest args
#   scripts/run_tests.sh -- -v --tb=long            # pytest args only
#
# Everything after a literal '--' is passed through to each per-file
# pytest invocation. Positional path arguments before '--' override
# the default discovery root (tests/).
#
# Windows without Git Bash? Use the native PowerShell twin:
#   powershell -ExecutionPolicy Bypass -File scripts/run_tests.ps1
# (same hermetic env + per-file isolation, no bash required).
#
# Before first use the venv needs the dev/test extras (pytest, ...):
#   uv sync --extra all --extra dev        # or: uv pip install -e ".[all,dev]"

set -euo pipefail

# ── Locate repo root ────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Activate venv ───────────────────────────────────────────────────────────
# Accept both POSIX and Windows venv layouts: the interpreter lives in
# `bin/python` on POSIX and `Scripts/python.exe` on Windows. The activate
# script marker differs the same way, so probe both.
VENV=""
PYTHON=""
for candidate in "$REPO_ROOT/.venv" "$REPO_ROOT/venv" "$HOME/.argus/argus-agent/venv"; do
  if [ -f "$candidate/bin/activate" ]; then
    VENV="$candidate"
    PYTHON="$candidate/bin/python"
    break
  fi
  if [ -f "$candidate/Scripts/activate" ]; then
    VENV="$candidate"
    PYTHON="$candidate/Scripts/python.exe"
    break
  fi
done

if [ -z "$VENV" ]; then
  echo "error: no virtualenv found in $REPO_ROOT/.venv or $REPO_ROOT/venv" >&2
  echo "  create one first with:  uv sync --extra all --extra dev" >&2
  exit 1
fi
if [ ! -x "$PYTHON" ] && [ ! -f "$PYTHON" ]; then
  echo "error: venv found at $VENV but no python at $PYTHON" >&2
  exit 1
fi

# ── Dev/test dependencies present? ───────────────────────────────────────────
# pytest is a [dev] extra, not a core dependency, so a bare `uv sync` won't
# have it. Fail fast with the exact fix instead of a confusing import error
# deep inside the parallel runner.
if ! "$PYTHON" -c "import pytest" >/dev/null 2>&1; then
  echo "error: pytest is not installed in $VENV" >&2
  echo "  install the dev/test extras:" >&2
  echo "    uv sync --extra all --extra dev" >&2
  echo "  or: uv pip install -e \".[all,dev]\"" >&2
  exit 1
fi


# ── Live-gateway plugin (computed before we drop env) ───────────────────────
EXTRA_PYTHONPATH=""
EXTRA_PYTEST_PLUGINS=""
if [ -f "$HOME/.argus/pytest_live_guard.py" ]; then
  EXTRA_PYTHONPATH="$HOME/.argus"
  EXTRA_PYTEST_PLUGINS="pytest_live_guard"
fi


# ── Run in hermetic env ──────────────────────────────────────────────────────
# env -i: start with empty environment, opt-in only what we need.
# No credential var can leak — you'd have to explicitly add it here.
echo "▶ running per-file parallel test suite via run_tests_parallel.py"
echo "  (TZ=UTC LANG=C.UTF-8 PYTHONHASHSEED=0; clean env)"

cd "$REPO_ROOT"

# Windows Python ignores LANG/LC_ALL (uses the console code page, e.g. GBK)
# and crashes on non-ASCII pytest output — force UTF-8 everywhere. Also keep
# USERPROFILE/HOMEDRIVE/HOMEPATH when present: Windows os.path.expanduser
# ignores HOME and raises "Could not determine home directory" without them.
exec env -i \
  PATH="$PATH" \
  HOME="$HOME" \
  TZ=UTC \
  LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 \
  PYTHONUTF8=1 \
  PYTHONHASHSEED=0 \
  PYTHONDONTWRITEBYTECODE=1 \
  ${USERPROFILE:+USERPROFILE="$USERPROFILE"} \
  ${HOMEDRIVE:+HOMEDRIVE="$HOMEDRIVE"} \
  ${HOMEPATH:+HOMEPATH="$HOMEPATH"} \
  ${ARGUS_RUN_SLOW_PET_TESTS:+ARGUS_RUN_SLOW_PET_TESTS="$ARGUS_RUN_SLOW_PET_TESTS"} \
  ${EXTRA_PYTHONPATH:+PYTHONPATH="$EXTRA_PYTHONPATH"} \
  ${EXTRA_PYTEST_PLUGINS:+PYTEST_PLUGINS="$EXTRA_PYTEST_PLUGINS"} \
  "$PYTHON" "$SCRIPT_DIR/run_tests_parallel.py" "$@"
