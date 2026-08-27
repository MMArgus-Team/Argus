"""Sync the project-root config.yaml into the active HERMES_HOME at startup.

The project directory's ``config.yaml`` is the single source of truth (tracked
in git, travels with the repo across machines). But every Argus reader loads
config from ``<HERMES_HOME>/config.yaml`` (platform default on Windows is
``%LOCALAPPDATA%\\argus``), and there is no config-path override. So at startup
we copy the project config over the HERMES_HOME copy — one-way, project wins.

Call ``sync_project_config()`` early in an entrypoint (before any load_config /
gateway spawn) so both the in-process reader and the spawned gateway (which
inherits HERMES_HOME) see the latest project config.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Project root = parent of hermes_cli/ (this file lives in hermes_cli/).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PROJECT_CONFIG = _PROJECT_ROOT / "config.yaml"
_PROJECT_WEIGHTS = _PROJECT_ROOT / "weights"


def sync_project_config() -> bool:
    """Copy ``<project>/config.yaml`` over ``<HERMES_HOME>/config.yaml``.

    One-way, project → HERMES_HOME, always overwrites. No-op (returns False)
    when the project config doesn't exist, so this never clobbers an existing
    HERMES_HOME config with nothing. Best-effort: any error is logged and
    swallowed (config sync must never block startup).

    Returns True when a copy happened, False otherwise.
    """
    try:
        src = _PROJECT_CONFIG
        if not src.is_file():
            logger.debug("[config-sync] no project config at %s; skip", src)
            return False
        from hermes_constants import get_config_path
        from hermes_cli.config import ensure_hermes_home

        ensure_hermes_home()
        dst = Path(get_config_path())
        # Skip a redundant copy when src/dst are the same file (e.g. someone set
        # HERMES_HOME to the project dir).
        try:
            if dst.resolve() == src.resolve():
                logger.debug("[config-sync] src == dst (%s); skip", dst)
                return False
        except Exception:
            pass
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        logger.info("[config-sync] project config → %s", dst)
        return True
    except Exception as exc:  # pragma: no cover - best-effort
        logger.warning("[config-sync] failed (non-fatal): %s", exc)
        return False


def normalize_config_shapes() -> bool:
    """Collapse ambiguous on-disk config shapes into their canonical form.

    Runs on every startup, AFTER ``sync_project_config`` (which re-copies the
    project config.yaml and therefore re-introduces the shapes we normalize).
    Each normalizer is idempotent, so a steady-state config is left untouched
    and no write happens.

    Currently one normalizer:

    * A bare ``model: {provider: custom, base_url: ...}`` gains a
      ``custom_providers`` row. Without it, that endpoint exists only as a
      picker row fabricated from ``model.base_url``, so switching the main model
      to another provider — which clears that URL — made the endpoint
      permanently unreachable from the UI. See
      ``config.normalize_bare_custom_endpoint`` for the full rationale.

    Best-effort: any failure is logged and swallowed (normalization must never
    block startup). Returns True when config was changed and saved.
    """
    try:
        from hermes_cli.config import (
            load_config,
            normalize_bare_custom_endpoint,
            save_config,
        )

        cfg = load_config()
        added = normalize_bare_custom_endpoint(cfg)
        if not added:
            return False

        # strip_defaults=False: this is a repair write on a user's real config,
        # not a fresh scaffold — don't let it prune keys that merely match a
        # default.
        save_config(cfg, strip_defaults=False)
        logger.info("[config-normalize] registered custom endpoint %r", added)
        return True
    except Exception as exc:  # pragma: no cover - best-effort
        logger.warning("[config-normalize] failed (non-fatal): %s", exc)
        return False


def sync_project_weights() -> bool:
    """Link ``<project>/weights`` → ``<HERMES_HOME>/weights`` so config.yaml's
    HERMES_HOME-relative paths resolve to the project's local model weights.

    The project directory holds the real weights (not tracked in git); every
    Argus reader resolves relative model paths under HERMES_HOME. So we link
    the two. One-way, project → HERMES_HOME.

    Link strategy (best-effort, never blocks startup):
      * If <HERMES_HOME>/weights already points at the project weights → no-op.
      * A real directory already there (not our link) → leave it, warn (don't
        clobber user data).
      * Otherwise create a link: symlink first; on Windows without symlink
        permission, fall back to a directory junction (mklink /J).

    Returns True when a usable link/target is in place, False otherwise.
    """
    try:
        src = _PROJECT_WEIGHTS
        if not src.is_dir():
            logger.debug("[weights-sync] no project weights at %s; skip", src)
            return False
        from hermes_cli.config import ensure_hermes_home, get_hermes_home

        ensure_hermes_home()
        home = Path(get_hermes_home())
        dst = home / "weights"

        # src == dst (HERMES_HOME is the project dir) → nothing to link.
        try:
            if dst.resolve() == src.resolve():
                logger.debug("[weights-sync] src == dst (%s); skip", dst)
                return True
        except Exception:
            pass

        # Already linked to our project weights?
        if dst.exists() or dst.is_symlink():
            try:
                if dst.is_symlink() and Path(os.readlink(dst)).resolve() == src.resolve():
                    return True
            except Exception:
                pass
            try:
                if dst.resolve() == src.resolve():
                    return True   # junction / already resolves to project
            except Exception:
                pass
            # A different real directory is there → don't clobber it.
            logger.warning(
                "[weights-sync] %s already exists and isn't the project weights link; "
                "leaving it. config.yaml relative model paths may not resolve to "
                "project weights.", dst)
            return False

        home.mkdir(parents=True, exist_ok=True)
        # Try a symlink first (works on macOS/Linux, and Windows w/ dev mode).
        try:
            os.symlink(str(src), str(dst), target_is_directory=True)
            logger.info("[weights-sync] symlink %s → %s", dst, src)
            return True
        except (OSError, NotImplementedError) as exc:
            if sys.platform != "win32":
                logger.warning("[weights-sync] symlink failed (non-fatal): %s", exc)
                return False
        # Windows fallback: directory junction (no admin/dev-mode needed).
        try:
            subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(dst), str(src)],
                check=True, capture_output=True, text=True)
            logger.info("[weights-sync] junction %s → %s", dst, src)
            return True
        except Exception as exc:
            logger.warning("[weights-sync] junction failed (non-fatal): %s", exc)
            return False
    except Exception as exc:  # pragma: no cover - best-effort
        logger.warning("[weights-sync] failed (non-fatal): %s", exc)
        return False
