"""``hermes mm`` subcommand parser (multimodal utilities).

Namespaced parent command for multimodal maintenance tools. Currently exposes
``hermes mm doctor`` (readiness self-check); more actions (e.g. ``mm status``)
can slot in under the same nested subparser.

Handler injected to avoid importing ``main`` (mirrors the other subcommands).
"""

from __future__ import annotations

from typing import Callable


def build_mm_parser(subparsers, *, cmd_mm: Callable) -> None:
    """Attach the ``mm`` subcommand (with nested actions) to ``subparsers``."""
    mm_parser = subparsers.add_parser(
        "mm",
        help="Multimodal utilities (readiness check, etc.)",
        description="Multimodal maintenance tools. "
        "Run `argus mm doctor` to check whether voice / deep-research / "
        "memory / tracking are ready and what's missing.",
    )
    mm_sub = mm_parser.add_subparsers(dest="mm_action")

    mm_sub.add_parser(
        "doctor",
        help="Check multimodal readiness (voice/search/memory/tracking)",
        description="Report each multimodal capability's readiness, what's "
        "missing, and how to fix it. Exit 0 if all required capabilities are "
        "ready, else 1.",
    )

    mm_parser.set_defaults(func=cmd_mm)
