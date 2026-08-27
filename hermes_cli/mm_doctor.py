"""``hermes mm doctor`` — multimodal readiness self-check.

Loads the real on-disk multimodal Config, runs the pure
``agent.multimodal.readiness.probe_mm_readiness`` probe, and prints each
capability's state (ok / missing / broken / unknown) with the reason and a
copy-pasteable fix — reusing doctor's check_ok/check_warn/check_fail style so
it looks like the rest of ``argus doctor``.

Exit code: 0 if all REQUIRED capabilities are ready, else 1 — so it can gate CI
and scripts.
"""

from __future__ import annotations

from hermes_cli.colors import Colors, color


def _load_mm_config():
    """Build the real multimodal Config + raw nested config from disk.

    Returns ``(cfg, raw_cfg)``: the flat Config (or None) and the original
    nested config dict (or None). ``raw_cfg`` carries values that live only in
    the nested layout (e.g. auxiliary.vision.base_url). Any failure
    degrades to (None, None) so the probe still reports gaps rather than crash.
    """
    raw_cfg = None
    try:
        from hermes_cli.config import load_config
        raw_cfg = load_config() or None
    except Exception:
        raw_cfg = None
    try:
        from agent.multimodal.hermes_glue import build_config
        return build_config(), raw_cfg
    except Exception as exc:  # pragma: no cover — defensive
        print(color(
            f"  ⚠ 无法加载多模态配置 ({exc});按默认值检查。",
            Colors.YELLOW))
        return None, raw_cfg


def run_mm_doctor(args) -> int:
    """Print the multimodal readiness report. Returns process exit code."""
    from agent.multimodal.readiness import (
        BROKEN,
        MISSING,
        OK,
        UNKNOWN,
        probe_mm_readiness,
    )

    print()
    print(color("◆ 多模态就绪检查 (mm doctor)", Colors.CYAN, Colors.BOLD))
    # Readiness reflects THIS interpreter's environment (find_spec/imports run in
    # the running process). Show which Python that is so a result from the wrong
    # env (system python vs the project venv) can't silently mislead — the checks
    # are only meaningful in the SAME interpreter that runs the backend.
    import sys
    print(color(f"  解释器: {sys.executable}", Colors.DIM))
    print()

    cfg, raw_cfg = _load_mm_config()
    report = probe_mm_readiness(cfg, raw_cfg)

    glyph = {
        OK: (color("✓", Colors.GREEN), Colors.GREEN),
        MISSING: (color("✗", Colors.RED), Colors.RED),
        BROKEN: (color("✗", Colors.RED), Colors.RED),
        UNKNOWN: (color("?", Colors.YELLOW), Colors.YELLOW),
    }

    for cap in report["capabilities"]:
        mark, _ = glyph.get(cap["status"], (color("?", Colors.YELLOW), Colors.YELLOW))
        tag = "" if cap["required"] else color(" (可选)", Colors.DIM)
        print(f"  {mark} {cap['label']}{tag}")
        if cap["status"] != OK:
            if cap["reason"]:
                print(f"      {color('原因', Colors.DIM)}: {cap['reason']}")
            if cap["fix"]:
                print(f"      {color('修复', Colors.CYAN)}: {cap['fix']}")

    print()
    if report["ready"]:
        print(color("  ✓ 多模态已就绪 — 必需能力全部可用。", Colors.GREEN, Colors.BOLD))
        print(color("    (可选能力如缺失,对应功能会自动降级/关闭。)", Colors.DIM))
        return 0

    missing_required = [
        c["label"] for c in report["capabilities"]
        if c["required"] and c["status"] != OK
    ]
    print(color(
        f"  ✗ 多模态未就绪 — 缺必需能力: {', '.join(missing_required)}",
        Colors.RED, Colors.BOLD))
    print(color(
        "    运行 `argus setup multimodal` 按引导补齐,或按上面的修复命令手动处理。",
        Colors.YELLOW))
    return 1
