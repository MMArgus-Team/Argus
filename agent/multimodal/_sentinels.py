"""Language-neutral sentinel constants shared between backend workers and frontend.

These replace the former hard-coded Chinese magic strings so that both the
backend filter logic and the frontend display logic can match without depending
on any particular human language.
"""

# ─── Synthetic thought placeholders ──────────────────────────────────────────
# _workers.py emits these as the `thought` field in a router_react event when
# the model's actual thought is empty (self-explanatory scene).  The watcher
# engine's scene-label extractor (_extract_scene_label) skips them, and the
# frontend's SegmentCard filters them out of the description row.

SYNTH_THOUGHT_DIRECT = "This segment can be answered directly from the frames."
SYNTH_THOUGHT_CONTINUE = "Continue inspecting this segment."
SYNTH_THOUGHTS = frozenset([SYNTH_THOUGHT_DIRECT, SYNTH_THOUGHT_CONTINUE])

# ─── Recall empty-result sentinel ────────────────────────────────────────────
# memory_backend.py uses this as the findings fallback when no clues are found.
# _workers.py and watcher_engine.py compare against it to derive the boolean
# `found` field, and the frontend uses it to decide whether to show "found" vs
# "not found" styling.

RECALL_NO_CLUES = "(no relevant clues found in memory)"
