"""Shared language policy for prompts authored in English.

Runtime instructions remain English for portability, while model-generated
natural-language fields should follow the user's language when it is clear.
"""
from __future__ import annotations

LANGUAGE_POLICY = (
    "Write natural-language field values in the user's language when it is clear; "
    "otherwise use English. Preserve original on-screen text, code, paths, URLs, "
    "numbers, names, and quoted strings exactly as seen."
)
