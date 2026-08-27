"""Computer use must remain an explicit, non-core capability."""

from toolsets import TOOLSETS, _HERMES_CORE_TOOLS, resolve_toolset
from hermes_cli.tools_config import _DEFAULT_OFF_TOOLSETS


def test_computer_use_is_not_exposed_by_the_core_toolset():
    assert "computer_use" not in _HERMES_CORE_TOOLS
    assert "computer_use" not in resolve_toolset("hermes-core")
    assert "computer_use" in _DEFAULT_OFF_TOOLSETS


def test_computer_use_remains_available_when_explicitly_enabled():
    assert TOOLSETS["computer_use"]["tools"] == ["computer_use"]
    assert "computer_use" in resolve_toolset("computer_use")
