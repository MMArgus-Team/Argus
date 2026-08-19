"""``session.info`` must tell the desktop whether the thinking switch does anything.

Parity gap, not a cosmetic one: ``can_toggle_reasoning`` existed only on the web
path (``hermes_cli/web_server.py`` + ``hermes_cli/inventory.py``, consumed by
``web/src/components/ChatModelPill.tsx``). The desktop talks to this gateway
instead, which never reported the flag — so the desktop had no way to distinguish
"this model reasons" from "this endpoint lets us change that", and could only ever
render an ungated switch. For a thinking-only model or an aggregator serving
another vendor's model, that switch silently does nothing.

Distinct from ``supports_reasoning``: that says the model reasons. This says WE can
turn it on and off on this endpoint. See ``providers.base.ProviderProfile.
can_toggle_reasoning``, which derives the answer from the OFF payload rather than
declaring it, so it cannot drift from the request actually sent.
"""

import types

from tui_gateway.server import _session_info

GLM_URL = "https://open.bigmodel.cn/api/paas/v4"


def _agent(**over):
    return types.SimpleNamespace(
        model=over.get("model", "glm-5v-turbo"),
        provider=over.get("provider", "custom"),
        base_url=over.get("base_url", GLM_URL),
        reasoning_config=over.get("reasoning_config"),
        service_tier=None,
    )


class TestCanToggleReasoningReported:
    def test_flag_is_present(self):
        """The desktop cannot gate on a key that never arrives."""
        assert "can_toggle_reasoning" in _session_info(_agent())

    def test_glm_via_custom_endpoint_is_toggleable(self):
        """GLM reached through a hand-configured ``custom`` endpoint.

        Resolution is by hostname as well as name, so this must NOT fall back to
        the generic custom profile and report a bare guess.
        """
        assert _session_info(_agent())["can_toggle_reasoning"] is True

    def test_unknown_endpoint_defaults_to_true(self):
        """Unknown → assume controllable.

        A missing switch is worse than a no-op one; same optimistic default as
        ``hermes_cli/inventory.py::_apply_capabilities``.
        """
        info = _session_info(
            _agent(provider="", base_url="https://nowhere.invalid/v1", model="mystery-1")
        )

        assert info["can_toggle_reasoning"] is True

    def test_reported_regardless_of_current_effort(self):
        """Capability, not current state.

        Thinking being OFF right now says nothing about whether it CAN be turned
        back on — conflating the two is what made the control unrecoverable.
        """
        info = _session_info(_agent(reasoning_config={"enabled": False}))

        assert info["reasoning_effort"] == "none"
        assert info["can_toggle_reasoning"] is True


class TestExistingContractIntact:
    def test_thinking_off_still_reports_explicit_none(self):
        """Guards the neighbouring fix: OFF must be "none", never "".

        Empty means "no explicit level" and every client maps it to medium/on, so
        echoing "" for a disabled config made turning thinking off impossible.
        """
        assert _session_info(_agent(reasoning_config={"enabled": False}))["reasoning_effort"] == "none"

    def test_effort_passes_through(self):
        info = _session_info(_agent(reasoning_config={"enabled": True, "effort": "low"}))

        assert info["reasoning_effort"] == "low"

    def test_no_reasoning_config_reports_empty(self):
        assert _session_info(_agent())["reasoning_effort"] == ""
