"""VoiceAgentV2 must use the dedicated DeepSeek client for every LLM role."""

import unittest
from unittest.mock import MagicMock, patch


class TestVoiceModelRouting(unittest.TestCase):
    def test_voice_agent_uses_one_voice_intent_client_for_all_roles(self):
        from tui_gateway import server

        deepseek_client = object()
        resolver = MagicMock(
            return_value=(deepseek_client, "deepseek-v4-flash")
        )
        voice_agent = MagicMock()
        voice_cls = MagicMock(return_value=voice_agent)
        session = {
            "_mm_live_watcher_agent": object(),
            "session_key": "voice-routing-test",
        }

        with (
            patch("agent.auxiliary_client.get_async_text_auxiliary_client", resolver),
            patch("agent.multimodal.voice_agent_v2.VoiceAgentV2", voice_cls),
            patch.object(server, "_load_cfg", return_value={}),
        ):
            result = server._get_voice_agent(session)

        self.assertIs(result, voice_agent)
        resolver.assert_called_once_with(task="voice_intent")
        kwargs = voice_cls.call_args.kwargs
        self.assertIs(kwargs["aux_client"], deepseek_client)
        self.assertIs(kwargs["intent_client"], deepseek_client)
        self.assertEqual(kwargs["aux_model"], "deepseek-v4-flash")
        self.assertEqual(kwargs["intent_model"], "deepseek-v4-flash")
        voice_agent.start.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
