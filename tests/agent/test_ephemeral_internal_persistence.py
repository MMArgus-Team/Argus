"""Hidden Monitor/Watcher turns must never mutate durable agent history."""

from copy import deepcopy
from unittest.mock import Mock

from run_agent import AIAgent


def test_ephemeral_internal_error_persistence_is_a_total_noop():
    agent = object.__new__(AIAgent)
    agent._ephemeral_internal_turn = True
    canonical = [
        {"role": "user", "content": "ordinary question"},
        {"role": "assistant", "content": "ordinary answer"},
    ]
    agent._session_messages = canonical
    hidden_messages = canonical + [
        {"role": "user", "content": "private monitor hook instruction"},
        {
            "role": "assistant",
            "content": "",
            "_empty_terminal_sentinel": True,
        },
    ]
    before = deepcopy(hidden_messages)
    agent._drop_trailing_empty_response_scaffolding = Mock()
    agent._apply_persist_user_message_override = Mock()
    agent._save_session_log = Mock()
    agent._flush_messages_to_session_db = Mock()

    agent._persist_session(hidden_messages, canonical)

    assert hidden_messages == before
    assert agent._session_messages is canonical
    agent._drop_trailing_empty_response_scaffolding.assert_not_called()
    agent._apply_persist_user_message_override.assert_not_called()
    agent._save_session_log.assert_not_called()
    agent._flush_messages_to_session_db.assert_not_called()
