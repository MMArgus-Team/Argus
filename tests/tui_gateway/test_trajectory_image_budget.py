"""Bounded in-memory image retention for multimodal worker trajectories."""

from __future__ import annotations

from unittest.mock import patch

import tui_gateway.server as server


def _frame_payload(frame_id: str, image: str, *, key: str = "jpeg_b64") -> dict:
    return {
        "phase": "recall_done",
        "frames": [{
            "frame_id": frame_id,
            "score": 0.9,
            key: image,
        }],
    }


def _stored_image(session: dict, entry_idx: int, *, key: str = "jpeg_b64") -> str:
    return session["_mm_trajectory"][entry_idx]["payload"]["frames"][0][key]


def test_budget_omits_old_images_but_current_emit_stays_complete(monkeypatch):
    monkeypatch.setattr(server, "_MM_TRAJECTORY_IMAGE_GROUPS_MAX", 8)
    monkeypatch.setattr(server, "_MM_TRAJECTORY_IMAGE_B64_BUDGET_CHARS", 12)
    session = {"source": "multimodal"}
    writes = []

    with (
        patch.dict(server._sessions, {"sid": session}, clear=True),
        patch.object(server, "write_json", side_effect=lambda obj: writes.append(obj) or True),
    ):
        server._emit(
            "multimodal.bg", "sid", _frame_payload("old", "A" * 8))
        server._emit(
            "multimodal.bg", "sid", _frame_payload("latest", "B" * 8))

    rows = session["_mm_trajectory"]
    assert len(rows) == 2
    assert rows[0]["payload"]["frames"][0]["frame_id"] == "old"
    assert rows[0]["payload"]["frames"][0]["score"] == 0.9
    assert _stored_image(session, 0).startswith("<omitted jpeg_b64:")
    assert _stored_image(session, 1) == "B" * 8

    # multimodal.bg writes its original event and then the normalized trajectory
    # event. Both current-event copies retain the complete latest image even
    # though the storage budget stripped an older entry during this same emit.
    original, normalized = writes[-2:]
    assert original["params"]["payload"]["frames"][0]["jpeg_b64"] == "B" * 8
    assert (
        normalized["params"]["payload"]["payload"]["frames"][0]["jpeg_b64"]
        == "B" * 8
    )


def test_recent_group_cap_applies_to_jpeg_and_thumb_fields(monkeypatch):
    monkeypatch.setattr(server, "_MM_TRAJECTORY_IMAGE_GROUPS_MAX", 2)
    monkeypatch.setattr(server, "_MM_TRAJECTORY_IMAGE_B64_BUDGET_CHARS", 1_000)
    session = {"source": "multimodal"}

    with patch.dict(server._sessions, {"sid": session}, clear=True):
        server._record_mm_trajectory(
            "multimodal.trajectory", "sid",
            _frame_payload("one", "1" * 10, key="thumb_b64"),
        )
        server._record_mm_trajectory(
            "multimodal.trajectory", "sid",
            _frame_payload("two", "2" * 10),
        )
        latest = server._record_mm_trajectory(
            "multimodal.trajectory", "sid",
            _frame_payload("three", "3" * 10, key="thumb_b64"),
        )

    assert _stored_image(session, 0, key="thumb_b64").startswith(
        "<omitted thumb_b64:")
    assert _stored_image(session, 1) == "2" * 10
    assert _stored_image(session, 2, key="thumb_b64") == "3" * 10
    # The object returned for immediate delivery is not the mutable storage copy.
    assert latest["payload"]["frames"][0]["thumb_b64"] == "3" * 10


def test_latest_image_group_is_atomic_even_when_it_exceeds_budget(monkeypatch):
    monkeypatch.setattr(server, "_MM_TRAJECTORY_IMAGE_GROUPS_MAX", 8)
    monkeypatch.setattr(server, "_MM_TRAJECTORY_IMAGE_B64_BUDGET_CHARS", 4)
    session = {"source": "multimodal"}
    payload = {
        "phase": "recall_done",
        "frames": [
            {"frame_id": "a", "jpeg_b64": "A" * 6},
            {"frame_id": "b", "thumb_b64": "B" * 6},
        ],
    }

    with patch.dict(server._sessions, {"sid": session}, clear=True):
        server._record_mm_trajectory("multimodal.trajectory", "sid", payload)

    [stored] = session["_mm_trajectory"]
    assert stored["payload"]["frames"][0]["jpeg_b64"] == "A" * 6
    assert stored["payload"]["frames"][1]["thumb_b64"] == "B" * 6


def test_single_event_caps_frames_before_latest_group_budget_exemption():
    payload = {
        "phase": "recall_done",
        "frames": [
            {"frame_id": f"f{idx}", "jpeg_b64": str(idx) * 10}
            for idx in range(15)
        ],
    }

    safe = server._trajectory_safe(payload)

    frames = safe["frames"]
    assert len(frames) == server._MM_TRAJECTORY_FRAMES_PER_ENTRY_MAX + 1
    assert [frame["frame_id"] for frame in frames[:-1]] == [
        f"f{idx}" for idx in range(12)
    ]
    assert frames[-1] == "<truncated 3 items>"


def test_plain_append_skips_full_image_budget_scan(monkeypatch):
    session = {"source": "multimodal"}
    calls = []
    monkeypatch.setattr(
        server,
        "_bound_mm_trajectory_images",
        lambda rows: calls.append(len(rows)),
    )

    with patch.dict(server._sessions, {"sid": session}, clear=True):
        server._record_mm_trajectory(
            "multimodal.trajectory", "sid",
            {"phase": "thinking", "text": "no image"},
        )
        assert calls == []
        server._record_mm_trajectory(
            "multimodal.trajectory", "sid",
            _frame_payload("f1", "A" * 10),
        )

    assert calls == [2]


def test_trajectory_list_returns_already_bounded_independent_snapshot(monkeypatch):
    monkeypatch.setattr(server, "_MM_TRAJECTORY_MAX_ENTRIES", 3)
    monkeypatch.setattr(server, "_MM_TRAJECTORY_IMAGE_GROUPS_MAX", 1)
    monkeypatch.setattr(server, "_MM_TRAJECTORY_IMAGE_B64_BUDGET_CHARS", 6)
    session = {"source": "multimodal"}

    with patch.dict(server._sessions, {"sid": session}, clear=True):
        for idx in range(5):
            server._record_mm_trajectory(
                "multimodal.trajectory", "sid",
                _frame_payload(f"f{idx}", str(idx) * 5),
            )
        response = server._methods["multimodal.trajectory.list"](
            "rpc-1", {"session_id": "sid", "limit": 999},
        )

    stored = session["_mm_trajectory"]
    entries = response["result"]["entries"]
    assert len(stored) == 3
    assert len(entries) == response["result"]["count"] == 3
    assert [row["payload"]["frames"][0]["frame_id"] for row in entries] == [
        "f2", "f3", "f4",
    ]
    assert _stored_image(session, 0).startswith("<omitted jpeg_b64:")
    assert _stored_image(session, 1).startswith("<omitted jpeg_b64:")
    assert _stored_image(session, 2) == "4" * 5

    entries[-1]["payload"]["frames"][0]["jpeg_b64"] = "client mutation"
    assert _stored_image(session, 2) == "4" * 5


def test_trajectory_masks_short_structured_secrets_and_keeps_images():
    jpeg = "sk-proj-image-evidence-must-stay-byte-identical"
    thumb = "ghp_thumbnailEvidenceMustStayByteIdentical"
    payload = {
        "password": "x",
        "apiKey": "k",
        "token": "t",
        "nested": {
            "clientSecret": "s",
            "headers": {
                "Authorization": "Basic a",
                "set-cookie": "sid=z",
            },
            "openai_api_key": "q",
        },
        "frames": [{
            "frame_id": "frame-1",
            "jpeg_b64": jpeg,
            "thumb_b64": thumb,
        }],
    }

    safe = server._trajectory_safe(payload)

    assert safe["password"] == "***"
    assert safe["apiKey"] == "***"
    assert safe["token"] == "***"
    assert safe["nested"]["clientSecret"] == "***"
    assert safe["nested"]["headers"]["Authorization"] == "***"
    assert safe["nested"]["headers"]["set-cookie"] == "***"
    assert safe["nested"]["openai_api_key"] == "***"
    assert safe["frames"][0]["jpeg_b64"] == jpeg
    assert safe["frames"][0]["thumb_b64"] == thumb


def test_trajectory_force_redacts_terminal_text_and_fallback_objects(monkeypatch):
    from agent import redact

    # Prove the inspector is a force-on safety boundary, independent of the
    # user's runtime logging-redaction preference.
    monkeypatch.setattr(redact, "_REDACT_ENABLED", False)
    command_secret = "sk-proj-commandsecret1234567890"
    result_secret = "ghp_resultsecret1234567890"
    fallback_secret = "xoxb-fallbacksecret1234567890"

    class FallbackValue:
        def __str__(self):
            return f"token: {fallback_secret}"

    safe = server._trajectory_safe({
        "tool_name": "terminal",
        "command": f"OPENAI_API_KEY={command_secret} hermes status",
        "result": f"Authorization: Bearer tiny\ntoken: {result_secret}",
        "fallback": FallbackValue(),
    })

    serialized = str(safe)
    assert command_secret not in serialized
    assert result_secret not in serialized
    assert fallback_secret not in serialized
    assert safe["tool_name"] == "terminal"
    assert "OPENAI_API_KEY=***" in safe["command"]
    assert "Authorization: Bearer ***" in safe["result"]
    assert safe["fallback"] == "token: ***"


def test_trajectory_masks_url_userinfo_without_rewriting_safe_urls_or_images():
    safe_url = "https://example.test/callback?code=opaque-callback&state=keep"
    credential_url = (
        "https://alice:supersecret@internal.example/api?code=opaque-callback"
    )
    jpeg = "https://camera:raw-image-bytes@frame-evidence"

    safe = server._trajectory_safe(
        {
            "url": credential_url,
            "safe_url": safe_url,
            "frames": [{"frame_id": "frame-1", "jpeg_b64": jpeg}],
        }
    )

    assert safe["url"] == (
        "https://alice:***@internal.example/api?code=opaque-callback"
    )
    assert safe["safe_url"] == safe_url
    # Recalled evidence is intentionally byte-identical; image fields bypass
    # prose redaction and are controlled by the separate image-retention cap.
    assert safe["frames"][0]["jpeg_b64"] == jpeg


def test_trajectory_redaction_preserves_identity_and_grouping_fields():
    session = {"source": "multimodal"}
    payload = {
        "id": "worker-event-7",
        "session_id": "live-session-1",
        "stored_session_id": "stored-session-1",
        "request_id": "request-1",
        "client_request_id": "client-request-1",
        "parent_user_message_id": "parent-message-1",
        "worker": "QueryWorker",
        "phase": "search_done",
        "prompt_tokens": 17,
        "completion_tokens": 4,
        "token_count": 21,
        "auth": "z",
    }

    with patch.dict(server._sessions, {"sid": session}, clear=True):
        entry = server._record_mm_trajectory(
            "multimodal.trajectory", "sid", payload,
        )

    assert entry["id"].startswith("tr_1_")
    assert entry["worker"] == "QueryWorker"
    assert entry["phase"] == "search_done"
    assert entry["payload"] == {
        **payload,
        "auth": "***",
    }
    assert session["_mm_trajectory"][0]["payload"] == entry["payload"]
