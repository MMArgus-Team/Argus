"""Profile isolation contracts for the multimodal memory inspector API."""

from __future__ import annotations

import inspect
import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def memory_debug_profiles(monkeypatch, _isolate_hermes_home):
    from hermes_constants import get_hermes_home
    from hermes_cli import profiles

    default_home = get_hermes_home()
    profiles_root = default_home / "profiles"
    worker_home = profiles_root / "worker_mm"
    for home in (default_home, worker_home):
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: default_home)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: profiles_root)
    return {"default": default_home, "worker_mm": worker_home}


@pytest.fixture
def client(memory_debug_profiles):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    test_client = TestClient(app)
    test_client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return test_client


def _write_memory_db(home: Path, db_name: str, frame_id: str, text: str) -> Path:
    root = home / "memories" / "multimodal"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{db_name}.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE screen_texts ("
            "frame_id TEXT PRIMARY KEY, t_observed REAL, app TEXT, "
            "window_title TEXT, ocr_blocks TEXT, raw_text TEXT, source TEXT, "
            "created_at REAL)"
        )
        connection.execute(
            "INSERT INTO screen_texts VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (frame_id, 12.5, "browser", "profile window", "[]", text, "ocr", 13.0),
        )
    return path


def _write_trace_db(
    home: Path,
    content: str,
    *,
    session_id: str = "shared-trace",
) -> None:
    from hermes_state import SessionDB

    db = SessionDB(db_path=home / "state.db")
    try:
        db.create_session(session_id=session_id, source="desktop")
        db.append_message(
            session_id,
            role="tool",
            content=content,
            tool_name="recall_multimodal_memory",
        )
    finally:
        db.close()


def test_memory_debug_route_handlers_are_sync_for_fastapi_threadpool():
    from hermes_cli import web_server

    handlers = (
        web_server.mm_debug_sessions,
        web_server.mm_debug_session,
        web_server.mm_debug_frame,
        web_server.mm_debug_search,
        web_server.mm_debug_trace,
    )

    assert all(not inspect.iscoroutinefunction(handler) for handler in handlers)


def test_memory_debug_endpoints_read_only_requested_profile(
    client, memory_debug_profiles,
):
    default_home = memory_debug_profiles["default"]
    worker_home = memory_debug_profiles["worker_mm"]
    default_frame = "f_aaaaaaaaaa"
    worker_frame = "f_bbbbbbbbbb"

    _write_memory_db(default_home, "shared", default_frame, "defaultneedle")
    _write_memory_db(worker_home, "shared", worker_frame, "workerneedle")
    _write_memory_db(default_home, "default_only", "f_cccccccccc", "default only")
    _write_memory_db(worker_home, "worker_only", "f_dddddddddd", "worker only")

    listing = client.get(
        "/api/multimodal/memory/debug/sessions",
        params={"profile": "worker_mm"},
    )
    assert listing.status_code == 200
    listing_data = listing.json()
    assert listing_data["root"] == str(worker_home / "memories" / "multimodal")
    assert {row["name"] for row in listing_data["sessions"]} == {
        "shared.sqlite",
        "worker_only.sqlite",
    }

    session = client.get(
        "/api/multimodal/memory/debug/session/shared",
        params={"profile": "worker_mm"},
    )
    assert session.status_code == 200
    assert [row["frame_id"] for row in session.json()["timeline"]] == [worker_frame]
    assert "defaultneedle" not in str(session.json())

    frame = client.get(
        f"/api/multimodal/memory/debug/session/shared/frame/{worker_frame}",
        params={"profile": "worker_mm"},
    )
    assert frame.status_code == 200
    assert frame.json()["screen_text"]["raw_text"] == "workerneedle"

    search = client.get(
        "/api/multimodal/memory/debug/search",
        params={"query": "workerneedle", "profile": "worker_mm"},
    )
    assert search.status_code == 200
    assert [row["frame_id"] for row in search.json()["results"]] == [worker_frame]

    # The context-local override is restored after each worker-thread request.
    unscoped = client.get("/api/multimodal/memory/debug/session/shared")
    assert unscoped.status_code == 200
    assert [row["frame_id"] for row in unscoped.json()["timeline"]] == [default_frame]


def test_memory_debug_trace_does_not_mix_default_profile(
    client, memory_debug_profiles,
):
    default_home = memory_debug_profiles["default"]
    worker_home = memory_debug_profiles["worker_mm"]
    _write_trace_db(default_home, "default trace sentinel")
    _write_trace_db(worker_home, "worker trace sentinel")
    _write_trace_db(
        worker_home,
        "worker sibling trace sentinel",
        session_id="shared-trace-sibling",
    )

    for home, lines in (
        (default_home, ["[shared-trace] default recall log sentinel"]),
        (
            worker_home,
            [
                "[shared-trace] worker recall log sentinel",
                "[shared-trace-sibling] sibling writer log sentinel",
                "[unrelated-session] unrelated OCR timeout sentinel",
            ],
        ),
    ):
        logs = home / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        (logs / "agent.log").write_text("\n".join(lines) + "\n", encoding="utf-8")

    response = client.get(
        "/api/multimodal/memory/debug/trace",
        params={"session_id": "shared-trace", "profile": "worker_mm"},
    )
    assert response.status_code == 200
    data = response.json()
    assert [message["content"] for message in data["messages"]] == [
        "worker trace sentinel"
    ]
    assert [line.strip() for line in data["logs"]] == [
        "[shared-trace] worker recall log sentinel"
    ]
    assert "default" not in str(data)
    assert "sibling" not in str(data)


def test_memory_debug_search_bounds_thumbnail_work_and_rejects_one_char_dump(
    client, memory_debug_profiles, monkeypatch,
):
    from hermes_cli import web_server

    default_home = memory_debug_profiles["default"]
    _write_memory_db(default_home, "match_a", "f_1111111111", "common needle A")
    _write_memory_db(default_home, "match_b", "f_2222222222", "common needle B")
    encoded = []

    def _fake_frame_b64(path, frame_id, *, thumb=True):
        encoded.append((path.name, frame_id, thumb))
        return "thumb"

    monkeypatch.setattr(web_server, "_mm_debug_frame_b64", _fake_frame_b64)

    broad = client.get(
        "/api/multimodal/memory/debug/search",
        params={"query": "common needle", "scope": "all", "limit": 1},
    )
    assert broad.status_code == 200
    assert len(broad.json()["results"]) == 1
    assert len(encoded) == 1

    encoded.clear()
    single_character = client.get(
        "/api/multimodal/memory/debug/search",
        params={"query": "n", "scope": "all", "limit": 100},
    )
    assert single_character.status_code == 200
    assert single_character.json()["results"] == []
    assert encoded == []
