"""Regression guard for SQLite connection leak in MemoryStore (finding C6).

Every ``_conn()`` caller used ``with self._conn() as c:``. A sqlite3 connection
used as a context manager only commits/rolls back the transaction — it does
NOT close the connection — so every read/write leaked a live connection (bad
under WAL). The fix adds a ``_connect()`` context manager that commits/rolls
back AND closes.

We verify by wrapping ``_conn`` so every connection it hands out is tracked,
then asserting all of them are closed after a batch of read/write calls.
Pure in-memory / temp-file SQLite (no cloud, no hardware).
"""
from agent.multimodal._config import Config
from agent.multimodal._memory import MemoryStore, MicroEvent


class _TrackedConn:
    """Proxy that records close() calls, delegating everything else."""

    def __init__(self, real):
        self._real = real
        self.closed = False

    def close(self):
        self.closed = True
        return self._real.close()

    def __getattr__(self, name):
        return getattr(self._real, name)


def _make_store(monkeypatch):
    store = MemoryStore(Config())  # empty mem_db_path -> temp sqlite file
    tracked: list[_TrackedConn] = []
    orig_conn = store._conn.__func__  # unbound, avoid recursion

    def _wrapped(self):
        tc = _TrackedConn(orig_conn(self))
        tracked.append(tc)
        return tc

    monkeypatch.setattr(MemoryStore, "_conn", _wrapped, raising=True)
    return store, tracked


def test_connect_closes_every_connection(monkeypatch):
    store, tracked = _make_store(monkeypatch)

    # A read path (pure select) and a write path (insert), each goes through
    # `with self._connect() as c:` now.
    store.insert_micro(MicroEvent(
        id="m1", t_start=0.0, t_end=1.0, description="d",
        subject="s", object="o", action="a", macro_id=None,
        facts_keys=[], frame_ids=[], created_at=0.0))
    # Trigger a few read paths that use _connect.
    _ = store.get_recent_entities(ask_ts=10.0, limit=5)
    _ = store.get_recent_macros(ask_ts=10.0, limit=3)

    assert tracked, "expected _connect() to open at least one connection"
    assert all(tc.closed for tc in tracked), (
        "some connections were not closed: "
        f"{[i for i, tc in enumerate(tracked) if not tc.closed]}")


def test_connect_closes_connection_on_exception(monkeypatch):
    store, tracked = _make_store(monkeypatch)

    # Force an error inside a _connect() block and ensure the connection is
    # still closed (rollback + close in the finally).
    try:
        with store._connect() as c:
            c.execute("SELECT * FROM this_table_does_not_exist")
    except Exception:
        pass

    assert tracked, "expected a tracked connection"
    assert tracked[-1].closed, "connection must be closed even on exception"


def test_set_session_id_persists_dashboard_binding():
    store = MemoryStore(Config())

    assert store.set_session_id("session-current") is True
    with store._connect() as c:
        row = c.execute(
            "SELECT value FROM meta WHERE key='hermes_session_id'"
        ).fetchone()

    assert row is not None
    assert row["value"] == "session-current"
