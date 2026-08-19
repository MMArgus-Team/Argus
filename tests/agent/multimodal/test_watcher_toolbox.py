"""Verification of the deep-research ToolBox + search layer (十项优化 #1/#3/#7).

These exercise the ReAct tool-execution surface WITHOUT a live model:

  #1  下游接口可用性 — text_search(AnySearch) parses a well-formed JSON-RPC
      response, and degrades gracefully (no crash, informative string) on HTTP
      error, transport failure, and a JSON-RPC {error} envelope. Anonymous
      (no api key) still issues the request.

  #3  工具不完善 — the ToolBox whitelist: text_search is the only live tool;
      the image_* tools are deprecated (never advertised in the prompt) and an
      unknown tool name returns a clean marker; enable_search=False gates ALL
      tools off.

  #7  搜索 query 生成 — the light query-variant generator strips parenthetical
      /bracketed noise, and oversized AnySearch results are truncated with a
      "换更精确的 query" hint (keeps ReAct context from ballooning).

All offline: httpx.AsyncClient is replaced by a fake so no network is touched.
"""

from __future__ import annotations

import asyncio

import pytest

from agent.multimodal._config import Config
from agent.multimodal._memory import FrameBuffer
from agent.multimodal._workers import ToolBox
import agent.multimodal._workers as workers_mod


def _toolbox(**cfg_over) -> ToolBox:
    cfg = Config()
    for k, v in cfg_over.items():
        setattr(cfg, k, v)
    buf = FrameBuffer(cfg)
    return ToolBox(cfg, buf, frame_store=None)


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(asyncio.wait_for(coro, timeout=10.0))
    finally:
        loop.close()


class _FakeResp:
    def __init__(self, *, json_data=None, raise_status=None, raise_send=None):
        self._json = json_data
        self._raise_status = raise_status
        self._raise_send = raise_send

    def raise_for_status(self):
        if self._raise_status:
            raise self._raise_status

    def json(self):
        return self._json


class _FakeClient:
    """Drop-in for httpx.AsyncClient: records the outgoing request, returns a
    canned response (or raises on send to simulate a network failure)."""

    last = {}

    def __init__(self, *a, **k):
        self._resp = _FakeClient._next_resp
        self._raise_send = _FakeClient._next_raise_send

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, endpoint, json=None, headers=None):
        _FakeClient.last = {"endpoint": endpoint, "json": json, "headers": headers}
        if self._raise_send:
            raise self._raise_send
        return self._resp


def _install_fake_httpx(monkeypatch, *, resp=None, raise_send=None):
    _FakeClient._next_resp = resp
    _FakeClient._next_raise_send = raise_send
    _FakeClient.last = {}
    monkeypatch.setattr(workers_mod.httpx, "AsyncClient", _FakeClient)


def _anysearch_ok(text):
    return _FakeResp(json_data={
        "jsonrpc": "2.0", "id": 1,
        "result": {"content": [{"type": "text", "text": text}]},
    })


# --------------------------------------------------------------------------- #
# #1 downstream availability + graceful degradation
# --------------------------------------------------------------------------- #
def test_text_search_parses_wellformed_response(monkeypatch):
    _install_fake_httpx(monkeypatch, resp=_anysearch_ok("甘道夫是托尔金笔下的巫师。"))
    tb = _toolbox(anysearch_api_key="")   # anonymous
    out = _run(tb.call("text_search", {"query": "甘道夫是谁"}))
    assert "甘道夫是托尔金" in out
    # Request actually went out to the configured endpoint, JSON-RPC shaped.
    sent = _FakeClient.last
    assert sent["endpoint"] == tb.cfg.anysearch_endpoint
    assert sent["json"]["method"] == "tools/call"
    assert sent["json"]["params"]["name"] == "search"
    assert sent["json"]["params"]["arguments"]["query"] == "甘道夫是谁"
    # Anonymous → no Authorization header.
    assert "Authorization" not in sent["headers"]


def test_text_search_sends_bearer_when_key_present(monkeypatch):
    _install_fake_httpx(monkeypatch, resp=_anysearch_ok("ok"))
    tb = _toolbox(anysearch_api_key="as_sk_test")
    _run(tb.call("text_search", {"query": "q"}))
    assert _FakeClient.last["headers"]["Authorization"] == "Bearer as_sk_test"


def test_text_search_graceful_on_http_error(monkeypatch):
    _install_fake_httpx(
        monkeypatch,
        resp=_FakeResp(raise_status=RuntimeError("HTTP 500 boom")))
    tb = _toolbox()
    out = _run(tb.call("text_search", {"query": "q"}))
    # No exception bubbles; user-facing degraded marker instead.
    assert out.startswith("[text_search]")
    assert "未返回结果" in out


def test_text_search_graceful_on_network_failure(monkeypatch):
    _install_fake_httpx(monkeypatch, raise_send=ConnectionError("dns fail"))
    tb = _toolbox()
    out = _run(tb.call("text_search", {"query": "q"}))
    assert out.startswith("[text_search]")
    assert "未返回结果" in out


def test_text_search_surfaces_jsonrpc_error(monkeypatch):
    _install_fake_httpx(monkeypatch, resp=_FakeResp(json_data={
        "jsonrpc": "2.0", "id": 1,
        "error": {"code": -32000, "message": "rate limited"}}))
    tb = _toolbox()
    out = _run(tb.call("text_search", {"query": "q"}))
    assert "外部检索出错" in out and "rate limited" in out


def test_text_search_empty_result(monkeypatch):
    _install_fake_httpx(monkeypatch, resp=_anysearch_ok(""))
    tb = _toolbox()
    out = _run(tb.call("text_search", {"query": "q"}))
    assert "无相关信息返回" in out


# --------------------------------------------------------------------------- #
# #3 ToolBox whitelist / tool completeness
# --------------------------------------------------------------------------- #
def test_unknown_tool_returns_clean_marker():
    tb = _toolbox()
    out = _run(tb.call("some_made_up_tool", {"x": 1}))
    assert out.startswith("[tool] 未知工具")


def test_missing_query_is_reported():
    tb = _toolbox()
    out = _run(tb.call("text_search", {}))
    assert "缺少 query" in out


def test_enable_search_false_gates_all_tools():
    tb = _toolbox(enable_search=False)
    out = _run(tb.call("text_search", {"query": "q"}))
    assert out == "[text_search] 已禁用"


def test_image_tools_are_deprecated_not_advertised():
    # The ReAct system prompt must NOT advertise the image_* tools (deprecated
    # 2026-07) so the LLM never dispatches them — text_search is the sole tool.
    prompt = workers_mod.WATCHER_REACT_SYSTEM
    assert "text_search" in prompt
    assert "image_search_current" not in prompt
    assert "image_search_crop" not in prompt


# --------------------------------------------------------------------------- #
# #7 search-query generation quality
# --------------------------------------------------------------------------- #
def test_query_variants_strip_bracketed_noise():
    v = ToolBox._text_query_variants("甘道夫（灰袍巫师）")
    assert v[0] == "甘道夫（灰袍巫师）"
    # A cleaned variant without the parenthetical is added for wider recall.
    assert any("灰袍巫师" not in x for x in v)
    assert len(v) <= 2


def test_query_variants_no_dup_when_clean():
    v = ToolBox._text_query_variants("Gandalf")
    assert v == ["Gandalf"]   # nothing to strip → single query, no wasted call


def test_oversized_result_is_truncated(monkeypatch):
    big = "结" * 9000
    _install_fake_httpx(monkeypatch, resp=_anysearch_ok(big))
    tb = _toolbox(anysearch_result_max_chars=4000)
    out = _run(tb.call("text_search", {"query": "q"}))
    assert "已截断" in out and "换更精确的 query" in out
    # Header + capped body + truncation notice — nowhere near the full 9000.
    assert len(out) < 4500
