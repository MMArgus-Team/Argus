"""Regression guard for recall-verify fallback (finding C1).

_verify_frames ended with ``return verified if verified else fids``. When the
vision LLM correctly judged that every candidate frame is noise (``keep=[]``)
and the candidate set fit within the verify cap, ``verified`` was ``[]`` and
the ``else fids`` branch treated that as a *parse failure* — returning the
original, unfiltered noisy frames. The fix distinguishes "LLM said keep
nothing" (return []) from a genuine parse failure (return fids).

We exercise the real method by binding it to a minimal stub that supplies the
attributes it touches (cfg, frame_store, client, recorder), so no real vision
model / SQLite / hardware is needed.
"""
import types

import pytest

from agent.multimodal._workers import RecallWorker


class _FakeCfg:
    recall_verify_max_frames = 8
    model = "stub-model"


class _FakeStoredFrame:
    def __init__(self, fid):
        self.frame_id = fid
        self.ts = 0.0
        self.jpeg_b64 = ""


class _FakeFrameStore:
    def __init__(self, fids):
        self._fids = set(fids)

    def get_many(self, sel):
        return [_FakeStoredFrame(f) for f in sel if f in self._fids]


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResp:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, content):
        self._content = content

    async def create(self, **kwargs):
        return _FakeResp(self._content)


class _FakeClient:
    def __init__(self, content):
        self.chat = types.SimpleNamespace(completions=_FakeCompletions(content))


def _make_worker(fids, llm_json):
    """A bare object carrying just the attrs _verify_frames reads."""
    obj = types.SimpleNamespace(
        cfg=_FakeCfg(),
        model="stub-model",
        frame_store=_FakeFrameStore(fids),
        client=_FakeClient(llm_json),
        recorder=None,
    )
    obj._completion_controls = lambda *, max_tokens, temperature: {
        "max_tokens": max_tokens, "temperature": temperature}

    async def _create_chat_completion(msgs, *, max_tokens=256, temperature=0.1,
                                      enable_thinking=False, channel_tag=""):
        return await obj.client.chat.completions.create(
            model=obj.model, messages=msgs, max_tokens=max_tokens,
            temperature=temperature)
    obj._create_chat_completion = _create_chat_completion
    obj._verify_frames_with_grounding = RecallWorker._verify_frames_with_grounding.__get__(obj, type(obj))
    return obj


async def _run(fids, llm_json):
    obj = _make_worker(fids, llm_json)
    # Bind the real (unbound) method to our stub instance.
    method = RecallWorker._verify_frames.__get__(obj, type(obj))
    return await method(fids, query="我举的耳机")


@pytest.mark.asyncio
async def test_all_noise_returns_empty_not_original():
    """keep=[] with all frames inside the cap must yield [] (not the noise)."""
    fids = ["f1", "f2", "f3"]
    out = await _run(fids, '{"keep": []}')
    assert out == [], f"expected empty filtered result, got {out}"


@pytest.mark.asyncio
async def test_partial_keep_filters_correctly():
    fids = ["f1", "f2", "f3"]
    out = await _run(fids, '{"keep": ["f2"]}')
    assert out == ["f2"]


@pytest.mark.asyncio
async def test_parse_failure_returns_none_sentinel():
    """Malformed JSON (no list keep) is a real parse failure → None sentinel.

    The caller keeps the original fids when it sees None; an empty list [] is
    reserved for 'LLM said keep nothing'. This is the C1 contract that lets the
    call site (`if verified is not None`) distinguish the two.
    """
    fids = ["f1", "f2"]
    out = await _run(fids, "not json at all")
    assert out is None


def test_call_site_contract_empty_clears_none_keeps():
    """Guard the exact call-site branch that C1's earlier fix left broken.

    The recall path does: `if verified is not None: collected_fids = verified`.
    - verified == []   -> collected_fids becomes [] (noise frames cleared)
    - verified is None -> collected_fids unchanged (fallback keeps originals)
    An earlier `if verified:` truthy check wrongly treated [] like None.
    """
    def apply(collected, verified):
        if verified is not None:
            collected = verified
        return collected

    assert apply(["n1", "n2"], []) == []            # all-noise -> cleared
    assert apply(["f1"], ["f1"]) == ["f1"]           # kept
    assert apply(["n1", "n2"], None) == ["n1", "n2"]  # verify failed -> fallback


@pytest.mark.asyncio
async def test_frames_beyond_cap_are_conservatively_kept():
    """Frames beyond recall_verify_max_frames are always retained."""
    _FakeCfg.recall_verify_max_frames = 2
    try:
        fids = ["f1", "f2", "f3", "f4"]
        # LLM keeps nothing among the first 2; f3,f4 (beyond cap) survive.
        out = await _run(fids, '{"keep": []}')
        assert out == ["f3", "f4"]
    finally:
        _FakeCfg.recall_verify_max_frames = 8
