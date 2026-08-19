from types import SimpleNamespace

from agent.multimodal.memory_backend import _env_audio_transcript_filter_reason


def _cfg(**overrides):
    base = {
        "env_audio_min_text_chars": 2,
        "env_audio_filter_fillers": True,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_env_audio_filters_silence_fillers():
    cfg = _cfg()

    assert _env_audio_transcript_filter_reason("嗯。", cfg) == "low_information_text"
    assert _env_audio_transcript_filter_reason("嗯嗯", cfg) == "low_information_text"
    assert _env_audio_transcript_filter_reason("呃。", cfg) == "low_information_text"
    assert _env_audio_transcript_filter_reason("oh...", cfg) == "low_information_text"


def test_env_audio_keeps_meaningful_transcripts_with_fillers():
    cfg = _cfg()

    assert _env_audio_transcript_filter_reason("嗯，我们继续看后排空间。", cfg) == ""
    assert _env_audio_transcript_filter_reason("纸杯上写着 Drink 多喝水。", cfg) == ""
    assert _env_audio_transcript_filter_reason("租车一天142元。", cfg) == ""


def test_env_audio_filler_filter_can_be_disabled():
    cfg = _cfg(env_audio_filter_fillers=False)

    assert _env_audio_transcript_filter_reason("嗯。", cfg) == "low_information_text"
    assert _env_audio_transcript_filter_reason("嗯嗯", cfg) == ""
