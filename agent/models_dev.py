"""Models.dev registry integration — primary database for providers and models.

Fetches from https://models.dev/api.json — a community-maintained database
of 4000+ models across 109+ providers.  Provides:

- **Provider metadata**: name, base URL, env vars, documentation link
- **Model metadata**: context window, max output, cost/M tokens, capabilities
  (reasoning, tools, vision, PDF, audio), modalities, knowledge cutoff,
  open-weights flag, family grouping, deprecation status

Normal lookup resolution order (offline-first, never network-blocking):
  1. Fresh in-process cache
  2. Fresh disk cache (~/.argus/models_dev_cache.json)
  3. Stale in-process/disk cache, returned immediately + background refresh
  4. Bundled snapshot (ships with the package), returned immediately +
     background refresh
  5. Empty result when every local source is unavailable + rate-limited
     background refresh

Only an explicit ``force_refresh=True`` performs a synchronous network fetch
from https://models.dev/api.json.

Other modules should import the dataclasses and query functions from here
rather than parsing the raw JSON themselves.
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils import atomic_json_write

import requests
import threading

logger = logging.getLogger(__name__)

MODELS_DEV_URL = "https://models.dev/api.json"
_MODELS_DEV_CACHE_TTL = 3600  # 1 hour in-memory
# When a cached copy exists (disk or in-mem) but is stale, we serve it
# immediately and refresh in the background. This is how long we wait
# before allowing another background refresh attempt, so a persistently
# unreachable models.dev doesn't spawn a refresh thread on every call.
_MODELS_DEV_BG_RETRY_INTERVAL = 300  # 5 min between background refresh attempts

# In-memory cache
_models_dev_cache: Dict[str, Any] = {}
_models_dev_cache_time: float = 0
# Cache data and its timestamp are one state transition.  Normal cold-start
# callers may parse the bundled snapshot concurrently with a fast background
# refresh; publishing both under one lock (and only accepting the newer
# timestamp) prevents a late snapshot parser from overwriting live data.
_models_dev_cache_lock = threading.RLock()

# Background-refresh coordination: a single in-flight refresh at a time,
# and a floor on how often we start one (see _MODELS_DEV_BG_RETRY_INTERVAL).
_bg_refresh_lock = threading.Lock()
_bg_refresh_inflight = False
_bg_refresh_last_attempt: float = 0.0


# ---------------------------------------------------------------------------
# Dataclasses — rich metadata for providers and models
# ---------------------------------------------------------------------------

@dataclass
class ModelInfo:
    """Full metadata for a single model from models.dev."""

    id: str
    name: str
    family: str
    provider_id: str        # models.dev provider ID (e.g. "anthropic")

    # Capabilities
    reasoning: bool = False
    tool_call: bool = False
    attachment: bool = False       # supports image/file attachments (vision)
    temperature: bool = False
    structured_output: bool = False
    open_weights: bool = False

    # Modalities
    input_modalities: Tuple[str, ...] = ()    # ("text", "image", "pdf", ...)
    output_modalities: Tuple[str, ...] = ()

    # Limits
    context_window: int = 0
    max_output: int = 0
    max_input: Optional[int] = None

    # Cost (per million tokens, USD)
    cost_input: float = 0.0
    cost_output: float = 0.0
    cost_cache_read: Optional[float] = None
    cost_cache_write: Optional[float] = None

    # Metadata
    knowledge_cutoff: str = ""
    release_date: str = ""
    status: str = ""          # "alpha", "beta", "deprecated", or ""
    interleaved: Any = False  # True or {"field": "reasoning_content"}

    def has_cost_data(self) -> bool:
        return self.cost_input > 0 or self.cost_output > 0

    def supports_vision(self) -> bool:
        return self.attachment or "image" in self.input_modalities

    def supports_pdf(self) -> bool:
        return "pdf" in self.input_modalities

    def supports_audio_input(self) -> bool:
        return "audio" in self.input_modalities

    def format_cost(self) -> str:
        """Human-readable cost string, e.g. '$3.00/M in, $15.00/M out'."""
        if not self.has_cost_data():
            return "unknown"
        parts = [f"${self.cost_input:.2f}/M in", f"${self.cost_output:.2f}/M out"]
        if self.cost_cache_read is not None:
            parts.append(f"cache read ${self.cost_cache_read:.2f}/M")
        return ", ".join(parts)

    def format_capabilities(self) -> str:
        """Human-readable capabilities, e.g. 'reasoning, tools, vision, PDF'."""
        caps = []
        if self.reasoning:
            caps.append("reasoning")
        if self.tool_call:
            caps.append("tools")
        if self.supports_vision():
            caps.append("vision")
        if self.supports_pdf():
            caps.append("PDF")
        if self.supports_audio_input():
            caps.append("audio")
        if self.structured_output:
            caps.append("structured output")
        if self.open_weights:
            caps.append("open weights")
        return ", ".join(caps) if caps else "basic"


@dataclass
class ProviderInfo:
    """Full metadata for a provider from models.dev."""

    id: str                         # models.dev provider ID
    name: str                       # display name
    env: Tuple[str, ...]            # env var names for API key
    api: str                        # base URL
    doc: str = ""                   # documentation URL
    model_count: int = 0


# ---------------------------------------------------------------------------
# Provider ID mapping: Hermes ↔ models.dev
# ---------------------------------------------------------------------------

# Hermes provider names → models.dev provider IDs
PROVIDER_TO_MODELS_DEV: Dict[str, str] = {
    "openrouter": "openrouter",
    "novita": "novita-ai",
    "anthropic": "anthropic",
    "openai": "openai",
    "openai-codex": "openai",
    "zai": "zai",
    "kimi": "kimi-for-coding",
    "kimi-coding": "kimi-for-coding",
    "moonshot": "kimi-for-coding",
    "stepfun": "stepfun",
    "kimi-coding-cn": "kimi-for-coding",
    "minimax": "minimax",
    "minimax-oauth": "minimax",
    "minimax-cn": "minimax-cn",
    "deepseek": "deepseek",
    "alibaba": "alibaba",
    "qwen-oauth": "alibaba",
    "copilot": "github-copilot",
    "opencode-zen": "opencode",
    "opencode-go": "opencode-go",
    "kilocode": "kilo",
    "fireworks": "fireworks-ai",
    "huggingface": "huggingface",
    "gemini": "google",
    "google": "google",
    "xai": "xai",
    # xAI OAuth is an authentication/transport path for the same xAI model
    # catalog, so model metadata should resolve through the xAI provider.
    "xai-oauth": "xai",
    "xiaomi": "xiaomi",
    "nvidia": "nvidia",
    "groq": "groq",
    "mistral": "mistral",
    "togetherai": "togetherai",
    "perplexity": "perplexity",
    "cohere": "cohere",
    "ollama-cloud": "ollama-cloud",
}

# Reverse mapping: models.dev → Hermes (built lazily)
_MODELS_DEV_TO_PROVIDER: Optional[Dict[str, str]] = None



def _get_cache_path() -> Path:
    """Return path to disk cache file."""
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "models_dev_cache.json"


def _get_bundled_snapshot_path() -> Path:
    """Return the offline registry snapshot shipped beside this module."""
    return Path(__file__).with_name("models_dev_api.json")


def _load_bundled_snapshot() -> Dict[str, Any]:
    """Load the bundled models.dev registry, or ``{}`` when unavailable.

    The snapshot is the cold-start/offline fallback.  It must never make a
    normal metadata lookup fail: source checkouts from before the snapshot was
    added, incomplete packages, and a locally-corrupted JSON file all degrade
    to an empty registry while the background refresher tries the network.
    """
    try:
        with open(_get_bundled_snapshot_path(), encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and data:
            return data
        logger.debug("Bundled models.dev snapshot is empty or malformed")
    except Exception as e:
        logger.debug("Failed to load bundled models.dev snapshot: %s", e)
    return {}


def _load_disk_cache() -> Dict[str, Any]:
    """Load models.dev data from disk cache."""
    try:
        cache_path = _get_cache_path()
        if cache_path.exists():
            with open(cache_path, encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.debug("Failed to load models.dev disk cache: %s", e)
    return {}


def _disk_cache_age_seconds() -> Optional[float]:
    """Return age (in seconds) of the disk cache file, or None if missing.

    Used by ``fetch_models_dev`` to short-circuit the network probe when
    a recent on-disk cache exists. Errors (missing file, permission
    denied, weird filesystem) all return None — callers fall through
    to the network fetch path.
    """
    try:
        cache_path = _get_cache_path()
        if not cache_path.exists():
            return None
        mtime = cache_path.stat().st_mtime
        age = time.time() - mtime
        # Negative age means the file's mtime is in the future (clock skew
        # or system clock reset). Treat as "unknown freshness" → fall
        # through to network so we don't serve potentially-bad data
        # forever.
        if age < 0:
            return None
        return age
    except Exception as e:
        logger.debug("Failed to stat models.dev disk cache: %s", e)
        return None


def _save_disk_cache(data: Dict[str, Any]) -> None:
    """Save models.dev data to disk cache atomically."""
    try:
        cache_path = _get_cache_path()
        atomic_json_write(cache_path, data, indent=None, separators=(",", ":"))
    except Exception as e:
        logger.debug("Failed to save models.dev disk cache: %s", e)


def _cache_snapshot() -> tuple[Dict[str, Any], float]:
    """Return the in-memory registry and timestamp as one atomic snapshot."""
    with _models_dev_cache_lock:
        return _models_dev_cache, _models_dev_cache_time


def _publish_cache_if_newer(
    data: Dict[str, Any],
    data_time: float,
) -> Dict[str, Any]:
    """Publish ``data`` unless a newer cache won the race.

    Network responses use their completion time. Disk data uses its file mtime,
    and bundled/stale fallbacks use an intentionally old timestamp. Therefore a
    live response always wins even when another cold-start caller finishes
    parsing the bundled snapshot later.
    """
    global _models_dev_cache, _models_dev_cache_time
    if not data:
        cached, _ = _cache_snapshot()
        return cached
    with _models_dev_cache_lock:
        if not _models_dev_cache or data_time > _models_dev_cache_time:
            _models_dev_cache = data
            _models_dev_cache_time = data_time
        return _models_dev_cache


def _fetch_from_network() -> Dict[str, Any]:
    """Blocking network fetch of the models.dev registry.

    Returns the registry dict on success, or ``{}`` on any failure. On
    success also updates the in-mem + disk caches. Safe to call from a
    background thread — cache publication is atomic and rejects older local
    fallback data that loses a race with this live response.
    """
    try:
        response = requests.get(MODELS_DEV_URL, timeout=15)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logger.debug("Failed to fetch models.dev: %s", e)
        return {}
    if isinstance(data, dict) and data:
        _publish_cache_if_newer(data, time.time())
        _save_disk_cache(data)
        logger.debug(
            "Fetched models.dev registry: %d providers, %d total models",
            len(data),
            sum(len(p.get("models", {})) for p in data.values() if isinstance(p, dict)),
        )
        return data
    return {}


def _trigger_background_refresh() -> None:
    """Kick off a network refresh in a daemon thread, if not already running.

    Rate-limited to one attempt per ``_MODELS_DEV_BG_RETRY_INTERVAL`` so a
    persistently unreachable models.dev doesn't spawn a thread on every
    lookup. Never blocks the caller — this is the whole point: a stale
    cache is served immediately and the network cost is paid off the hot
    path (crucially, off the event loop when called from an async handler
    via the threadpool).
    """
    global _bg_refresh_inflight, _bg_refresh_last_attempt
    now = time.time()
    with _bg_refresh_lock:
        if _bg_refresh_inflight:
            return
        if (now - _bg_refresh_last_attempt) < _MODELS_DEV_BG_RETRY_INTERVAL:
            return
        _bg_refresh_inflight = True
        _bg_refresh_last_attempt = now

    def _run() -> None:
        global _bg_refresh_inflight
        try:
            _fetch_from_network()
        finally:
            with _bg_refresh_lock:
                _bg_refresh_inflight = False

    threading.Thread(
        target=_run, name="models-dev-refresh", daemon=True
    ).start()


def fetch_models_dev(force_refresh: bool = False) -> Dict[str, Any]:
    """Fetch models.dev registry without blocking normal lookups on network I/O.

    Returns the full registry dict keyed by provider ID, or empty dict on failure.

    Cache hierarchy (when ``force_refresh=False``):
      1. In-memory cache, populated and < TTL old → return immediately.
      2. Disk cache file < TTL old by mtime → load, populate in-mem, return.
      3. Cache exists but is STALE (in-mem or on disk) → return the stale copy
         IMMEDIATELY and refresh from the network in a background thread. The
         caller NEVER blocks on the network when any cached copy is available.
         This is what keeps ``/api/model/options`` (and every other
         capability lookup) responsive when models.dev is slow/unreachable.
      4. No cache at all (first run, empty disk) → return the bundled
         snapshot immediately and refresh it in the background.
      5. Bundled snapshot missing/corrupt → return ``{}`` immediately and
         start the same rate-limited background refresh. Repeated callers never
         retry the 15s network request on their foreground path.

    When ``force_refresh=True`` (used by ``argus config refresh``, the
    \"refresh model catalog\" code path), stages 1-3 are skipped and the
    function performs a blocking network fetch, falling back to disk and then
    the bundled snapshot on failure. This is the ONLY path that intentionally
    blocks on the network.
    """
    # force_refresh: user explicitly asked for a live catalog — block on the
    # network, fall back to disk/bundled data on failure.
    if force_refresh:
        data = _fetch_from_network()
        if data:
            return data
        cached, _ = _cache_snapshot()
        fallback = cached or _load_disk_cache() or _load_bundled_snapshot()
        if not fallback:
            return {}
        # Treat fallback data as stale-but-usable: normal calls serve it
        # immediately and become eligible for another background attempt after
        # the retry interval, rather than waiting a full cache TTL.
        fallback_time = (
            time.time()
            - _MODELS_DEV_CACHE_TTL
            + _MODELS_DEV_BG_RETRY_INTERVAL
        )
        return _publish_cache_if_newer(fallback, fallback_time)

    # Stage 1: fresh in-memory cache wins. Hot path on long-lived processes.
    cached, cached_time = _cache_snapshot()
    if cached and (time.time() - cached_time) < _MODELS_DEV_CACHE_TTL:
        return cached

    # Stage 2: fresh-by-mtime disk cache — load it into memory, no network.
    disk_age = _disk_cache_age_seconds()
    if disk_age is not None and disk_age < _MODELS_DEV_CACHE_TTL:
        disk_data = _load_disk_cache()
        if disk_data:
            # Anchor in-mem TTL to the disk file's age so we don't extend an
            # already-aging cache by another full hour.
            cached = _publish_cache_if_newer(
                disk_data,
                time.time() - disk_age,
            )
            logger.debug(
                "Loaded models.dev from fresh disk cache "
                "(%d providers, age=%.0fs)", len(disk_data), disk_age,
            )
            return cached

    # Stage 3: a STALE cache exists (expired in-mem, or expired/any on disk).
    # Serve it immediately and refresh in the background — never block the
    # caller on the network when we have SOMETHING usable to return.
    cached, _ = _cache_snapshot()
    stale = cached or _load_disk_cache()
    if stale:
        if not cached:
            # Warm the in-mem cache from disk so subsequent calls hit Stage 1
            # (within TTL of THIS load) instead of re-reading disk each time.
            stale = _publish_cache_if_newer(
                stale,
                time.time()
                - _MODELS_DEV_CACHE_TTL
                + _MODELS_DEV_BG_RETRY_INTERVAL,
            )
        _trigger_background_refresh()
        logger.debug(
            "Serving stale models.dev cache (%d providers); refreshing in background",
            len(stale),
        )
        return stale

    # Stage 4: first run / no writable cache. The checked-in snapshot is the
    # foreground answer; network freshness is strictly background work. This
    # also prevents one /api/model/options request from paying the same 15s
    # timeout repeatedly through independent metadata helpers.
    bundled = _load_bundled_snapshot()
    if bundled:
        # As with a stale disk cache, make the bundled copy eligible for a new
        # background attempt after the retry interval while keeping every
        # intervening lookup on the in-memory fast path.
        bundled = _publish_cache_if_newer(
            bundled,
            time.time()
            - _MODELS_DEV_CACHE_TTL
            + _MODELS_DEV_BG_RETRY_INTERVAL,
        )
        _trigger_background_refresh()
        logger.debug(
            "Loaded bundled models.dev snapshot (%d providers); "
            "refreshing in background",
            len(bundled),
        )
        return bundled

    # An incomplete package or corrupt snapshot still must not put network I/O
    # back on the caller's critical path. _trigger_background_refresh() already
    # supplies single-flight + retry throttling, which doubles as a negative
    # cache for this empty-result case.
    _trigger_background_refresh()
    logger.debug(
        "No models.dev cache or bundled snapshot; refreshing in background"
    )
    return {}


def lookup_models_dev_context(provider: str, model: str) -> Optional[int]:
    """Look up context_length for a provider+model combo in models.dev.

    Returns the context window in tokens, or None if not found.
    Handles case-insensitive matching and filters out context=0 entries.
    """
    mdev_provider_id = PROVIDER_TO_MODELS_DEV.get(provider)
    if not mdev_provider_id:
        return None

    data = fetch_models_dev()
    provider_data = data.get(mdev_provider_id)
    if not isinstance(provider_data, dict):
        return None

    models = provider_data.get("models", {})
    if not isinstance(models, dict):
        return None

    # Exact match
    entry = models.get(model)
    if entry:
        ctx = _extract_context(entry)
        if ctx:
            return ctx

    # Case-insensitive match
    model_lower = model.lower()
    for mid, mdata in models.items():
        if mid.lower() == model_lower:
            ctx = _extract_context(mdata)
            if ctx:
                return ctx

    # Suffix-aware fallback: some providers (e.g. ollama-cloud) store
    # model IDs with :cloud / -cloud suffixes in models.dev while the
    # live API returns bare names.  Without this, kimi-k2.6 misses the
    # kimi-k2.6:cloud entry and falls through to stale OpenRouter metadata
    # reporting 32768 — tripping the 64k minimum-context guard.
    # The suffix-stripping in fetch_ollama_cloud_models() handles the
    # model-picker UX; this handles the context-length lookup path.
    for suffix in (":cloud", "-cloud"):
        suffixed_key = model + suffix
        entry = models.get(suffixed_key)
        if entry:
            ctx = _extract_context(entry)
            if ctx:
                return ctx
        # Also try case-insensitive
        suffixed_lower = model_lower + suffix
        for mid, mdata in models.items():
            if mid.lower() == suffixed_lower:
                ctx = _extract_context(mdata)
                if ctx:
                    return ctx

    return None


def _extract_context(entry: Dict[str, Any]) -> Optional[int]:
    """Extract context_length from a models.dev model entry.

    Returns None for invalid/zero values (some audio/image models have context=0).
    """
    if not isinstance(entry, dict):
        return None
    limit = entry.get("limit")
    if not isinstance(limit, dict):
        return None
    ctx = limit.get("context")
    if isinstance(ctx, (int, float)) and ctx > 0:
        return int(ctx)
    return None


# ---------------------------------------------------------------------------
# Model capability metadata
# ---------------------------------------------------------------------------


@dataclass
class ModelCapabilities:
    """Structured capability metadata for a model from models.dev."""

    supports_tools: bool = True
    supports_vision: bool = False
    supports_reasoning: bool = False
    context_window: int = 200000
    max_output_tokens: int = 8192
    model_family: str = ""


def _get_provider_models(provider: str) -> Optional[Dict[str, Any]]:
    """Resolve a Hermes provider ID to its models dict from models.dev.

    Returns the models dict or None if the provider is unknown or has no data.
    """
    mdev_provider_id = PROVIDER_TO_MODELS_DEV.get(provider)
    if not mdev_provider_id:
        return None

    data = fetch_models_dev()
    provider_data = data.get(mdev_provider_id)
    if not isinstance(provider_data, dict):
        return None

    models = provider_data.get("models", {})
    if not isinstance(models, dict):
        return None

    return models


def _find_model_entry(models: Dict[str, Any], model: str) -> Optional[Dict[str, Any]]:
    """Find a model entry by exact match, then case-insensitive fallback."""
    # Exact match
    entry = models.get(model)
    if isinstance(entry, dict):
        return entry

    # Case-insensitive match
    model_lower = model.lower()
    for mid, mdata in models.items():
        if mid.lower() == model_lower and isinstance(mdata, dict):
            return mdata

    return None


def get_model_capabilities(provider: str, model: str) -> Optional[ModelCapabilities]:
    """Look up full capability metadata from models.dev cache.

    Uses the existing fetch_models_dev() and PROVIDER_TO_MODELS_DEV mapping.
    Returns None if model not found.

    Extracts from model entry fields:
      - reasoning  (bool)  → supports_reasoning
      - tool_call  (bool)  → supports_tools
      - attachment (bool)  → supports_vision
      - limit.context (int) → context_window
      - limit.output  (int) → max_output_tokens
      - family     (str)   → model_family
    """
    models = _get_provider_models(provider)
    if models is None:
        return None

    entry = _find_model_entry(models, model)
    if entry is None:
        return None

    # Extract capability flags (default to False if missing)
    supports_tools = bool(entry.get("tool_call", False))
    # Vision: prefer explicit `modalities.input` when models.dev provides it.
    # The older `attachment` flag can be stale or too broad for image routing;
    # fall back to it only when the input modalities are absent/invalid.
    input_mods = entry.get("modalities", {})
    if isinstance(input_mods, dict):
        input_mods = input_mods.get("input")
    else:
        input_mods = None
    if isinstance(input_mods, list):
        supports_vision = "image" in input_mods
    else:
        supports_vision = bool(entry.get("attachment", False))
    supports_reasoning = bool(entry.get("reasoning", False))

    # Extract limits
    limit = entry.get("limit", {})
    if not isinstance(limit, dict):
        limit = {}

    ctx = limit.get("context")
    context_window = int(ctx) if isinstance(ctx, (int, float)) and ctx > 0 else 200000

    out = limit.get("output")
    max_output_tokens = int(out) if isinstance(out, (int, float)) and out > 0 else 8192

    model_family = entry.get("family", "") or ""

    return ModelCapabilities(
        supports_tools=supports_tools,
        supports_vision=supports_vision,
        supports_reasoning=supports_reasoning,
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        model_family=model_family,
    )


def list_provider_models(provider: str) -> List[str]:
    """Return all model IDs for a provider from models.dev.

    Returns an empty list if the provider is unknown or has no data.
    """
    from hermes_cli.models import normalize_provider
    provider = normalize_provider(provider) or provider
    
    models = _get_provider_models(provider)
    if models is None:
        return []
    return [
        mid for mid in models.keys()
        if not _should_hide_from_provider_catalog(provider, mid)
    ]


# Patterns that indicate non-agentic or noise models (TTS, embedding,
# dated preview snapshots, live/streaming-only, image-only).
import re
_NOISE_PATTERNS: re.Pattern = re.compile(
    r"-tts\b|embedding|live-|-(preview|exp)-\d{2,4}[-_]|"
    r"-image\b|-image-preview\b|-customtools\b",
    re.IGNORECASE,
)

# Google's live Gemini catalogs currently include a mix of stale slugs and
# Gemma models whose TPM quotas are too small for normal Hermes agent traffic.
# Keep capability metadata available for direct/manual use, but hide these from
# the Gemini model catalogs we surface in setup and model selection.
_GOOGLE_HIDDEN_MODELS = frozenset({
    # Low-TPM Gemma models that trip Google input-token quota walls under
    # agent-style traffic despite advertising large context windows.
    "gemma-4-31b-it",
    "gemma-4-26b-it",
    "gemma-4-26b-a4b-it",
    "gemma-3-1b",
    "gemma-3-1b-it",
    "gemma-3-2b",
    "gemma-3-2b-it",
    "gemma-3-4b",
    "gemma-3-4b-it",
    "gemma-3-12b",
    "gemma-3-12b-it",
    "gemma-3-27b",
    "gemma-3-27b-it",
    # Stale/retired Google slugs that still surface through models.dev-backed
    # Gemini selection but 404 on the current Google endpoints.
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-1.5-flash-8b",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
})


def _should_hide_from_provider_catalog(provider: str, model_id: str) -> bool:
    provider_lower = (provider or "").strip().lower()
    model_lower = (model_id or "").strip().lower()
    if provider_lower in {"gemini", "google"} and model_lower in _GOOGLE_HIDDEN_MODELS:
        return True
    return False


def list_agentic_models(provider: str) -> List[str]:
    """Return model IDs suitable for agentic use from models.dev.

    Filters for tool_call=True and excludes noise (TTS, embedding,
    dated preview snapshots, live/streaming, image-only models).
    Returns an empty list on any failure.
    """
    models = _get_provider_models(provider)
    if models is None:
        return []

    result = []
    for mid, entry in models.items():
        if not isinstance(entry, dict):
            continue
        if _should_hide_from_provider_catalog(provider, mid):
            continue
        if not entry.get("tool_call", False):
            continue
        if _NOISE_PATTERNS.search(mid):
            continue
        result.append(mid)
    return result



# ---------------------------------------------------------------------------
# Rich dataclass constructors — parse raw models.dev JSON into dataclasses
# ---------------------------------------------------------------------------

def _parse_model_info(model_id: str, raw: Dict[str, Any], provider_id: str) -> ModelInfo:
    """Convert a raw models.dev model entry dict into a ModelInfo dataclass."""
    limit = raw.get("limit") or {}
    if not isinstance(limit, dict):
        limit = {}

    cost = raw.get("cost") or {}
    if not isinstance(cost, dict):
        cost = {}

    modalities = raw.get("modalities") or {}
    if not isinstance(modalities, dict):
        modalities = {}

    input_mods = modalities.get("input") or []
    output_mods = modalities.get("output") or []

    ctx = limit.get("context")
    ctx_int = int(ctx) if isinstance(ctx, (int, float)) and ctx > 0 else 0
    out = limit.get("output")
    out_int = int(out) if isinstance(out, (int, float)) and out > 0 else 0
    inp = limit.get("input")
    inp_int = int(inp) if isinstance(inp, (int, float)) and inp > 0 else None

    return ModelInfo(
        id=model_id,
        name=raw.get("name", "") or model_id,
        family=raw.get("family", "") or "",
        provider_id=provider_id,
        reasoning=bool(raw.get("reasoning", False)),
        tool_call=bool(raw.get("tool_call", False)),
        attachment=bool(raw.get("attachment", False)),
        temperature=bool(raw.get("temperature", False)),
        structured_output=bool(raw.get("structured_output", False)),
        open_weights=bool(raw.get("open_weights", False)),
        input_modalities=tuple(input_mods) if isinstance(input_mods, list) else (),
        output_modalities=tuple(output_mods) if isinstance(output_mods, list) else (),
        context_window=ctx_int,
        max_output=out_int,
        max_input=inp_int,
        cost_input=float(cost.get("input", 0) or 0),
        cost_output=float(cost.get("output", 0) or 0),
        cost_cache_read=float(cost["cache_read"]) if "cache_read" in cost and cost["cache_read"] is not None else None,
        cost_cache_write=float(cost["cache_write"]) if "cache_write" in cost and cost["cache_write"] is not None else None,
        knowledge_cutoff=raw.get("knowledge", "") or "",
        release_date=raw.get("release_date", "") or "",
        status=raw.get("status", "") or "",
        interleaved=raw.get("interleaved", False),
    )


def _parse_provider_info(provider_id: str, raw: Dict[str, Any]) -> ProviderInfo:
    """Convert a raw models.dev provider entry dict into a ProviderInfo."""
    env = raw.get("env") or []
    models = raw.get("models") or {}
    return ProviderInfo(
        id=provider_id,
        name=raw.get("name", "") or provider_id,
        env=tuple(env) if isinstance(env, list) else (),
        api=raw.get("api", "") or "",
        doc=raw.get("doc", "") or "",
        model_count=len(models) if isinstance(models, dict) else 0,
    )


# ---------------------------------------------------------------------------
# Provider-level queries
# ---------------------------------------------------------------------------

def get_provider_info(provider_id: str) -> Optional[ProviderInfo]:
    """Get full provider metadata from models.dev.

    Accepts either a Hermes provider ID (e.g. "kilocode") or a models.dev
    ID (e.g. "kilo").  Returns None if the provider is not in the catalog.
    """
    # Resolve Hermes ID → models.dev ID
    mdev_id = PROVIDER_TO_MODELS_DEV.get(provider_id, provider_id)

    data = fetch_models_dev()
    raw = data.get(mdev_id)
    if not isinstance(raw, dict):
        return None

    return _parse_provider_info(mdev_id, raw)


# ---------------------------------------------------------------------------
# Model-level queries (rich ModelInfo)
# ---------------------------------------------------------------------------

def get_model_info(
    provider_id: str, model_id: str
) -> Optional[ModelInfo]:
    """Get full model metadata from models.dev.

    Accepts Hermes or models.dev provider ID.  Tries exact match then
    case-insensitive fallback.  Returns None if not found.
    """
    mdev_id = PROVIDER_TO_MODELS_DEV.get(provider_id, provider_id)

    data = fetch_models_dev()
    pdata = data.get(mdev_id)
    if not isinstance(pdata, dict):
        return None

    models = pdata.get("models", {})
    if not isinstance(models, dict):
        return None

    # Exact match
    raw = models.get(model_id)
    if isinstance(raw, dict):
        return _parse_model_info(model_id, raw, mdev_id)

    # Case-insensitive fallback
    model_lower = model_id.lower()
    for mid, mdata in models.items():
        if mid.lower() == model_lower and isinstance(mdata, dict):
            return _parse_model_info(mid, mdata, mdev_id)

    return None
