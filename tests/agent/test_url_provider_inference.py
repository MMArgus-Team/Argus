"""Hostname → provider inference must match strictly.

This lookup decides which credentials and which wire format an endpoint gets
(it backs both context-length resolution and
``providers.resolve_provider_profile``), so a false positive is not cosmetic.

It used to be a plain substring test (``if url_part in host``) — exactly the
class of bug ``utils.base_url_host_matches`` was written to prevent, and whose
docstring warns about it.
"""

import pytest

from agent.model_metadata import _infer_provider_from_url as infer


class TestExactVendors:
    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://api.openai.com/v1", "openai"),
            ("https://api.anthropic.com", "anthropic"),
            ("https://open.bigmodel.cn/api/paas/v4/", "zai"),
            ("https://api.z.ai/api/paas/v4", "zai"),
            ("https://api.moonshot.ai/v1", "kimi-coding"),
            ("https://api.moonshot.cn/v1", "kimi-coding-cn"),
            ("https://api.deepseek.com/v1", "deepseek"),
            ("https://openrouter.ai/api/v1", "openrouter"),
            ("https://ollama.com/v1", "ollama-cloud"),
        ],
    )
    def test_known_host(self, url, expected):
        assert infer(url) == expected

    def test_subdomains_still_match(self):
        """Vendors legitimately shard by region/subdomain."""
        assert infer("https://bedrock-runtime.us-east-1.amazonaws.com") == "bedrock"
        assert infer("https://token-plan-cn.xiaomimimo.com/v1") == "xiaomi"

    def test_most_specific_host_wins(self):
        """Longest key first, so dict order can't decide the answer."""
        assert infer("https://dashscope-intl.aliyuncs.com/compatible-mode/v1") == "alibaba"
        assert infer("https://dashscope.aliyuncs.com/compatible-mode/v1") == "alibaba"


class TestMiniMaxRegionSplit:
    """The two MiniMax regions are DIFFERENT providers with different keys.

    A single bare ``api.minimax`` entry plus substring matching meant the China
    host resolved to the international profile — and therefore looked for
    ``MINIMAX_API_KEY`` instead of ``MINIMAX_CN_API_KEY``.
    """

    def test_international(self):
        assert infer("https://api.minimax.io/anthropic") == "minimax"

    def test_china(self):
        assert infer("https://api.minimaxi.com/anthropic") == "minimax-cn"
        assert infer("https://api.minimaxi.com/v1") == "minimax-cn"


class TestLookalikeHostsRejected:
    """Each of these was a false positive under substring matching."""

    @pytest.mark.parametrize(
        "url",
        [
            # Suffix-appended lookalikes.
            "https://api.openai.com.evil.example/v1",
            "https://not-openrouter.ai.attacker.test/v1",
            "https://ollama.com.cn/v1",
            # Prefix-glued lookalike: `myopenrouter.ai` is not `openrouter.ai`.
            "https://myopenrouter.ai/v1",
            # Vendor host appearing only in the PATH.
            "https://evil.example/api.openai.com/v1",
        ],
    )
    def test_rejected(self, url):
        assert infer(url) is None


class TestNoMatch:
    @pytest.mark.parametrize("url", ["", None, "https://mystery.example/v1", "not a url"])
    def test_unknown_is_none(self, url):
        assert infer(url) is None

    def test_localhost_is_not_a_vendor(self):
        assert infer("http://localhost:11434/v1") is None
        assert infer("http://127.0.0.1:8000/v1") is None
