"""OpenAI-compatible backend: response parsing, retries, throttling.

The network call itself is mocked, these tests pin the behaviour around it,
which is where the bugs actually live: rate-limit handling, backoff, and
failing loudly instead of returning an empty answer.
"""
import time
import types

import pytest

from sunrai_rag.config import Config, ConfigError, load_config
from sunrai_rag.rag.llm import (
    PROVIDER_PRESETS,
    OpenAICompatibleBackend,
    RateLimiter,
    ResponseCache,
    build_llm,
)


class _Response:
    def __init__(self, status=200, payload=None, headers=None, text=""):
        self.status_code = status
        self._payload = payload or {}
        self.headers = headers or {}
        self.text = text

    def json(self):
        return self._payload


def _ok(content="the answer"):
    return _Response(200, {"choices": [{"message": {"content": content}}]})


@pytest.fixture
def patched_requests(monkeypatch):
    """Install a fake `requests` module and record every call made."""
    calls = []

    def install(responses):
        queue = list(responses)

        def post(url, json=None, headers=None, timeout=None):
            calls.append({"url": url, "json": json, "headers": headers})
            return queue.pop(0) if queue else _ok()

        monkeypatch.setitem(
            __import__("sys").modules, "requests", types.SimpleNamespace(post=post)
        )
        return calls

    return install


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Backoff waits are asserted on, not actually slept through."""
    slept = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
    return slept


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key-123")


# ---------- happy path ----------

def test_returns_message_content(patched_requests):
    patched_requests([_ok("Paris is the capital.")])
    assert OpenAICompatibleBackend().complete("q?") == "Paris is the capital."


def test_sends_bearer_token_and_model(patched_requests):
    calls = patched_requests([_ok()])
    OpenAICompatibleBackend(model="llama-3.3-70b-versatile").complete("q?")
    assert calls[0]["headers"]["Authorization"] == "Bearer test-key-123"
    assert calls[0]["json"]["model"] == "llama-3.3-70b-versatile"


def test_system_prompt_becomes_first_message(patched_requests):
    calls = patched_requests([_ok()])
    OpenAICompatibleBackend().complete("q?", system="be terse")
    messages = calls[0]["json"]["messages"]
    assert messages[0] == {"role": "system", "content": "be terse"}
    assert messages[1]["role"] == "user"


def test_omits_system_message_when_absent(patched_requests):
    calls = patched_requests([_ok()])
    OpenAICompatibleBackend().complete("q?")
    assert [m["role"] for m in calls[0]["json"]["messages"]] == ["user"]


def test_temperature_zero_is_sent(patched_requests):
    calls = patched_requests([_ok()])
    OpenAICompatibleBackend(temperature=0.0).complete("q?")
    assert calls[0]["json"]["temperature"] == 0.0


def test_endpoint_path_is_correct(patched_requests):
    calls = patched_requests([_ok()])
    OpenAICompatibleBackend(base_url="https://api.groq.com/openai/v1/").complete("q?")
    assert calls[0]["url"] == "https://api.groq.com/openai/v1/chat/completions"


# ---------- rate limiting and retries ----------

def test_retries_on_429_then_succeeds(patched_requests, no_sleep):
    patched_requests([_Response(429, headers={}), _ok("recovered")])
    assert OpenAICompatibleBackend().complete("q?") == "recovered"
    assert no_sleep, "should have backed off before retrying"


def test_honours_retry_after_header(patched_requests, no_sleep):
    patched_requests([_Response(429, headers={"Retry-After": "7"}), _ok()])
    OpenAICompatibleBackend().complete("q?")
    assert 7.0 in no_sleep, f"should wait the server-specified 7s, slept {no_sleep}"


def test_malformed_retry_after_falls_back_to_backoff(patched_requests, no_sleep):
    patched_requests([_Response(429, headers={"Retry-After": "soon"}), _ok()])
    OpenAICompatibleBackend().complete("q?")
    assert no_sleep and all(isinstance(s, (int, float)) for s in no_sleep)


def test_backoff_grows_between_attempts(patched_requests, no_sleep):
    patched_requests([_Response(429), _Response(429), _Response(429), _ok()])
    OpenAICompatibleBackend().complete("q?")
    assert no_sleep[1] > no_sleep[0], f"backoff should grow: {no_sleep}"


def test_retries_on_server_error(patched_requests):
    patched_requests([_Response(503, text="upstream down"), _ok("fine")])
    assert OpenAICompatibleBackend().complete("q?") == "fine"


def test_gives_up_after_max_retries(patched_requests):
    patched_requests([_Response(429)] * 6)
    with pytest.raises(RuntimeError, match="Failed after 3 attempts"):
        OpenAICompatibleBackend(max_retries=3).complete("q?")


def test_client_error_fails_immediately(patched_requests):
    """A 400 will not fix itself; retrying just wastes the rate-limit budget."""
    calls = patched_requests([_Response(400, text="bad model name"), _ok()])
    with pytest.raises(RuntimeError, match="HTTP 400"):
        OpenAICompatibleBackend().complete("q?")
    assert len(calls) == 1


def test_network_error_is_retried(patched_requests, monkeypatch):
    attempts = {"n": 0}

    def flaky(url, json=None, headers=None, timeout=None):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise ConnectionError("dns failure")
        return _ok("second time lucky")

    monkeypatch.setitem(
        __import__("sys").modules, "requests", types.SimpleNamespace(post=flaky)
    )
    assert OpenAICompatibleBackend().complete("q?") == "second time lucky"


# ---------- failure modes that must not be silent ----------

def test_missing_api_key_names_the_variable(patched_requests, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    patched_requests([_ok()])
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        OpenAICompatibleBackend().complete("q?")


def test_unexpected_response_shape_raises(patched_requests):
    patched_requests([_Response(200, {"unexpected": "shape"})])
    with pytest.raises(RuntimeError, match="Unexpected response shape"):
        OpenAICompatibleBackend().complete("q?")


def test_empty_choices_returns_empty_not_crash(patched_requests):
    patched_requests([_Response(200, {"choices": []})])
    assert OpenAICompatibleBackend().complete("q?") == ""


# ---------- caching ----------

def test_cache_prevents_a_second_call(patched_requests, tmp_path):
    calls = patched_requests([_ok("cached value")])
    cache = ResponseCache(tmp_path / "c.json")
    backend = OpenAICompatibleBackend(cache=cache)
    assert backend.complete("same prompt") == "cached value"
    assert backend.complete("same prompt") == "cached value"
    assert len(calls) == 1, "second identical prompt must hit the cache"


def test_cache_key_separates_models(patched_requests, tmp_path):
    calls = patched_requests([_ok("a"), _ok("b")])
    cache = ResponseCache(tmp_path / "c.json")
    OpenAICompatibleBackend(model="m1", cache=cache).complete("p")
    OpenAICompatibleBackend(model="m2", cache=cache).complete("p")
    assert len(calls) == 2, "different models must not share cache entries"


# ---------- throttle ----------

def test_rate_limiter_paces_requests(no_sleep):
    limiter = RateLimiter(rpm=60)  # one per second
    limiter.wait()
    limiter.wait()
    assert no_sleep, "second immediate call should be paced"


def test_rate_limiter_can_be_disabled(no_sleep):
    limiter = RateLimiter(rpm=0)
    limiter.wait()
    limiter.wait()
    assert not no_sleep


# ---------- config integration ----------

@pytest.mark.parametrize("provider", sorted(PROVIDER_PRESETS))
def test_every_preset_builds(provider):
    cfg = Config()
    cfg.llm.backend = provider
    cfg.llm.model = ""
    cfg.llm.base_url = ""
    cfg.llm.api_key_env = ""
    backend = build_llm(cfg)
    assert backend.base_url and backend.model and backend.api_key_env


def test_explicit_config_overrides_preset():
    cfg = Config()
    cfg.llm.backend = "groq"
    cfg.llm.model = "llama-3.1-8b-instant"
    backend = build_llm(cfg)
    assert backend.model == "llama-3.1-8b-instant"
    assert backend.base_url == PROVIDER_PRESETS["groq"]["base_url"]


def test_custom_endpoint_supported():
    cfg = Config()
    cfg.llm.backend = "openai_compatible"
    cfg.llm.base_url = "http://localhost:8000/v1"
    cfg.llm.model = "my-local-model"
    cfg.llm.api_key_env = "LOCAL_KEY"
    assert build_llm(cfg).base_url == "http://localhost:8000/v1"


def test_custom_endpoint_requires_url(tmp_path):
    import yaml
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"llm": {"backend": "openai_compatible"}}))
    with pytest.raises(ConfigError, match="requires llm.base_url"):
        load_config(p)


def test_unknown_backend_lists_valid_options(tmp_path):
    import yaml
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"llm": {"backend": "gpt5-turbo-max"}}))
    with pytest.raises(ConfigError, match="groq"):
        load_config(p)


def test_negative_rpm_rejected(tmp_path):
    import yaml
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"llm": {"requests_per_minute": -1}}))
    with pytest.raises(ConfigError, match="requests_per_minute"):
        load_config(p)
