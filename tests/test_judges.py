"""Protocol and accounting tests use fake transports, never a live API."""
from io import BytesIO
import json
from urllib.error import HTTPError, URLError

import pytest

from prism.judges import APIJudge, judge_settings, make_judge, requires_gpu


def config():
    return {"execution": {"allow_paid_api_calls": True}, "judges": {
        "training": {"backend": "deepseek", "model": "deepseek-flash",
                     "base_url": "https://api.deepseek.com/v1", "api": "chat_completions",
                     "api_key_env": "PRISM_TEST_DEEPSEEK_KEY", "thinking": "disabled",
                     "temperature": 1.0, "max_output_tokens": 64},
        "evaluation": {"backend": "openai", "model": "gpt-6-luna",
                       "base_url": "https://api.openai.com/v1", "api": "responses",
                       "api_key_env": "PRISM_TEST_OPENAI_KEY", "reasoning_effort": "low",
                       "max_output_tokens": 1024}}}


def test_deepseek_payload_orientation_and_usage(monkeypatch):
    monkeypatch.setenv("PRISM_TEST_DEEPSEEK_KEY", "TEST_ONLY_NOT_A_REAL_KEY")
    seen = []
    def transport(url, payload, api_key, timeout):
        seen.append((url, payload, timeout))
        return {"id": "response-1", "model": "served-version", "system_fingerprint": "fp",
                "choices": [{"message": {"content": "A"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 250, "completion_tokens": 1,
                          "prompt_cache_hit_tokens": 10, "arbitrary_string": "ignored"}}, "request-1"
    judge = make_judge(config(), transport=transport)
    result = judge.call([{"role": "user", "content": "question"}], "canonical first", "canonical second", 77, True)
    url, payload, timeout = seen[0]
    assert url == "https://api.deepseek.com/v1/chat/completions"
    assert payload["thinking"] == {"type": "disabled"} and payload["temperature"] == 1.0
    assert payload["max_tokens"] == 64 and "seed" not in payload and "tools" not in payload
    displayed = json.loads(payload["messages"][1]["content"])
    assert displayed["response_A"] == "canonical second"
    assert result["judge_protocol"] == "preference_ab_task_boundary_v2"
    assert result["label"] == 0 and result["model"] == "served-version"
    assert result["request_id"] == "request-1" and result["response_id"] == "response-1"
    assert result["usage"] == {"prompt_tokens": 250, "completion_tokens": 1, "prompt_cache_hit_tokens": 10}
    assert result["seed_scope"] == "client_ordering_only" and timeout == 60


def test_multiple_choice_conversation_is_preserved_as_comparison_data(monkeypatch):
    monkeypatch.setenv("PRISM_TEST_DEEPSEEK_KEY", "TEST_ONLY_NOT_A_REAL_KEY")
    messages = [{"role": "user", "content": "Choose the correct option: (A) 1 (B) 2 (C) 3 (D) 4"}]
    def transport(url, payload, key, timeout):
        data = json.loads(payload["messages"][1]["content"])
        assert data["conversation"] == messages
        assert data["response_A"] == "(D)" and data["response_B"] == "(C)"
        assert list(data)[-1] == "evaluation_instruction"
        assert "not multiple-choice options" in data["evaluation_instruction"]
        assert "Do not answer the conversation yourself" in data["evaluation_instruction"]
        assert payload["max_tokens"] == 64 and payload["temperature"] == 1
        return {"choices": [{"message": {"content": "A"}, "finish_reason": "stop"}]}, None
    result = make_judge(config(), transport=transport).call(messages, "(D)", "(C)", 5, False)
    assert result["label"] == 1


def test_openai_responses_reads_final_text_not_reasoning_and_accepts_tie(monkeypatch):
    monkeypatch.setenv("PRISM_TEST_OPENAI_KEY", "TEST_ONLY_NOT_A_REAL_KEY")
    seen = []
    def transport(url, payload, api_key, timeout):
        seen.append((url, payload))
        return {"id": "response-2", "model": "gpt-6-luna", "status": "completed",
                "output": [{"type": "reasoning", "summary": [{"text": "A"}]},
                           {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "TIE\n"}]}],
                "usage": {"input_tokens": 800, "output_tokens": 90,
                          "output_tokens_details": {"reasoning_tokens": 88}}}, "request-2"
    result = make_judge(config(), "evaluation", transport).call([], "a", "b", 5, True)
    url, payload = seen[0]
    assert url == "https://api.openai.com/v1/responses"
    assert payload["reasoning"] == {"effort": "low"} and payload["max_output_tokens"] == 1024
    assert payload["store"] is False and "temperature" not in payload and "seed" not in payload
    assert "TIE" in payload["input"][0]["content"]
    assert result["label"] == .5 and result["usage"]["output_tokens_details"]["reasoning_tokens"] == 88


@pytest.mark.parametrize("text,finish_reason,error", [("A because better", "stop", "invalid_judgment_format"),
                                                      ("TIE", "stop", "invalid_judgment_format"),
                                                      ("A", "length", "incomplete_response")])
def test_training_format_and_finish_failures_keep_usage(monkeypatch, text, finish_reason, error):
    monkeypatch.setenv("PRISM_TEST_DEEPSEEK_KEY", "TEST_ONLY_NOT_A_REAL_KEY")
    def transport(*args):
        return {"choices": [{"message": {"content": text}, "finish_reason": finish_reason}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 64}}, None
    result = make_judge(config(), transport=transport).call([], "a", "b", 0, False)
    assert result["label"] is None and result["error"]["type"] == error
    assert result["usage"]["completion_tokens"] == 64


def test_paid_guard_and_missing_key_precede_transport(monkeypatch):
    calls = []
    def transport(*args):
        calls.append(args)
        raise AssertionError("must not be called")
    c = config()
    c["execution"]["allow_paid_api_calls"] = False
    judge = make_judge(c, transport=transport)
    with pytest.raises(RuntimeError, match="Paid API calls are disabled"):
        judge.call([], "a", "b", 0, False)
    assert not calls
    c["execution"]["allow_paid_api_calls"] = True
    monkeypatch.delenv("PRISM_TEST_DEEPSEEK_KEY", raising=False)
    with pytest.raises(RuntimeError, match="environment variable"):
        judge.call([], "a", "b", 0, False)
    assert not calls


def test_remote_errors_never_log_key_or_provider_body(monkeypatch):
    fake_key = "TEST_ONLY_NOT_A_REAL_KEY"
    monkeypatch.setenv("PRISM_TEST_DEEPSEEK_KEY", fake_key)
    def transport(*args):
        raise HTTPError(args[0], 429, fake_key, {"x-request-id": "request-err"}, BytesIO(fake_key.encode()))
    result = make_judge(config(), transport=transport).call([], "a", "b", 0, False)
    assert result["error"] == {"type": "http_error", "status": 429, "retryable": True}
    assert fake_key not in json.dumps(result) and result["request_id"] == "request-err"
    def network(*args):
        raise URLError(fake_key)
    result = make_judge(config(), transport=network).call([], "a", "b", 0, False)
    assert result["error"]["type"] == "network_error" and fake_key not in json.dumps(result)


def test_legacy_factory_keeps_lazy_hf_interface(monkeypatch):
    import prism.hf
    class LegacyJudge:
        def __init__(self, c):
            self.c = c
    monkeypatch.setattr(prism.hf, "Judge", LegacyJudge)
    c = {"judge": {"max_new_tokens": 32, "temperature": 1.0, "max_attempts": 3}}
    assert isinstance(make_judge(c), LegacyJudge)
    assert requires_gpu(c) and not requires_gpu(config())
    assert judge_settings(c)["max_output_tokens"] == 32


def test_transport_configuration_rejects_url_credentials():
    c = config()
    c["judges"]["training"]["base_url"] = "https://private:secret@example.invalid/v1"
    with pytest.raises(ValueError, match="without credentials"):
        APIJudge(c)
    c = config()
    c["judges"]["training"]["api"] = "responses"
    with pytest.raises(ValueError, match="Implemented API backends"):
        APIJudge(c)


def test_non_object_json_is_a_logged_format_failure(monkeypatch):
    monkeypatch.setenv("PRISM_TEST_DEEPSEEK_KEY", "TEST_ONLY_NOT_A_REAL_KEY")
    result = make_judge(config(), transport=lambda *args: ([], None)).call([], "a", "b", 0, False)
    assert result["label"] is None and result["error"]["type"] == "malformed_response"
