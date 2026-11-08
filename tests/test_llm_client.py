import threading
import time
from pathlib import Path

import httpx2
import pytest
from google import genai
from google.genai import _transformers, types

from ai_chess_coach.llm.chat import EngineTool
from ai_chess_coach.llm.client import DEFAULT_MODEL, REQUEST_TIMEOUT_S, GeminiClient, LLMError, classify_api_error, model_from_env
from ai_chess_coach.llm.commentary import CommentaryBatch, MoveComment
from fakes import FakeSDK, api_error, fake_analyse, response

PER_DAY = {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}
PER_MINUTE = {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel"}]}
RETRY_37S = {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "37s"}

BATCH = CommentaryBatch(comments=[MoveComment(ply=1, move_san="e4", move_commentary="Good start.")])


def make(outcomes):
    sleeps = []
    sdk = FakeSDK(outcomes)
    return GeminiClient("fake-key", "test-model", sdk_client=sdk, sleep=sleeps.append), sdk, sleeps


def test_model_from_env(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-from-env")
    assert model_from_env() == "gemini-from-env"
    assert GeminiClient("k", sdk_client=FakeSDK([])).model == "gemini-from-env"
    monkeypatch.delenv("GEMINI_MODEL")
    assert model_from_env("fallback") == "fallback"


@pytest.mark.parametrize("err, kind, delay", [
    (api_error(429, "Quota exceeded", [PER_DAY]), "quota_exhausted", None),
    (api_error(429, "Too many requests", [PER_MINUTE, RETRY_37S]), "rate_limited", 37.0),
    (api_error(429, "Too many requests"), "rate_limited", None),
    (api_error(503), "unavailable", None),
    (api_error(504, "Deadline expired before operation could complete."), "too_slow", None),
    (api_error(400, "Invalid argument"), "bad_request", None),
    (api_error(404, "models/x is not found"), "model_not_found", None),
])
def test_classify_api_error(err, kind, delay):
    assert classify_api_error(err) == (kind, delay)


def test_generate_json_uses_parsed_object():
    client, sdk, _ = make([response(text="{}", parsed=BATCH)])
    assert client.generate_json("prompt", CommentaryBatch, system="sys") is BATCH
    config = sdk.calls[0]["config"]
    assert config.response_schema is CommentaryBatch and config.system_instruction == "sys"
    assert sdk.calls[0]["model"] == "test-model"


def test_generate_json_falls_back_to_text():
    client, _, _ = make([response(text=BATCH.model_dump_json())])
    assert client.generate_json("p", CommentaryBatch) == BATCH


def test_generate_json_bad_answer_retried_once_then_fails():
    client, sdk, _ = make([response(text='{"oops": 1}'), response(text="not json")])
    with pytest.raises(LLMError) as exc:
        client.generate_json("p", CommentaryBatch)
    assert exc.value.kind == "bad_response" and len(sdk.calls) == 2


def test_generate_json_empty_answer_is_blocked():
    client, _, _ = make([response(finish_reason="SAFETY"), response(finish_reason="SAFETY")])
    with pytest.raises(LLMError) as exc:
        client.generate_json("p", CommentaryBatch)
    assert exc.value.kind == "blocked" and "SAFETY" in str(exc.value)


def test_rate_limit_waits_server_delay_then_succeeds():
    client, _, sleeps = make([api_error(429, "slow down", [RETRY_37S]), response(parsed=BATCH)])
    assert client.generate_json("p", CommentaryBatch) is BATCH
    assert sleeps == [37.0]


def test_daily_quota_is_not_retried():
    client, sdk, sleeps = make([api_error(429, "Quota exceeded", [PER_DAY])])
    with pytest.raises(LLMError) as exc:
        client.generate_json("p", CommentaryBatch)
    assert exc.value.kind == "quota_exhausted" and sleeps == [] and len(sdk.calls) == 1


def test_server_errors_back_off_then_give_up():
    client, sdk, sleeps = make([api_error(503)] * 4)
    with pytest.raises(LLMError) as exc:
        client.generate_json("p", CommentaryBatch)
    assert exc.value.kind == "unavailable" and len(sdk.calls) == 4
    assert len(sleeps) == 3 and sleeps[0] < sleeps[1] < sleeps[2]


def test_bad_request_is_not_retried():
    client, sdk, sleeps = make([api_error(400, "Invalid argument")])
    with pytest.raises(LLMError) as exc:
        client.generate_json("p", CommentaryBatch)
    assert exc.value.kind == "bad_request" and len(sdk.calls) == 1 and sleeps == []


def test_chat_passes_system_history_and_tools():
    client, sdk, _ = make([response(text="Play Nf3.")])
    tool = EngineTool(fake_analyse).as_function()
    history = [types.Content(role="user", parts=[types.Part(text="hi")])]
    assert client.chat("system rules", history, "question?", tools=[tool]) == "Play Nf3."
    call = sdk.calls[0]
    assert call["config"].system_instruction == "system rules"
    assert call["config"].tools == [tool]
    assert call["history"] == history and call["message"] == "question?"


def test_real_sdk_accepts_schema_and_tool():
    """The SDK must convert our schema and tool for the Developer API (no network call)."""
    api_client = genai.Client(api_key="fake")._api_client
    schema = _transformers.t_schema(api_client, CommentaryBatch)
    assert schema.required == ["comments"]
    decl = types.FunctionDeclaration.from_callable(client=api_client, callable=EngineTool(fake_analyse).as_function())
    assert decl.name == "evaluate_position"
    assert set(decl.parameters.properties) == {"fen", "move_san", "move_number"} and not decl.parameters.required


def test_unknown_model_is_not_retried_and_names_the_setting():
    client, sdk, sleeps = make([api_error(404, "models/test-model is not found")])
    with pytest.raises(LLMError) as exc:
        client.generate_json("p", CommentaryBatch)
    assert exc.value.kind == "model_not_found" and len(sdk.calls) == 1 and sleeps == []
    assert "GEMINI_MODEL" in str(exc.value) and "test-model" in str(exc.value)


def test_timeouts_are_retried():
    client, sdk, sleeps = make([httpx2.ReadTimeout("stalled"), response(parsed=BATCH)])
    assert client.generate_json("p", CommentaryBatch) == BATCH
    assert len(sdk.calls) == 2 and len(sleeps) == 1


def test_real_client_sets_a_request_timeout():
    client = GeminiClient("fake-key", "m")
    assert client._sdk._api_client._http_options.timeout == REQUEST_TIMEOUT_S * 1000


def test_env_example_uses_the_default_model():
    env_example = (Path(__file__).resolve().parent.parent / ".env.example").read_text(encoding="utf-8")
    assert f"GEMINI_MODEL={DEFAULT_MODEL}\n" in env_example


def test_requests_never_overlap():
    """Chat and commentary run on different threads; the client sends one request at a time."""
    active, peak = [0], [0]

    def slow(*_):
        active[0] += 1
        peak[0] = max(peak[0], active[0])
        time.sleep(0.05)
        active[0] -= 1
        return response(parsed=BATCH)

    client, sdk, _ = make([slow] * 4)
    threads = [threading.Thread(target=client.generate_json, args=("p", CommentaryBatch)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(sdk.calls) == 4 and peak[0] == 1


def test_deadline_is_retried_once():
    client, sdk, sleeps = make([api_error(504, "Deadline expired"), response(parsed=BATCH)])
    assert client.generate_json("p", CommentaryBatch) == BATCH
    assert len(sdk.calls) == 2 and len(sleeps) == 1


@pytest.mark.parametrize("first", [api_error(504, "Deadline expired"), httpx2.ReadTimeout("stalled")])
def test_second_deadline_miss_says_the_model_is_too_slow(first):
    client, sdk, sleeps = make([first, api_error(504, "Deadline expired"), response(parsed=BATCH)])
    with pytest.raises(LLMError) as exc:
        client.generate_json("p", CommentaryBatch)
    assert exc.value.kind == "too_slow" and len(sdk.calls) == 2 and len(sleeps) == 1
    assert "too slow" in str(exc.value) and "GEMINI_MODEL" in str(exc.value)


def test_timeout_is_60_seconds():
    assert REQUEST_TIMEOUT_S == 60
