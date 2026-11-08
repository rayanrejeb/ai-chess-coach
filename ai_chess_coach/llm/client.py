"""The only module that calls the Gemini API (google-genai).

Every failure surfaces as LLMError with a `kind`:
  quota_exhausted  daily quota used up; not retried
  rate_limited     429 still failing after retries
  unavailable      5xx, timeout or connection error still failing after retries
  model_not_found  404: GEMINI_MODEL names a model this key cannot use; not retried
  bad_request      other 4xx; not retried
  blocked          the model returned nothing (safety or other finish reason), twice
  bad_response     the answer did not match the schema, twice
"""

import logging
import os
import random
import re
import threading
import time
from typing import Callable, Optional, Sequence, TypeVar

import httpx2  # google-genai's HTTP transport
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError

log = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.5-flash"
MAX_API_ATTEMPTS = 4          # for 429 / 5xx
MAX_CONTENT_ATTEMPTS = 2      # for empty, blocked or malformed answers
BASE_DELAY_S = 2.0
MAX_DELAY_S = 60.0
MAX_TOOL_CALLS = 5
REQUEST_TIMEOUT_S = 120     # per HTTP request; without it a stalled request hangs forever

# Chess vocabulary ("attack", "kill", "sacrifice") must not trip safety filters.
_SAFETY = [
    types.SafetySetting(category=c, threshold=types.HarmBlockThreshold.BLOCK_NONE)
    for c in (
        types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    )
]

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def model_from_env(default: str = DEFAULT_MODEL) -> str:
    return os.environ.get("GEMINI_MODEL") or default


def _error_details(err: errors.APIError) -> list:
    body = err.details if isinstance(err.details, dict) else {}
    return body.get("error", {}).get("details", []) or []


def classify_api_error(err: errors.APIError) -> tuple[str, Optional[float]]:
    """(kind, server-suggested retry delay in seconds or None)."""
    if err.code == 429:
        details = _error_details(err)
        quota_ids = [v.get("quotaId", "") for d in details for v in d.get("violations", [])]
        if any("PerDay" in q for q in quota_ids) or re.search(r"per day", err.message or "", re.IGNORECASE):
            return "quota_exhausted", None
        delays = [d.get("retryDelay", "") for d in details if "retryDelay" in d]
        match = re.fullmatch(r"(\d+(?:\.\d+)?)s", delays[0]) if delays else None
        return "rate_limited", float(match.group(1)) if match else None
    if err.code == 404:
        return "model_not_found", None
    if isinstance(err, errors.ServerError):
        return "unavailable", None
    return "bad_request", None


class GeminiClient:
    def __init__(self, api_key: str, model: Optional[str] = None, *, sdk_client=None, sleep: Callable[[float], None] = time.sleep):
        self.model = model or model_from_env()
        self._sdk = sdk_client or genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_S * 1000))
        self._sleep = sleep
        # One request at a time, so chat and commentary never overlap and trip the per-minute rate limit.
        # Held through retry waits too, so nothing is sent while backing off from a 429.
        self._lock = threading.Lock()

    def _call_api(self, fn: Callable[[], object]):
        """Run one API call, retrying 429 and 5xx with backoff. Calls from other threads wait their turn."""
        with self._lock:
            return self._call_api_locked(fn)

    def _call_api_locked(self, fn: Callable[[], object]):
        for attempt in range(1, MAX_API_ATTEMPTS + 1):
            try:
                return fn()
            except (errors.APIError, httpx2.TransportError) as err:  # TransportError: timeouts, connection errors
                kind, delay = classify_api_error(err) if isinstance(err, errors.APIError) else ("unavailable", None)
                if kind == "model_not_found":
                    raise LLMError(kind, f"Gemini model '{self.model}' is not available for this API key. Change GEMINI_MODEL in .env. ({err})") from err
                if kind in ("quota_exhausted", "bad_request") or attempt == MAX_API_ATTEMPTS:
                    raise LLMError(kind, str(err)) from err
                wait = delay if delay is not None else min(MAX_DELAY_S, BASE_DELAY_S * 2 ** (attempt - 1)) * random.uniform(0.8, 1.2)
                log.warning("Gemini %s (attempt %d/%d), retrying in %.1fs", kind, attempt, MAX_API_ATTEMPTS, wait)
                self._sleep(wait)

    @staticmethod
    def _why_empty(response) -> str:
        feedback = getattr(response, "prompt_feedback", None)
        if feedback is not None and getattr(feedback, "block_reason", None):
            return f"prompt blocked: {feedback.block_reason}"
        candidates = getattr(response, "candidates", None) or []
        reason = getattr(candidates[0], "finish_reason", None) if candidates else None
        return f"no content (finish reason: {reason})"

    def generate_json(self, prompt: str, schema: type[T], system: Optional[str] = None) -> T:
        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=schema,
            safety_settings=_SAFETY,
        )
        problem = None
        for _ in range(MAX_CONTENT_ATTEMPTS):
            response = self._call_api(lambda: self._sdk.models.generate_content(model=self.model, contents=prompt, config=config))
            parsed = getattr(response, "parsed", None)
            if isinstance(parsed, schema):
                return parsed
            if getattr(response, "text", None):
                try:
                    return schema.model_validate_json(response.text)
                except ValidationError as err:
                    problem = ("bad_response", f"answer does not match the schema: {err.error_count()} errors")
            else:
                problem = ("blocked", self._why_empty(response))
            log.warning("Gemini %s: %s", *problem)
        raise LLMError(*problem)

    def chat(self, system: str, history: Sequence[types.Content], message: str, tools: Sequence[Callable] = ()) -> str:
        """One chat turn. Python functions in `tools` are called automatically by the SDK."""
        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=list(tools) or None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(maximum_remote_calls=MAX_TOOL_CALLS),
            safety_settings=_SAFETY,
        )
        problem = None
        for _ in range(MAX_CONTENT_ATTEMPTS):
            # A fresh chat per attempt, so a failed attempt leaves no half-finished tool calls behind.
            def send():
                session = self._sdk.chats.create(model=self.model, config=config, history=list(history))
                return session.send_message(message)

            response = self._call_api(send)
            if getattr(response, "text", None):
                return response.text
            problem = ("blocked", self._why_empty(response))
            log.warning("Gemini %s: %s", *problem)
        raise LLMError(*problem)
