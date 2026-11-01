"""DeepSeek chat client configured exclusively through the environment."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .provenance import api_call_metadata, safe_identifier, utc_now


class LLMNetworkError(RuntimeError):
    """A transient transport or API availability error."""


class LLMAPIError(RuntimeError):
    """A non-retryable API configuration or response error."""


@dataclass
class LLMResponse:
    text: str
    latency: float = 0.0
    finish_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt_tokens(self) -> int:
        return int(self.usage.get("prompt_tokens", 0) or 0)

    @property
    def completion_tokens(self) -> int:
        return int(self.usage.get("completion_tokens", 0) or 0)

    @property
    def total_tokens(self) -> int:
        return int(self.usage.get("total_tokens", self.prompt_tokens + self.completion_tokens) or 0)


class LLMClient:
    """A chat client, safe to share across evaluation worker threads.

    All roles use the DeepSeek API. API credentials are never written to logs
    or checkpoints.
    """

    def __init__(self, *, model: str, api_key: str, base_url: str,
                 timeout: float = 600.0, max_tokens: int | None = 32768,
                 json_mode: bool = True):
        if not api_key or api_key == "YOUR_API_KEY":
            raise ValueError("Set AUTOREM_API_KEY to your API credential.")
        parts = urlsplit(base_url)
        if parts.username or parts.password or parts.query or parts.fragment:
            raise ValueError("API base URL must not contain credentials, a query, or a fragment.")
        if parts.scheme != "https" and not (
            parts.scheme == "http" and parts.hostname in ("localhost", "127.0.0.1", "::1")
        ):
            raise ValueError("Use HTTPS for remote API endpoints.")
        if not parts.hostname or not model or timeout <= 0 or (max_tokens is not None and max_tokens < 1):
            raise ValueError("Invalid API client configuration.")
        self.model = model
        self._api_key = api_key
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.json_mode = json_mode
        self._metadata_logger = None

    def safe_model_identifier(self) -> str | None:
        return safe_identifier(self.model, secret=self._api_key)

    def safe_request_settings(self) -> dict:
        """Return only non-sensitive settings, without endpoint or credentials."""
        return {
            "temperature": 0,
            "stream": False,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"} if self.json_mode else None,
            "timeout_seconds": self.timeout,
        }

    def set_metadata_logger(self, logger):
        """Set an optional successful-response logger and return the prior one."""
        previous = self._metadata_logger
        self._metadata_logger = logger
        return previous

    @classmethod
    def from_env(cls, *, coder: bool = False) -> "LLMClient":
        prefix = "AUTOREM_CODER_" if coder else "AUTOREM_"
        def setting(name: str, default: str = "") -> str:
            return os.environ.get(prefix + name, default)
        base_url = setting("BASE_URL", os.environ.get("AUTOREM_BASE_URL", "https://api.deepseek.com/v1"))
        api_key = setting("API_KEY")
        # A single DeepSeek key can serve both roles at the same endpoint.
        base_endpoint = os.environ.get("AUTOREM_BASE_URL", "https://api.deepseek.com/v1")
        if coder and not api_key and base_url.rstrip("/") == base_endpoint.rstrip("/"):
            api_key = os.environ.get("AUTOREM_API_KEY", "")
        if not api_key:
            raise ValueError(f"Set {prefix}API_KEY for the selected API endpoint.")
        model = os.environ.get(prefix + "MODEL", "deepseek-v4-pro" if coder else "deepseek-v4-flash")
        max_tokens = setting("MAX_TOKENS", "" if coder else "32768")
        json_mode = setting("JSON_MODE", "false" if coder else "true").lower()
        if json_mode not in ("true", "false"):
            raise ValueError("JSON_MODE must be true or false.")
        return cls(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=float(setting("TIMEOUT", "600")),
            max_tokens=int(max_tokens) if max_tokens else None,
            json_mode=json_mode == "true",
        )

    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a JSON-only assistant that MUST reply with a single valid JSON object without extra text."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "temperature": 0,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        allowed = {"temperature", "max_tokens"}
        if set(kwargs) - allowed:
            raise ValueError("Unsupported completion options.")
        payload.update(kwargs)
        request = Request(self.url, data=json.dumps(payload).encode("utf-8"), headers={
            "Authorization": "Bearer " + self._api_key,
            "Content-Type": "application/json",
        }, method="POST")
        request_started_utc = utc_now()
        start = time.perf_counter()
        try:
            with urlopen(request, timeout=self.timeout) as response:
                data = json.load(response)
        except HTTPError as exc:
            status = exc.code
            exc.close()
            # Response bodies may echo credentials; do not expose them.
            if status in (408, 429) or status >= 500:
                raise LLMNetworkError(f"API temporarily unavailable (HTTP {status}).") from None
            raise LLMAPIError(f"API request rejected (HTTP {status}); check endpoint, model, and credentials.") from None
        except (URLError, TimeoutError, OSError):
            raise LLMNetworkError("API connection failed or timed out.") from None
        except (ValueError, UnicodeError):
            raise LLMAPIError("API returned an invalid JSON response.") from None
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
            if not isinstance(content, str):
                raise TypeError
            response_received_utc = utc_now()
            result = LLMResponse(text=content, latency=time.perf_counter() - start,
                                 finish_reason=choice.get("finish_reason"),
                                 usage=data.get("usage") or {})
        except (KeyError, IndexError, TypeError):
            raise LLMAPIError("API response does not contain a text completion.") from None
        if self._metadata_logger is not None:
            self._metadata_logger(api_call_metadata(
                request_started_utc=request_started_utc,
                response_received_utc=response_received_utc,
                requested_model=self.model, response=data, secret=self._api_key,
            ))
        return result
